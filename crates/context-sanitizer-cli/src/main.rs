//! Context Sanitizer standalone CLI binary (`context-sanitizer xray`).

pub mod bench;
pub mod xray;

use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::{Args, Parser, Subcommand};
use context_sanitizer::arena::SpanArena;
use context_sanitizer::compress::compress_text_with_arena;
use context_sanitizer::guard::run_tier1_guard;
use context_sanitizer::wire::sanitize_and_compress_chat_completions_json;
use context_sanitizer::{SanitizerError, ThreatGatekeeper, VaultHandle};
use xray::XRayReport;

#[derive(Parser, Debug)]
#[command(name = "context-sanitizer")]
#[command(
    about = "Context Sanitizer: Pure-Rust System One semantic firewall and syntax-safe context compressor."
)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Context X-Ray diagnostic analysis, economics estimation, and benchmark runner.
    Xray(XrayArgs),
}

#[derive(Args, Debug)]
pub struct XrayArgs {
    /// Path to input payload file, or "-" to read from standard input.
    #[arg(value_name = "PAYLOAD")]
    pub payload: Option<PathBuf>,

    /// Target statistical context compression rate (0.40 - 0.70).
    #[arg(short, long, default_value_t = 0.60)]
    pub rate: f32,

    /// Threat gatekeeper sensitivity threshold (0.0 - 1.0).
    #[arg(long, default_value_t = 0.85)]
    pub threat_threshold: f32,

    /// Execute automated micro-benchmark harness and nanosecond SLA verification.
    #[arg(long)]
    pub bench: bool,

    /// Output diagnostic telemetry in machine-readable JSON format.
    #[arg(long)]
    pub json: bool,

    /// Optional path to write sanitized and compressed output payload.
    #[arg(short, long, value_name = "OUTPUT")]
    pub output: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Commands::Xray(args) => run_xray(args),
    }
}

fn run_xray(args: XrayArgs) -> ExitCode {
    if args.bench {
        return execute_benchmarks(args.json);
    }

    let payload_path = match &args.payload {
        Some(p) => p,
        None => {
            eprintln!(
                "Error: Target payload file or '-' is required when not running with --bench"
            );
            return ExitCode::FAILURE;
        }
    };

    let raw_input = match read_payload(payload_path) {
        Ok(content) => content,
        Err(err) => {
            eprintln!("Error reading payload: {err}");
            return ExitCode::FAILURE;
        }
    };

    let (output_text, report) = analyze_payload(&raw_input, args.rate, args.threat_threshold);

    if !report.is_blocked {
        if let Some(out_path) = &args.output {
            if let Err(err) = fs::write(out_path, &output_text) {
                eprintln!("Error writing output payload: {err}");
                return ExitCode::FAILURE;
            }
        }
    }

    if args.json {
        match serde_json::to_string_pretty(&report) {
            Ok(json_str) => println!("{json_str}"),
            Err(err) => {
                eprintln!("Error serializing JSON report: {err}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        println!("{}", report.render_ascii_dashboard());
    }

    if report.is_blocked {
        eprintln!("Threat Gatekeeper Error: Payload blocked by Tier 3 threat gatekeeper.");
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn read_payload(path: &PathBuf) -> io::Result<String> {
    if path.as_os_str() == "-" {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;
        Ok(buf)
    } else {
        fs::read_to_string(path)
    }
}

fn analyze_payload(raw: &str, rate: f32, threat_threshold: f32) -> (String, XRayReport) {
    let vault = VaultHandle::new();

    // 1. Configure Tier 3 Threat Gatekeeper
    #[allow(unused_mut)]
    let mut gatekeeper = ThreatGatekeeper::new().with_threat_threshold(threat_threshold);

    #[cfg(feature = "ml")]
    {
        let candidates = [
            (
                "crates/context-sanitizer/tests/fixtures/ml/micro_model.safetensors",
                "crates/context-sanitizer/tests/fixtures/ml/tokenizer.json",
            ),
            (
                "tests/fixtures/ml/micro_model.safetensors",
                "tests/fixtures/ml/tokenizer.json",
            ),
        ];
        for (wp, tp) in candidates {
            if std::path::Path::new(wp).exists() && std::path::Path::new(tp).exists() {
                if let Ok(guard) = context_sanitizer::CandleGuard::from_files(wp, tp) {
                    gatekeeper = gatekeeper.with_guard(guard);
                    break;
                }
            }
        }
    }

    // 2. Evaluate threat
    let gk_eval = gatekeeper.evaluate_threat(raw);
    let (is_blocked, threat_verdict, max_threat_score, evaluated_windows, tier3_latency_ns) =
        match gk_eval {
            Ok(res) => (
                false,
                "PASS".to_owned(),
                res.max_threat_score,
                res.evaluated_windows,
                res.latency_ns,
            ),
            Err(SanitizerError::ThreatBlocked {
                score_bps,
                reason: _,
            }) => (
                true,
                "BLOCKED".to_owned(),
                (score_bps as f32) / 10_000.0,
                0,
                0,
            ),
            Err(_) => (false, "PASS".to_owned(), 0.0, 0, 0),
        };

    if is_blocked {
        let original_tokens = raw.len().div_ceil(4);
        let report = XRayReport::new(
            raw.len(),
            raw.len(),
            original_tokens,
            original_tokens,
            0,
            0,
            tier3_latency_ns,
            evaluated_windows,
            max_threat_score,
            threat_threshold,
            threat_verdict,
            true,
            0,
            0,
            true,
            false,
        );
        return (raw.to_owned(), report);
    }

    let is_chat_json = raw.trim().starts_with('{')
        && serde_json::from_str::<serde_json::Value>(raw)
            .map(|v| v.get("messages").is_some())
            .unwrap_or(false);

    if is_chat_json {
        let t1_start = Instant::now();
        // Run Tier 1 + Tier 2 via wire compressor
        match sanitize_and_compress_chat_completions_json(raw, &vault, rate) {
            Ok((compressed, stats)) => {
                let tier1_ns = t1_start.elapsed().as_nanos() as u64;
                let report = XRayReport::new(
                    stats.original_bytes,
                    stats.compressed_bytes,
                    stats.original_tokens,
                    stats.compressed_tokens,
                    tier1_ns.saturating_sub(stats.elapsed_ns),
                    stats.elapsed_ns,
                    tier3_latency_ns,
                    evaluated_windows,
                    max_threat_score,
                    threat_threshold,
                    threat_verdict,
                    false,
                    compressed.matches("[REDACTED_CREDENTIAL]").count(),
                    vault.len(),
                    !stats.fail_open_triggered,
                    stats.fail_open_triggered,
                );
                (compressed, report)
            }
            Err(_) => analyze_as_raw_text(
                raw,
                rate,
                threat_threshold,
                tier3_latency_ns,
                evaluated_windows,
                max_threat_score,
                &threat_verdict,
                &vault,
            ),
        }
    } else {
        analyze_as_raw_text(
            raw,
            rate,
            threat_threshold,
            tier3_latency_ns,
            evaluated_windows,
            max_threat_score,
            &threat_verdict,
            &vault,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn analyze_as_raw_text(
    raw: &str,
    rate: f32,
    threat_threshold: f32,
    tier3_latency_ns: u64,
    evaluated_windows: usize,
    max_threat_score: f32,
    threat_verdict: &str,
    vault: &VaultHandle,
) -> (String, XRayReport) {
    let mut arena = SpanArena::new(raw);
    let t1_start = Instant::now();
    run_tier1_guard(&mut arena, vault);
    let tier1_ns = t1_start.elapsed().as_nanos() as u64;

    let t2_start = Instant::now();
    let (compressed, stats) = compress_text_with_arena(&mut arena, rate, None);
    let tier2_ns = t2_start.elapsed().as_nanos() as u64;

    let redacted_count = arena
        .spans()
        .iter()
        .filter(|s| matches!(s.kind, context_sanitizer::SpanKind::Redacted(_)))
        .count();

    let report = XRayReport::new(
        stats.original_bytes,
        stats.compressed_bytes,
        stats.original_tokens,
        stats.compressed_tokens,
        tier1_ns,
        tier2_ns,
        tier3_latency_ns,
        evaluated_windows,
        max_threat_score,
        threat_threshold,
        threat_verdict.to_owned(),
        false,
        redacted_count,
        vault.len(),
        !stats.fail_open_triggered,
        stats.fail_open_triggered,
    );

    (compressed, report)
}

fn execute_benchmarks(as_json: bool) -> ExitCode {
    println!("Running Context Sanitizer automated micro-benchmarks...\n");
    let (results, all_passed) = bench::run_all_benchmarks();

    if as_json {
        let json_results: Vec<_> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "corpus": r.corpus_name,
                    "payload_bytes": r.payload_bytes,
                    "iterations": r.iterations,
                    "p50_ns": r.p50_ns,
                    "p99_ns": r.p99_ns,
                    "throughput_mb_per_sec": r.throughput_mb_per_sec,
                    "sla_passed": r.sla_passed,
                })
            })
            .collect();
        match serde_json::to_string_pretty(&json_results) {
            Ok(s) => println!("{s}"),
            Err(e) => eprintln!("Error formatting JSON: {e}"),
        }
    } else {
        println!("{:=<90}", "");
        println!(
            "{:<35} | {:>8} | {:>12} | {:>12} | {:>10} | {:>6}",
            "Corpus", "Bytes", "P50 (ns)", "P99 (ns)", "MB/sec", "SLA"
        );
        println!("{:-<90}", "");
        for r in &results {
            let sla_str = if r.sla_passed { "PASS" } else { "FAIL" };
            println!(
                "{:<35} | {:>8} | {:>12} | {:>12} | {:>10.2} | {:>6}",
                r.corpus_name,
                r.payload_bytes,
                r.p50_ns,
                r.p99_ns,
                r.throughput_mb_per_sec,
                sla_str
            );
        }
        println!("{:=<90}", "");
    }

    if all_passed {
        println!("\nAll nanosecond SLA gates PASSED (P99 < 1,500,000 ns).");
        ExitCode::SUCCESS
    } else {
        eprintln!("\nOne or more SLA gates FAILED.");
        ExitCode::FAILURE
    }
}
