//! Deterministic micro-benchmark harness and SLA assertion runner (`FR-12`, `NFR-1`).

use std::time::Instant;

use context_sanitizer::compress::compress_text;

/// Maximum allowed P99 latency SLA for 4KB payloads in nanoseconds ($< 1,500,000$ ns in release mode).
#[cfg(not(debug_assertions))]
pub const MAX_P99_LATENCY_SLA_NS: u64 = 1_500_000;

#[cfg(debug_assertions)]
pub const MAX_P99_LATENCY_SLA_NS: u64 = 60_000_000;

/// Standard MeetingBank conversation corpus sample.
pub const MEETINGBANK_SAMPLE: &str = concat!(
    "Speaker 1: Welcome everyone to the quarterly infrastructure sync. Today we need to discuss our cloud expenditure and cluster migration.\n",
    "Speaker 2: Looking at the metrics, our primary database nodes in us-east-1 are experiencing occasional latency spikes during peak ingest.\n",
    "Speaker 1: Have we identified whether the bottleneck is storage IOPS or network egress between the VPC peering connections?\n",
    "Speaker 3: We analyzed the Prometheus dashboards. Disk latency remained below 3ms, but container memory usage climbed steadily over 48 hours.\n",
    "Speaker 2: That suggests a connection pool leak in the background worker pods. We should patch the connection reaper setting.\n",
    "Speaker 1: Agreed. Let's schedule the configuration rollout for tomorrow morning during the maintenance window.\n"
);

/// Standard GSM8K mathematical reasoning corpus sample.
pub const GSM8K_SAMPLE: &str = concat!(
    "Question: Janet buys 3 boxes of markers. Each box contains 12 markers. She gives 5 markers to her brother and 7 markers to her friend. ",
    "Then, she buys 2 more packs of markers with 8 markers in each pack. How many markers does Janet have in total at the end?\n",
    "Reasoning: First, Janet starts with 3 boxes * 12 markers = 36 markers. ",
    "Next, she gives away 5 + 7 = 12 markers. So she has 36 - 12 = 24 markers left. ",
    "Then, she buys 2 packs * 8 markers = 16 additional markers. ",
    "Finally, 24 + 16 = 40 markers. Answer: 40.\n"
);

/// Synthetic credentials and operational PII corpus sample.
pub const SYNTHETIC_SECURITY_SAMPLE: &str = concat!(
    "User profile update event:\n",
    "Customer contact: alice.smith@acme-corp.com, phone: +1-555-0199, ip: 192.168.1.50\n",
    "Authorization config:\n",
    "aws_access_key = \"AKIAIOSFODNN7EXAMPLE\"\n",
    "aws_secret_key = \"wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\"\n",
    "database_url = \"postgres://admin:super_secret_password_12345@db.internal:5432/main\"\n"
);

/// Summary statistics for a benchmark corpus run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub corpus_name: &'static str,
    pub iterations: usize,
    pub payload_bytes: usize,
    pub p50_ns: u64,
    pub p99_ns: u64,
    pub throughput_mb_per_sec: f64,
    pub sla_passed: bool,
}

/// Runs a deterministic micro-benchmark against a named corpus evaluating Tier 2 compression latency.
#[must_use]
pub fn run_corpus_benchmark(
    corpus_name: &'static str,
    payload: &str,
    iterations: usize,
) -> BenchmarkResult {
    let mut latencies_ns = Vec::with_capacity(iterations);

    // Warm-up iteration
    let _ = compress_text(payload, 0.60, None);

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = compress_text(payload, 0.60, None);
        latencies_ns.push(start.elapsed().as_nanos() as u64);
    }

    latencies_ns.sort_unstable();
    let p50_idx = iterations / 2;
    let p99_idx = ((iterations as f64) * 0.99) as usize;

    let p50_ns = latencies_ns.get(p50_idx).copied().unwrap_or(0);
    let p99_ns = latencies_ns
        .get(p99_idx.min(iterations.saturating_sub(1)))
        .copied()
        .unwrap_or(0);

    let total_ns: u64 = latencies_ns.iter().sum();
    let total_bytes = payload.len() * iterations;
    let throughput_mb_per_sec = if total_ns > 0 {
        ((total_bytes as f64) / (1024.0 * 1024.0)) / ((total_ns as f64) / 1_000_000_000.0)
    } else {
        0.0
    };

    let sla_passed = p99_ns <= MAX_P99_LATENCY_SLA_NS;

    BenchmarkResult {
        corpus_name,
        iterations,
        payload_bytes: payload.len(),
        p50_ns,
        p99_ns,
        throughput_mb_per_sec,
        sla_passed,
    }
}

/// Executes the full benchmark suite across standard and adversarial compression-bomb corpora.
pub fn run_all_benchmarks() -> (Vec<BenchmarkResult>, bool) {
    // 1. Construct adversarial 4KB compression bomb payload with 2000+ single-char tokens and nested delimiters
    let mut bomb_payload = String::with_capacity(4096 + 256);
    let bomb_unit = "{ \"k\": [ a, b, c, d, `e`, $(f), $G, \"h\" ], ";
    while bomb_payload.len() < 4096 {
        bomb_payload.push_str(bomb_unit);
    }
    bomb_payload.truncate(4096);

    let corpora = [
        ("MeetingBank (Conversational)", MEETINGBANK_SAMPLE),
        ("GSM8K (Reasoning)", GSM8K_SAMPLE),
        ("Synthetic (Credentials & PII)", SYNTHETIC_SECURITY_SAMPLE),
        ("Adversarial 4KB Compression Bomb", &bomb_payload),
    ];

    let mut results = Vec::new();
    let mut all_passed = true;

    for (name, payload) in corpora {
        let res = run_corpus_benchmark(name, payload, 50);
        if !res.sla_passed {
            all_passed = false;
        }
        results.push(res);
    }

    (results, all_passed)
}
