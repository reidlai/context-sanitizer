/**
 * Universal WebAssembly and WASI client-side sanitization typings (`Story 5.1`, `FR13`, `AD-3`).
 */

/**
 * Result returned by `sanitize()`.
 */
export interface SanitizeResult {
    /**
     * Sanitized and optionally compressed JSON payload or text.
     */
    readonly sanitizedJson: string;
    /**
     * Cryptographic session vault handle.
     */
    readonly vault: WasmVaultHandle;
}

/**
 * Custom error thrown when a prompt injection or exfiltration attempt is blocked (`FR8`, `FR13`).
 */
export class InjectionBlockedError extends Error {
    readonly name: "InjectionBlockedError";
    readonly reason: string;
    readonly score: number;
    readonly score_bps: number;
}

/**
 * Custom error thrown when rehydration is attempted for an unauthorized destination (`FR3`, `FR13`).
 */
export class UnauthorizedRehydrationError extends Error {
    readonly name: "UnauthorizedRehydrationError";
    readonly destination: string;
}

/**
 * Opaque memory-safe handle wrapping the cryptographic session vault (`AD-3`, `FR13`, `NFR2`).
 */
export class WasmVaultHandle {
    /**
     * Creates an empty vault handle.
     */
    constructor();

    /**
     * Rehydrates `<SHADOW:...>` tokens in tool arguments back to original PII values
     * if and only if `destination` matches the session whitelist.
     *
     * @param toolArgs JSON string or object containing shadow tokens.
     * @param destination Target tool or service destination identifier.
     * @throws {UnauthorizedRehydrationError} if destination is not in whitelist.
     */
    rehydrate(toolArgs: string, destination: string): string;
    rehydrate<T = any>(toolArgs: T, destination: string): T;

    /**
     * Explicitly frees the vault, immediately dropping all session salt and tokens (`NFR2`).
     */
    free(): void;
}

/**
 * Sanitizes and optionally compresses an OpenAI chat completion payload, completion payload, or raw text.
 *
 * @param payload JSON string or raw text to sanitize.
 * @param rate Target compression rate between 0.0 and 1.0 (default: 1.0 = no compression).
 * @param whitelist Optional array of allowed destination names for tool rehydration.
 * @returns {SanitizeResult} Sanitized JSON string and session vault handle.
 * @throws {InjectionBlockedError} If a prompt injection is detected.
 * @throws {Error} If payload is malformed JSON or rate is invalid.
 */
export function sanitize(
    payload: string | object,
    rate?: number,
    whitelist?: string[]
): SanitizeResult;

/**
 * Minifies OpenAI tool schema definitions in-place, stripping verbose prose descriptions (`FR-6`).
 *
 * @param toolsJson JSON string of tools array or schema object.
 * @returns Minified JSON string.
 */
export function minify_tools(toolsJson: string): string;
