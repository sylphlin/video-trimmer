"""Centrally managed named constants. Replace scattered literal values."""

# --- Speech/silence gating threshold: use an absolute dBFS floor. ---
# Basis: EBU/ITU-R BS.1770 loudness gating concepts and common industry
# tool ranges. Studio noise floor sits below -60dBFS. Speech peaks sit
# near -20dBFS. Use the midpoint as the absolute gate (CLI can override).
SILENCE_THRESHOLD_DBFS = -40.0

# --- Audio crossfade duration. ---
# Basis: AGENTS.md mandates a 15ms equal-power micro-crossfade
# (afade curve=iqsin/qsin) to prevent acoustic pops and clicks.
CROSSFADE_DURATION_SEC = 0.015

# --- Empirical values below. No public industry standard found. Keep the
# values unchanged; only give them names. ---
HANGOVER_STEPS = 7  # 70ms silence settle window (10ms per step)

# --- Allowed input video extensions (case-insensitive). ---
ALLOWED_INPUT_EXTENSIONS = (".mp4", ".mov")

# --- Gemini API network call retry parameters. ---
# Retry up to 3 times (4 attempts total: 1 call + 3 retries).
# Use exponential backoff: 1s -> 2s -> 4s between attempts.
# Raise GeminiAPIError after the 3rd retry (4th attempt) fails.
GEMINI_RETRY_ATTEMPTS = 4
GEMINI_RETRY_WAIT_MIN_SEC = 1
GEMINI_RETRY_WAIT_MAX_SEC = 4

# --- Gemini output token limit ---
# Set max_output_tokens to 65536 while preserving native dynamic thinking.
GEMINI_MAX_OUTPUT_TOKENS = 65536
