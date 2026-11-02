"""Centrally managed named constants for the Video Trimmer pipeline."""

# Absolute dBFS silence floor based on EBU/ITU-R BS.1770 loudness gating (-40 dBFS).
SILENCE_THRESHOLD_DBFS = -40.0

# Equal-power audio micro-crossfade duration (20ms, lower bound of the 20-50ms click-free range).
CROSSFADE_DURATION_SEC = 0.020

# Number of 10ms steps (70ms total) required to confirm trailing speech decay into silence.
HANGOVER_STEPS = 7

# Supported input video file extensions (case-insensitive).
ALLOWED_INPUT_EXTENSIONS = (".mp4", ".mov")

# Vertex AI network retry configuration (1 initial attempt + up to 3 exponential-backoff retries).
GEMINI_RETRY_ATTEMPTS = 4
GEMINI_RETRY_WAIT_MIN_SEC = 1
GEMINI_RETRY_WAIT_MAX_SEC = 4

# Maximum output token limit for Gemini multimodal JSON responses.
GEMINI_MAX_OUTPUT_TOKENS = 65536

# Dual-engine word-level ASR defaults (Chirp 3 cloud default, Whisper local optional).
DEFAULT_ASR_ENGINE = "chirp3"
DEFAULT_CHIRP3_MODEL = "chirp_3"
DEFAULT_STT_LOCATION = "us"
DEFAULT_WHISPER_MODEL = "small"

# Composite Sub-Span (Last-Take-Anchored Prefix Backfill & Intra-Sentence Excision) parameters.
COMPOSITE_MIN_PREFIX_SEC = 0.55
COMPOSITE_MIN_CJK_CHARS = 4
COMPOSITE_MIN_LATIN_WORDS = 3
COMPOSITE_MIN_LATIN_CHARS = 10
COMPOSITE_PUNCH_IN_SCALE = 1.12
COMPOSITE_SEAM_OUT_MARGIN_SEC = 0.040
COMPOSITE_SEAM_IN_MARGIN_SEC = 0.040
COMPOSITE_ONSET_MAX_INTERNAL_GAP_SEC = 0.120
COMPOSITE_GAIN_MATCH_THRESHOLD_DB = 1.5
COMPOSITE_GAIN_MATCH_MAX_DB = 4.5

