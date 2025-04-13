# config.py
import os
import sys # Import sys for stderr
import logging
from dotenv import load_dotenv

# --- Load Environment Variables ---
# Determine the absolute path to the project root directory
# Assuming config.py is in 'deep_research_app' which is one level below the project root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"INFO: Project Root determined as: {_PROJECT_ROOT}")

# Load .env file from the project root
dotenv_path = os.path.join(_PROJECT_ROOT, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"INFO: Loaded environment variables from: {dotenv_path}")
else:
    print(f"INFO: '.env' file not found at {dotenv_path}. Relying on system environment variables.")

# --- Core App Settings ---
# Load required secrets from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash-latest") # Sensible default model
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY") # MUST be set in production for security

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Default log directory within the project root
_DEFAULT_LOG_DIR = os.path.join(_PROJECT_ROOT, 'logs')
# Ensure the default log file path is absolute
_DEFAULT_LOG_FILE = os.path.abspath(os.path.join(_DEFAULT_LOG_DIR, 'deep_research.log'))
# Use absolute path from env var if provided, otherwise use the calculated absolute default
LOG_FILE_PATH = os.path.abspath(os.getenv("LOG_FILE_PATH", _DEFAULT_LOG_FILE))
LOG_ROTATION_DAYS = int(os.getenv("LOG_ROTATION_DAYS", 14)) # Keep logs for 2 weeks

# --- Rate Limiting ---
# This limit applies ONLY to routes explicitly decorated with @limiter.limit(DEFAULT_RATE_LIMIT)
# Ensure DEFAULT_RATE_LIMIT env var is unset or matches this if overriding.
DEFAULT_RATE_LIMIT = os.getenv("DEFAULT_RATE_LIMIT", "1 per minute")

# --- Search Configuration ---
# Sensible defaults for production, adjustable via environment variables
MAX_SEARCH_RESULTS_PER_ENGINE_STEP: int = int(os.getenv("MAX_SEARCH_RESULTS_PER_ENGINE_STEP", 5))
MAX_TOTAL_URLS_TO_SCRAPE: int = int(os.getenv("MAX_TOTAL_URLS_TO_SCRAPE", 20)) # Reduced slightly from user value
MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", 3)) # Adjust based on server CPU/RAM
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", 15)) # Slightly shorter timeout
USER_AGENT: str = os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)")

# Production-safe search delays - adjust via env vars if needed
# Very low values (like 0.5s/1s) risk frequent external rate limits from DDG
DDGS_RETRY_DELAY_SECONDS: float = float(os.getenv("DDGS_RETRY_DELAY_SECONDS", 5.0)) # More robust retry delay
INTER_SEARCH_DELAY_SECONDS: float = float(os.getenv("INTER_SEARCH_DELAY_SECONDS", 8.0)) # More robust delay between steps

# --- Content Processing ---
# User's higher token limit - ensure sufficient RAM and LLM support
TARGET_MAX_CONTEXT_TOKENS: int = int(os.getenv("TARGET_MAX_CONTEXT_TOKENS", 700000))
CHARS_PER_TOKEN_ESTIMATE: float = 4.0 # General estimate
MAX_CONTEXT_CHARS_SAFETY_MARGIN: float = 0.9 # Use 90% of estimated char limit
MAX_CONTEXT_CHARS: int = int(TARGET_MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN_ESTIMATE * MAX_CONTEXT_CHARS_SAFETY_MARGIN)

MIN_MEANINGFUL_WORDS_PER_PAGE: int = int(os.getenv("MIN_MEANINGFUL_WORDS_PER_PAGE", 50)) # Increased slightly
MAX_SCRAPE_CONTENT_LENGTH_MB: int = int(os.getenv("MAX_SCRAPE_CONTENT_LENGTH_MB", 2))
MAX_SCRAPE_CONTENT_BYTES: int = MAX_SCRAPE_CONTENT_LENGTH_MB * 1024 * 1024
MIN_SANITIZED_CONTENT_LENGTH: int = int(os.getenv("MIN_SANITIZED_CONTENT_LENGTH", 200)) # Increased slightly

# --- LLM Configuration ---
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.5)) # Slightly lower temp for more deterministic reports
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", 2)) # Fewer retries by default
LLM_RETRY_DELAY: int = int(os.getenv("LLM_RETRY_DELAY", 7)) # Slightly longer retry delay

BLOCK_LEVEL = os.getenv("GOOGLE_SAFETY_BLOCK_LEVEL", "BLOCK_MEDIUM_AND_ABOVE")
_VALID_BLOCK_LEVELS = ["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"]
if BLOCK_LEVEL not in _VALID_BLOCK_LEVELS:
    print(f"WARNING: Invalid GOOGLE_SAFETY_BLOCK_LEVEL '{BLOCK_LEVEL}'. Falling back to BLOCK_MEDIUM_AND_ABOVE.", file=sys.stderr)
    BLOCK_LEVEL = "BLOCK_MEDIUM_AND_ABOVE"

SAFETY_SETTINGS = {
    "HARM_CATEGORY_HARASSMENT": BLOCK_LEVEL,
    "HARM_CATEGORY_HATE_SPEECH": BLOCK_LEVEL,
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": BLOCK_LEVEL,
    "HARM_CATEGORY_DANGEROUS_CONTENT": BLOCK_LEVEL,
}

# --- File Handling ---
_TEMP_FILE_DIR_NAME = "temp_scrape_files"
# Ensure TEMP_FILE_DIR is absolute, placed inside the project root
TEMP_FILE_DIR = os.path.abspath(os.path.join(_PROJECT_ROOT, _TEMP_FILE_DIR_NAME))

# Create TEMP_FILE_DIR if it doesn't exist during config loading
# Handles potential permission issues during startup
if not os.path.exists(TEMP_FILE_DIR):
    try:
        os.makedirs(TEMP_FILE_DIR, exist_ok=True) # Use exist_ok=True
        print(f"INFO: Ensured temporary directory exists: {TEMP_FILE_DIR}")
        # Check writability after ensuring existence
        if not os.access(TEMP_FILE_DIR, os.W_OK):
             raise OSError(f"Temporary directory '{TEMP_FILE_DIR}' exists but is not writable.")
    except OSError as e:
        print(f"ERROR: Could not create or access temporary directory {TEMP_FILE_DIR}: {e}. Scraping will likely fail.", file=sys.stderr)
    except Exception as e: # Catch other potential errors like permission issues
        print(f"ERROR: Unexpected error setting up temporary directory {TEMP_FILE_DIR}: {e}. Scraping will likely fail.", file=sys.stderr)


# --- Frontend ---
PICO_CSS_CDN: str = "https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css"


# --- Validation ---
# Ensure critical secrets/configs are loaded
_validation_errors = []
if not GOOGLE_API_KEY:
    _validation_errors.append("GOOGLE_API_KEY environment variable not set.")
if not GOOGLE_MODEL_NAME:
     _validation_errors.append("GOOGLE_MODEL_NAME environment variable not set (or using default).")
if not FLASK_SECRET_KEY:
    _validation_errors.append("FLASK_SECRET_KEY environment variable not set (critical for session security).")

if _validation_errors:
     print("--- CRITICAL CONFIGURATION ERRORS ---", file=sys.stderr)
     for err in _validation_errors:
         print(f"- {err}", file=sys.stderr)
     print("--- APPLICATION STARTUP HALTED ---", file=sys.stderr)
     raise ValueError("FATAL: Missing critical configuration. Please set the required environment variables.")


# --- Final Configuration Summary ---
print("\n--- Configuration Summary ---")
print(f"Project Root: {_PROJECT_ROOT}")
print(f"Log Level: {LOG_LEVEL}")
print(f"Log File Path (Absolute): {LOG_FILE_PATH}")
print(f"Temporary Files Directory: {TEMP_FILE_DIR}")
print(f"Google Model: {GOOGLE_MODEL_NAME}")
print(f"Flask Secret Key Set: {'Yes' if FLASK_SECRET_KEY else 'NO (WARNING!)'}")
print(f"Max Scrape URLs: {MAX_TOTAL_URLS_TO_SCRAPE}")
print(f"Max Workers (Scraping): {MAX_WORKERS}")
print(f"Target Max Context Tokens (LLM): {TARGET_MAX_CONTEXT_TOKENS:,}")
print(f"Calculated Max Context Chars (LLM): {MAX_CONTEXT_CHARS:,}")
print(f"Max Scrape Size (MB): {MAX_SCRAPE_CONTENT_LENGTH_MB}")
print(f"LLM Safety Level: {BLOCK_LEVEL}")
print(f"Rate Limit (Decorated Routes): {DEFAULT_RATE_LIMIT} per IP")
print(f"DDGS Retry Delay Base: {DDGS_RETRY_DELAY_SECONDS}s")
print(f"Inter-Search Step Delay: {INTER_SEARCH_DELAY_SECONDS}s")
print("---------------------------\n")