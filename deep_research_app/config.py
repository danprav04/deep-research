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

dotenv_path = os.path.join(_PROJECT_ROOT, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"INFO: Loaded environment variables from: {dotenv_path}")
else:
    # Check one level up from project root as a fallback (e.g., if running tests from root)
    dotenv_path_alt = os.path.join(_PROJECT_ROOT, '..', '.env')
    if os.path.exists(dotenv_path_alt):
        load_dotenv(dotenv_path=dotenv_path_alt)
        print(f"INFO: Loaded environment variables from alternative path: {dotenv_path_alt}")
    else:
        print(f"INFO: '.env' file not found at {dotenv_path} or {dotenv_path_alt}. Relying on system environment variables.")


# --- Core App Settings ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Default log directory within the project root
_DEFAULT_LOG_DIR = os.path.join(_PROJECT_ROOT, 'logs')
# Ensure the default log file path is absolute
_DEFAULT_LOG_FILE = os.path.abspath(os.path.join(_DEFAULT_LOG_DIR, 'deep_research.log'))
# Use absolute path from env var if provided, otherwise use the calculated absolute default
LOG_FILE_PATH = os.path.abspath(os.getenv("LOG_FILE_PATH", _DEFAULT_LOG_FILE))
LOG_ROTATION_DAYS = int(os.getenv("LOG_ROTATION_DAYS", 7))

# --- Rate Limiting ---
DEFAULT_RATE_LIMIT = os.getenv("DEFAULT_RATE_LIMIT", "15 per minute")

# --- Search Configuration ---
# ** TUNED FOR LOWER RAM (1GB SERVER) - MONITOR AND ADJUST **
MAX_SEARCH_RESULTS_PER_ENGINE_STEP: int = int(os.getenv("MAX_SEARCH_RESULTS_PER_ENGINE_STEP", 5))
MAX_TOTAL_URLS_TO_SCRAPE: int = int(os.getenv("MAX_TOTAL_URLS_TO_SCRAPE", 25))
MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", 3))
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", 20))
USER_AGENT: str = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"

# ** DECREASED DELAYS AS REQUESTED - MAY INCREASE RATE LIMITING **
DDGS_RETRY_DELAY_SECONDS: float = float(os.getenv("DDGS_RETRY_DELAY_SECONDS", 0.5)) # << SIGNIFICANTLY DECREASED base retry delay
INTER_SEARCH_DELAY_SECONDS: float = float(os.getenv("INTER_SEARCH_DELAY_SECONDS", 1)) # << SIGNIFICANTLY DECREASED delay between search steps

# --- Content Processing ---
TARGET_MAX_CONTEXT_TOKENS: int = int(os.getenv("TARGET_MAX_CONTEXT_TOKENS", 700000))
CHARS_PER_TOKEN_ESTIMATE: float = 4.0
MAX_CONTEXT_CHARS_SAFETY_MARGIN: float = 0.9
MAX_CONTEXT_CHARS: int = int(TARGET_MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN_ESTIMATE * MAX_CONTEXT_CHARS_SAFETY_MARGIN)

MIN_MEANINGFUL_WORDS_PER_PAGE: int = int(os.getenv("MIN_MEANINGFUL_WORDS_PER_PAGE", 30))
MAX_SCRAPE_CONTENT_LENGTH_MB: int = int(os.getenv("MAX_SCRAPE_CONTENT_LENGTH_MB", 2))
MAX_SCRAPE_CONTENT_BYTES: int = MAX_SCRAPE_CONTENT_LENGTH_MB * 1024 * 1024
MIN_SANITIZED_CONTENT_LENGTH: int = int(os.getenv("MIN_SANITIZED_CONTENT_LENGTH", 150))

# --- LLM Configuration ---
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.6))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", 3))
LLM_RETRY_DELAY: int = int(os.getenv("LLM_RETRY_DELAY", 5))

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
if not os.path.exists(TEMP_FILE_DIR):
    try:
        os.makedirs(TEMP_FILE_DIR)
        print(f"INFO: Created temporary directory: {TEMP_FILE_DIR}")
    except OSError as e:
        # Log error but allow app to continue; scraping will likely fail later
        print(f"ERROR: Could not create temporary directory {TEMP_FILE_DIR}: {e}. Scraping might fail.", file=sys.stderr)

# --- Frontend ---
PICO_CSS_CDN: str = "https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css"


# --- Validation ---
if not GOOGLE_API_KEY:
    # Use print to stderr before logging might be fully configured
    print("CRITICAL: GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
    raise ValueError("FATAL: GOOGLE_API_KEY environment variable not set.")
if not GOOGLE_MODEL_NAME:
    print("CRITICAL: GOOGLE_MODEL_NAME environment variable not set.", file=sys.stderr)
    raise ValueError("FATAL: GOOGLE_MODEL_NAME environment variable not set.")

print("--- Configuration Summary ---")
print(f"Project Root: {_PROJECT_ROOT}")
print(f"Log Level: {LOG_LEVEL}")
print(f"Log File Path (Absolute): {LOG_FILE_PATH}")
print(f"Temporary Files Directory: {TEMP_FILE_DIR}")
print(f"Google Model: {GOOGLE_MODEL_NAME}")
print(f"Max Scrape URLs: {MAX_TOTAL_URLS_TO_SCRAPE}")
print(f"Max Workers: {MAX_WORKERS}")
print(f"Target Max Context Tokens: {TARGET_MAX_CONTEXT_TOKENS:,}")
print(f"Calculated Max Context Chars: {MAX_CONTEXT_CHARS:,}")
print(f"Max Scrape Size (MB): {MAX_SCRAPE_CONTENT_LENGTH_MB}")
print(f"LLM Safety Level: {BLOCK_LEVEL}")
print(f"Rate Limit: {DEFAULT_RATE_LIMIT}")
print(f"DDGS Retry Delay Base: {DDGS_RETRY_DELAY_SECONDS}s") # <-- Updated
print(f"Inter-Search Step Delay: {INTER_SEARCH_DELAY_SECONDS}s") # <-- Updated
print("---------------------------")