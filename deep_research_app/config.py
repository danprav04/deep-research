# config.py
import os
import logging
import sys
from dotenv import load_dotenv

# --- Load Environment Variables ---
# Load from .env file in the project root (one level up from deep_research_app)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"INFO: Loaded environment variables from: {dotenv_path}")
else:
    print(f"INFO: '.env' file not found at {dotenv_path}. Relying on system environment variables.")

# --- Core App Settings ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME")
# Optional: Flask secret key (highly recommended for production)
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

# --- Logging Configuration ---
# Get log settings from environment or use defaults
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", os.path.join(os.path.dirname(__file__), '..', 'logs', 'deep_research.log'))
LOG_ROTATION_DAYS = int(os.getenv("LOG_ROTATION_DAYS", 7))

# --- Rate Limiting ---
DEFAULT_RATE_LIMIT = os.getenv("DEFAULT_RATE_LIMIT", "15 per minute")

# --- Search Configuration ---
# ** TUNED FOR LOWER RAM (1GB SERVER) - MONITOR AND ADJUST **
MAX_SEARCH_RESULTS_PER_ENGINE_STEP: int = int(os.getenv("MAX_SEARCH_RESULTS_PER_ENGINE_STEP", 5)) # Fewer results per step
MAX_TOTAL_URLS_TO_SCRAPE: int = int(os.getenv("MAX_TOTAL_URLS_TO_SCRAPE", 15)) # Significantly reduced
MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", 3)) # Fewer concurrent workers
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", 20)) # Slightly longer timeout
USER_AGENT: str = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)" # Be a good citizen? Or keep generic? Let's try Googlebot UA.
# USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
DDGS_RETRY_DELAY_SECONDS: float = float(os.getenv("DDGS_RETRY_DELAY_SECONDS", 5.0)) # Increased delay
INTER_SEARCH_DELAY_SECONDS: float = float(os.getenv("INTER_SEARCH_DELAY_SECONDS", 1.5)) # Slightly increased delay

# --- Content Processing ---
# ** TUNED FOR LOWER RAM (1GB SERVER) - MONITOR AND ADJUST **
# Reduced target tokens significantly to lower memory pressure during context building and LLM calls.
# Test performance/cost trade-offs. Gemini Flash may handle more.
TARGET_MAX_CONTEXT_TOKENS: int = int(os.getenv("TARGET_MAX_CONTEXT_TOKENS", 80000)) # Drastically reduced
CHARS_PER_TOKEN_ESTIMATE: float = 4.0 # General estimate
MAX_CONTEXT_CHARS_SAFETY_MARGIN: float = 0.9 # Use 90% of estimated max chars
MAX_CONTEXT_CHARS: int = int(TARGET_MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN_ESTIMATE * MAX_CONTEXT_CHARS_SAFETY_MARGIN)

MIN_MEANINGFUL_WORDS_PER_PAGE: int = int(os.getenv("MIN_MEANINGFUL_WORDS_PER_PAGE", 30)) # Slightly reduced threshold
MAX_SCRAPE_CONTENT_LENGTH_MB: int = int(os.getenv("MAX_SCRAPE_CONTENT_LENGTH_MB", 2)) # Reduced max download size
MAX_SCRAPE_CONTENT_BYTES: int = MAX_SCRAPE_CONTENT_LENGTH_MB * 1024 * 1024
MIN_SANITIZED_CONTENT_LENGTH: int = int(os.getenv("MIN_SANITIZED_CONTENT_LENGTH", 150)) # Minimum chars *after* sanitization

# --- LLM Configuration ---
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.6))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", 3))
LLM_RETRY_DELAY: int = int(os.getenv("LLM_RETRY_DELAY", 5))

# Safety Settings for Google Generative AI (BLOCK_MEDIUM_AND_ABOVE is a reasonable default)
BLOCK_LEVEL = os.getenv("GOOGLE_SAFETY_BLOCK_LEVEL", "BLOCK_MEDIUM_AND_ABOVE")
_VALID_BLOCK_LEVELS = ["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"]
if BLOCK_LEVEL not in _VALID_BLOCK_LEVELS:
    print(f"WARNING: Invalid GOOGLE_SAFETY_BLOCK_LEVEL '{BLOCK_LEVEL}'. Falling back to BLOCK_MEDIUM_AND_ABOVE.")
    BLOCK_LEVEL = "BLOCK_MEDIUM_AND_ABOVE"

SAFETY_SETTINGS = {
    "HARM_CATEGORY_HARASSMENT": BLOCK_LEVEL,
    "HARM_CATEGORY_HATE_SPEECH": BLOCK_LEVEL,
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": BLOCK_LEVEL,
    "HARM_CATEGORY_DANGEROUS_CONTENT": BLOCK_LEVEL,
}

# --- File Handling ---
_TEMP_FILE_DIR_NAME = "temp_scrape_files"
TEMP_FILE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), _TEMP_FILE_DIR_NAME))

# Ensure the temp directory exists
if not os.path.exists(TEMP_FILE_DIR):
    try:
        os.makedirs(TEMP_FILE_DIR)
        print(f"INFO: Created temporary directory: {TEMP_FILE_DIR}")
    except OSError as e:
        # Log this properly once logging is fully set up in app.py
        print(f"ERROR: Could not create temporary directory {TEMP_FILE_DIR}: {e}. Please create it manually.", file=sys.stderr)
        # Depending on requirements, might raise an error later if dir is critical
        # raise OSError(f"Failed to create required temporary directory: {TEMP_FILE_DIR}") from e

# --- Frontend ---
PICO_CSS_CDN: str = "https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css" # Keep using CDN for Pico


# --- Validation (moved logging setup to app.py for better context) ---
# Basic check for essential keys after loading everything
if not GOOGLE_API_KEY:
    raise ValueError("FATAL: GOOGLE_API_KEY environment variable not set.")
if not GOOGLE_MODEL_NAME:
    raise ValueError("FATAL: GOOGLE_MODEL_NAME environment variable not set.")

print("--- Configuration Summary ---")
print(f"Log Level: {LOG_LEVEL}")
print(f"Log File: {LOG_FILE_PATH}")
print(f"Google Model: {GOOGLE_MODEL_NAME}")
print(f"Max Scrape URLs: {MAX_TOTAL_URLS_TO_SCRAPE}")
print(f"Max Workers: {MAX_WORKERS}")
print(f"Target Max Context Tokens: {TARGET_MAX_CONTEXT_TOKENS:,}")
print(f"Calculated Max Context Chars: {MAX_CONTEXT_CHARS:,}")
print(f"Max Scrape Size (MB): {MAX_SCRAPE_CONTENT_LENGTH_MB}")
print(f"LLM Safety Level: {BLOCK_LEVEL}")
print(f"Rate Limit: {DEFAULT_RATE_LIMIT}")
print("---------------------------")