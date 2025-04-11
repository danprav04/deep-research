# config.py
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # Look for .env in parent dir
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"INFO: Loaded environment variables from: {dotenv_path}") # Use print before logger might be fully set up
else:
    print(f"INFO: '.env' file not found at {dotenv_path}. Relying on system environment variables.")

# --- Configure Logging Early ---
# Basic config, can be overridden/enhanced in app.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Google API Key and Model Name ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME")

# Early exit if critical variables are missing
if not GOOGLE_API_KEY:
    logging.critical("FATAL: GOOGLE_API_KEY environment variable not set.")
    raise ValueError("FATAL: GOOGLE_API_KEY environment variable not set.")
if not GOOGLE_MODEL_NAME:
    logging.critical("FATAL: GOOGLE_MODEL_NAME environment variable not set.")
    raise ValueError("FATAL: GOOGLE_MODEL_NAME environment variable not set.")

# --- Search Configuration ---
MAX_SEARCH_RESULTS_PER_ENGINE_STEP: int = 8 # Slightly reduced default
MAX_TOTAL_URLS_TO_SCRAPE: int = 50 # Reduced default for RAM/time
MAX_WORKERS: int = 5 # Concurrent scraping workers
REQUEST_TIMEOUT: int = 15 # Slightly increased timeout for requests
USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36" # Keep generic UA
PICO_CSS_CDN: str = "https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css"
DDGS_RETRY_DELAY_SECONDS: float = 3.0 # Delay for DDGS rate limit retries
INTER_SEARCH_DELAY_SECONDS: float = 1.0 # Shorter delay between search keywords within a step

# --- Content Processing ---
# Target token limit for the LLM context (adjust based on the specific model used)
# Gemini 1.5 Flash/Pro often handle large contexts well, but keep it reasonable for performance/cost.
TARGET_MAX_CONTEXT_TOKENS: int = 500000 # Reduced default target
CHARS_PER_TOKEN_ESTIMATE: float = 4.0 # General estimate
MAX_CONTEXT_CHARS_SAFETY_MARGIN: float = 0.9 # Use 90% of estimated max chars
# Calculate the approximate character limit for the *entire* context payload
MAX_CONTEXT_CHARS: int = int(TARGET_MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN_ESTIMATE * MAX_CONTEXT_CHARS_SAFETY_MARGIN)
logging.info(f"Calculated MAX_CONTEXT_CHARS for LLM: {MAX_CONTEXT_CHARS:,}")

MIN_MEANINGFUL_WORDS_PER_PAGE: int = 50 # Minimum words to consider a scrape successful
MAX_SCRAPE_CONTENT_LENGTH_MB: int = 5 # Limit max size of HTML download for scraping
MAX_SCRAPE_CONTENT_BYTES: int = MAX_SCRAPE_CONTENT_LENGTH_MB * 1024 * 1024

# --- LLM Configuration ---
LLM_TEMPERATURE: float = 0.6 # Controls randomness (creativity vs. factuality)
LLM_MAX_RETRIES: int = 3 # Retries on transient API errors
LLM_RETRY_DELAY: int = 5 # Seconds between retries

# Safety Settings for Google Generative AI
# Using string names directly as supported by recent API versions
# Valid settings: BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE, BLOCK_LOW_AND_ABOVE
BLOCK_LEVEL = "BLOCK_MEDIUM_AND_ABOVE"
SAFETY_SETTINGS = {
    "HARM_CATEGORY_HARASSMENT": BLOCK_LEVEL,
    "HARM_CATEGORY_HATE_SPEECH": BLOCK_LEVEL,
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": BLOCK_LEVEL,
    "HARM_CATEGORY_DANGEROUS_CONTENT": BLOCK_LEVEL,
}
logging.info(f"Using LLM safety block level: {BLOCK_LEVEL}")

# --- File Handling ---
# Define the directory for temporary scraped files (relative to this config file's location)
_TEMP_FILE_DIR_NAME = "temp_scrape_files"
TEMP_FILE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), _TEMP_FILE_DIR_NAME))

# Ensure the temp directory exists (optional: create if not present)
if not os.path.exists(TEMP_FILE_DIR):
    try:
        os.makedirs(TEMP_FILE_DIR)
        logging.info(f"Created temporary directory: {TEMP_FILE_DIR}")
    except OSError as e:
        logging.error(f"Could not create temporary directory {TEMP_FILE_DIR}: {e}. Please create it manually.", exc_info=True)
        # Depending on requirements, you might want to raise an error here
        # raise OSError(f"Failed to create required temporary directory: {TEMP_FILE_DIR}") from e

DOWNLOAD_FILENAME_MAX_LENGTH: int = 200 # Max length for downloaded filenames (unused currently)