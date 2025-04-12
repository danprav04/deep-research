# web_research.py
import time
import re
import requests
import concurrent.futures
import tempfile
import os
import logging
from typing import List, Dict, Tuple, Optional, Set, Any
from urllib.parse import urlparse, unquote

from bs4 import BeautifulSoup, SoupStrainer # Keep bs4 for basic structure finding if needed
from duckduckgo_search import DDGS, exceptions as ddgs_exceptions

import config as config
from utils import sanitize_scraped_content # Import the sanitizer

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Search Functions ---

def search_duckduckgo_provider(keywords: List[str], max_results: int) -> Dict[str, Any]:
    """
    Performs a search on DuckDuckGo using duckduckgo_search with retry logic.

    Args:
        keywords: A list of keywords to form the search query.
        max_results: The maximum number of results to retrieve.

    Returns:
        A dictionary containing:
        {'engine': 'DuckDuckGo', 'urls': list_of_urls, 'success': bool, 'error': error_message_or_None}
    """
    query = " ".join(keywords)
    urls: List[str] = []
    engine_name = "DuckDuckGo"

    if not query:
        logger.warning("DuckDuckGo search requested with empty keywords.")
        return {"engine": engine_name, "urls": [], "success": True, "error": None} # Success, but no results

    # Reduce retries slightly for faster failure if DDG is blocking
    max_outer_retries = 2
    base_delay = config.DDGS_RETRY_DELAY_SECONDS

    logger.info(f"Searching {engine_name} for: '{query}' (max_results={max_results})")

    for attempt in range(max_outer_retries + 1):
        error_msg = "Unknown search error"
        is_rate_limit = False
        try:
            # Use context manager for DDGS instance
            with DDGS(timeout=config.REQUEST_TIMEOUT) as ddgs:
                 # Use text search, iterate through results
                 results_iterator = ddgs.text(query, max_results=max_results)
                 raw_urls = [r['href'] for r in results_iterator if r and isinstance(r, dict) and r.get('href')]

                 # --- Clean and Validate URLs ---
                 cleaned_urls = set()
                 for raw_url in raw_urls:
                     try:
                         # Decode URL encoding (e.g., %20 -> space) - important for some DDG results
                         decoded_url = unquote(raw_url)
                         # Basic validation: must start with http/https
                         if decoded_url.startswith(('http://', 'https://')):
                             cleaned_urls.add(decoded_url)
                         else:
                             logger.debug(f"DDG Result Filtered (Invalid scheme): {raw_url}")
                     except Exception as url_err:
                         logger.warning(f"Error processing DDG result URL '{raw_url}': {url_err}")

                 urls = list(cleaned_urls)
                 logger.info(f"{engine_name} search for '{query}' successful (Attempt {attempt+1}). Found {len(raw_urls)} raw results, yielding {len(urls)} valid URLs.")
                 return {"engine": engine_name, "urls": urls, "success": True, "error": None}

        except ddgs_exceptions.RateLimitException as e:
            error_msg = f"Rate limit hit searching {engine_name} for '{query}' (Attempt {attempt+1}): {e}"
            is_rate_limit = True
            logger.warning(error_msg, exc_info=False)
        except (ddgs_exceptions.DuckDuckGoSearchException, requests.exceptions.RequestException, Exception) as e:
            # Catch library-specific errors, network errors, and general exceptions
            error_msg = f"Error searching {engine_name} for '{query}' (Attempt {attempt+1}): {type(e).__name__} - {e}"
            error_str_lower = str(e).lower()
            # Check for common indicators of rate limiting even in generic exceptions
            is_rate_limit = is_rate_limit or "rate limit" in error_str_lower or "429" in error_str_lower or "too many requests" in error_str_lower
            logger.warning(error_msg, exc_info=False) # Log only message for warnings


        # --- Retry Logic ---
        if attempt < max_outer_retries:
             # Exponential backoff, capped to avoid excessive waits
             delay = min(base_delay * (2 ** attempt), 30.0) # Cap delay at 30s
             log_prefix = "Rate limit suspected" if is_rate_limit else "Error encountered"
             logger.info(f"  -> {log_prefix} for {engine_name} search '{query}'. Retrying attempt {attempt + 2}/{max_outer_retries + 1} in {delay:.1f}s...")
             time.sleep(delay)
        else:
             # Max retries reached
             final_error_msg = f"Max retries reached for {engine_name} search '{query}'. Last error: {error_msg}"
             logger.error(final_error_msg)
             return {"engine": engine_name, "urls": [], "success": False, "error": final_error_msg}

    # Fallback if loop finishes unexpectedly (should not happen)
    logger.error(f"Unexpected exit from {engine_name} retry loop for query '{query}'.")
    return {"engine": engine_name, "urls": [], "success": False, "error": "Max retries reached unexpectedly."}


def perform_web_search(keywords: List[str]) -> Tuple[List[str], List[str]]:
    """
    Performs searches using configured providers and aggregates unique results.

    Args:
        keywords: List of keywords for the search query.

    Returns:
        A tuple containing:
        - List of unique, validated URLs found across all providers.
        - List of error messages encountered during searches.
    """
    all_unique_urls: Set[str] = set()
    search_errors: List[str] = []

    # Define the search providers to use
    search_providers = [
        search_duckduckgo_provider,
        # Add other search functions here if needed (e.g., Google Custom Search)
        # search_google_cse_provider,
    ]

    # Consider running providers concurrently if multiple are added
    # For now, run sequentially as only DDG is used.
    for i, provider_func in enumerate(search_providers):
        try:
            provider_result = provider_func(keywords, max_results=config.MAX_SEARCH_RESULTS_PER_ENGINE_STEP)
            engine = provider_result.get("engine", f"Unknown Engine {i+1}")
            if provider_result.get("success"):
                provider_urls = provider_result.get("urls", [])
                # URLs from provider should already be validated (http/https)
                new_urls_found = len(set(provider_urls) - all_unique_urls)
                all_unique_urls.update(provider_urls)
                logger.debug(f"  -> {engine} contributed {len(provider_urls)} URLs ({new_urls_found} new). Total unique: {len(all_unique_urls)}")
            else:
                error_msg = provider_result.get("error", "Unknown search error")
                logger.warning(f"  -> {engine} search failed: {error_msg}")
                search_errors.append(f"{engine}: {error_msg}")
        except Exception as e:
            # Catch errors in the provider function call itself
            engine_name = getattr(provider_func, '__name__', f'Provider_{i+1}')
            logger.error(f"Error executing search provider {engine_name}: {e}", exc_info=True)
            search_errors.append(f"{engine_name}: Execution Error - {e}")


    final_url_list = sorted(list(all_unique_urls))
    logger.info(f"Web search for keywords '{' '.join(keywords)}' completed. Found {len(final_url_list)} unique URLs. Encountered {len(search_errors)} errors.")
    return final_url_list, search_errors


# --- Scraping Function ---

def scrape_url(url: str) -> Optional[Dict[str, Any]]:
    """
    Scrapes content from a given URL, sanitizes it using Bleach, saves the
    *sanitized* content to a temporary file if it meets minimum length criteria,
    and returns metadata including the temp file path.

    Args:
        url: The URL to scrape.

    Returns:
        A dictionary {'url': url, 'temp_filepath': path, 'original_size': bytes, 'sanitized_size': bytes} on success,
        None if scraping fails, content is unsuitable (wrong type, too small, empty after sanitization), or an error occurs.
    """
    log_url = url[:75] + '...' if len(url) > 75 else url
    response: Optional[requests.Response] = None
    temp_filepath: Optional[str] = None # Initialize here for broader scope in finally block
    original_size = 0
    sanitized_size = 0

    try:
        headers = {
            'User-Agent': config.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br', # Accept compressed content
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1' # Do Not Track header
            }
        response = requests.get(
            url,
            headers=headers,
            timeout=config.REQUEST_TIMEOUT,
            allow_redirects=True,
            stream=True # Download content in chunks
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # --- Validate Content-Type ---
        content_type = response.headers.get('content-type', '').lower()
        # Be more restrictive: only allow explicit text/html or application/xhtml+xml
        if not ('text/html' in content_type or 'application/xhtml+xml' in content_type):
            # Allow application/xml or text/xml only if explicitly needed? For now, restrict.
            logger.debug(f"Skipping non-HTML URL (Content-Type: {content_type}): {log_url}")
            response.close()
            return None

        # --- Stream content download & check size limit ---
        html_content_bytes = b""
        bytes_read = 0
        max_bytes = config.MAX_SCRAPE_CONTENT_BYTES

        # Get encoding from headers *before* consuming content. requests' response.encoding attempts this.
        encoding_from_headers = response.encoding # Might be None

        for chunk in response.iter_content(chunk_size=8192):
            if not chunk: continue
            bytes_read += len(chunk)
            if bytes_read > max_bytes:
                logger.warning(f"Content download exceeds size limit ({max_bytes / 1024 / 1024:.1f} MB) for {log_url}. Skipping.")
                response.close() # Close the connection
                return None
            html_content_bytes += chunk

        response.close() # Close connection now that content is read
        original_size = bytes_read
        logger.debug(f"Downloaded {original_size / 1024:.1f} KB for {log_url}")

        if not html_content_bytes:
             logger.warning(f"Downloaded empty content for {log_url}.")
             return None

        # --- Decode content carefully ---
        html_content_str = ""
        detected_encoding = None
        # Try decoding using header encoding first (if available), then requests' apparent encoding (guessed from content), finally UTF-8.
        encodings_to_try = [encoding_from_headers, response.apparent_encoding, 'utf-8', 'iso-8859-1', 'windows-1252']
        # Filter out None values and duplicates, keeping order
        unique_encodings = []
        for enc in encodings_to_try:
            # Normalize encoding names for comparison
            normalized_enc = enc.lower() if enc else None
            if normalized_enc and normalized_enc not in [e.lower() for e in unique_encodings if e]:
                 unique_encodings.append(enc)

        for encoding in unique_encodings:
             try:
                 html_content_str = html_content_bytes.decode(encoding, errors='replace') # Use 'replace' for robustness
                 detected_encoding = encoding
                 logger.debug(f"Successfully decoded {log_url} using encoding: {encoding}")
                 break # Stop on first successful decode
             except (UnicodeDecodeError, LookupError) as decode_err:
                 logger.debug(f"Decoding attempt failed for {log_url} with encoding '{encoding}': {decode_err}")
        else: # If loop completes without break (no encoding worked)
            logger.error(f"Failed to decode content for {log_url} using attempted encodings: {unique_encodings}")
            return None # Cannot decode content

        if not html_content_str.strip():
             logger.warning(f"Decoded content is empty or whitespace for {log_url} (used encoding: {detected_encoding}).")
             return None

        # --- Sanitize HTML content using Bleach (via utils function) ---
        # This is the CRITICAL step before sending to LLM
        sanitized_content = sanitize_scraped_content(html_content_str, url=log_url)
        sanitized_size = len(sanitized_content.encode('utf-8')) # Size *after* sanitization

        # --- Check if meaningful content remains after sanitization ---
        if not sanitized_content or len(sanitized_content) < config.MIN_SANITIZED_CONTENT_LENGTH:
            logger.debug(f"Content too short ({len(sanitized_content)} chars) or empty after sanitization for {log_url}. Min required: {config.MIN_SANITIZED_CONTENT_LENGTH}. Skipping.")
            return None

        # --- Save *sanitized* text to a temporary file ---
        try:
            # Use mkstemp for slightly more secure temp file creation
            fd, temp_filepath = tempfile.mkstemp(
                suffix=".txt", prefix="sanitized_", dir=config.TEMP_FILE_DIR, text=True
            )
            with os.fdopen(fd, 'w', encoding='utf-8') as temp_file_obj:
                temp_file_obj.write(sanitized_content)

            # Verify file was written
            if not os.path.exists(temp_filepath) or os.path.getsize(temp_filepath) == 0:
                 logger.error(f"Failed to write or created empty temp file for {log_url} at {temp_filepath}")
                 # Clean up empty file if it exists
                 if temp_filepath and os.path.exists(temp_filepath): os.remove(temp_filepath)
                 temp_filepath = None # Ensure path is None if writing failed
                 return None

            file_size_kb = os.path.getsize(temp_filepath) / 1024
            logger.info(f"Successfully scraped & sanitized {log_url} -> {os.path.basename(temp_filepath)} ({file_size_kb:.1f} KB sanitized)")
            return {
                'url': url,
                'temp_filepath': temp_filepath,
                'original_size': original_size,
                'sanitized_size': sanitized_size
                }

        except IOError as file_err:
            logger.error(f"IOError writing temp file for {log_url}: {file_err}", exc_info=True)
            # Cleanup attempt if temp_filepath was assigned by mkstemp
            if temp_filepath and os.path.exists(temp_filepath):
                try: os.remove(temp_filepath)
                except OSError as e: logger.error(f"Failed to cleanup temp file {temp_filepath} after IO error: {e}")
            temp_filepath = None
            return None
        except Exception as file_err:
            logger.error(f"Unexpected error writing temp file for {log_url}: {file_err}", exc_info=True)
            # Cleanup attempt if temp_filepath was assigned by mkstemp
            if temp_filepath and os.path.exists(temp_filepath):
                try: os.remove(temp_filepath)
                except OSError as e: logger.error(f"Failed to cleanup temp file {temp_filepath} after unexpected error: {e}")
            temp_filepath = None
            return None

    # --- Exception Handling for Request/Scraping Process ---
    except requests.exceptions.Timeout as e:
        logger.warning(f"Timeout error fetching URL {log_url}: {e}")
    except requests.exceptions.HTTPError as e:
        # Log client errors as warning, server errors as error
        if 400 <= e.response.status_code < 500:
             logger.warning(f"HTTP Client Error {e.response.status_code} fetching URL {log_url}: {e}")
        else:
            logger.error(f"HTTP Server Error {e.response.status_code} fetching URL {log_url}: {e}")
    except requests.exceptions.ConnectionError as e:
         # Often transient network issues or DNS errors
         logger.warning(f"Connection Error fetching URL {log_url}: {e}")
    except requests.exceptions.RequestException as e:
        # Catch other requests-related errors (e.g., InvalidURL, TooManyRedirects)
        logger.error(f"Request Error fetching URL {log_url}: {type(e).__name__} - {e}", exc_info=False)
    except Exception as e:
        # Catch-all for any other unexpected errors during the process
        logger.error(f"Unexpected Error processing URL {log_url}: {type(e).__name__} - {e}", exc_info=True) # Log stack trace for unexpected errors

    # --- Cleanup in case of error ---
    finally:
        # Ensure response object is closed if it exists
        if response is not None:
            try: response.close()
            except Exception: pass
        # Ensure temporary file is deleted if created but function failed before returning it
        # This is a safeguard, primary cleanup happens in app.py based on success events
        # Check if temp_filepath was assigned but the function is returning None (error)
        # if temp_filepath and 'return' not in locals(): # Heuristic, not fully reliable
        #     if os.path.exists(temp_filepath):
        #          logger.debug(f"Cleaning up orphaned temp file in scrape_url finally block: {temp_filepath}")
        #          try: os.remove(temp_filepath)
        #          except OSError as e: logger.error(f"Error cleaning orphaned temp file {temp_filepath}: {e}")

    return None # Return None if any error prevented successful scraping and saving