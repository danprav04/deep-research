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

from bs4 import BeautifulSoup # Keep bs4 for potential use in sanitization fallback
from duckduckgo_search import DDGS, exceptions as ddgs_exceptions
import chardet

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

    max_outer_retries = 2
    base_delay = config.DDGS_RETRY_DELAY_SECONDS

    logger.info(f"Searching {engine_name} for: '{query}' (max_results={max_results})")

    for attempt in range(max_outer_retries + 1):
        error_msg = "Unknown search error"
        is_rate_limit = False
        try:
            with DDGS(timeout=config.REQUEST_TIMEOUT) as ddgs:
                 results_iterator = ddgs.text(query, max_results=max_results)
                 raw_urls = [r['href'] for r in results_iterator if r and isinstance(r, dict) and r.get('href')]

                 cleaned_urls = set()
                 for raw_url in raw_urls:
                     try:
                         decoded_url = unquote(raw_url)
                         if decoded_url.startswith(('http://', 'https://')):
                             cleaned_urls.add(decoded_url)
                         else:
                             logger.debug(f"DDG Result Filtered (Invalid scheme): {raw_url}")
                     except Exception as url_err:
                         logger.warning(f"Error processing DDG result URL '{raw_url}': {url_err}")

                 urls = list(cleaned_urls)
                 logger.info(f"{engine_name} search for '{query}' successful (Attempt {attempt+1}). Found {len(raw_urls)} raw results, yielding {len(urls)} valid URLs.")
                 return {"engine": engine_name, "urls": urls, "success": True, "error": None}

        # --- CORRECTED EXCEPTION NAME BELOW ---
        except ddgs_exceptions.RatelimitException as e: # Changed RateLimitException -> RatelimitException
            error_msg = f"Rate limit hit searching {engine_name} for '{query}' (Attempt {attempt+1}): {e}"
            is_rate_limit = True
            logger.warning(error_msg, exc_info=False)
        except (ddgs_exceptions.DuckDuckGoSearchException, requests.exceptions.RequestException, Exception) as e:
            error_msg = f"Error searching {engine_name} for '{query}' (Attempt {attempt+1}): {type(e).__name__} - {e}"
            error_str_lower = str(e).lower()
            is_rate_limit = is_rate_limit or "rate limit" in error_str_lower or "429" in error_str_lower or "too many requests" in error_str_lower or "202" in error_str_lower # Check for 202 status which lib uses for rate limit
            logger.warning(error_msg, exc_info=False)

        if attempt < max_outer_retries:
             delay = min(base_delay * (2 ** attempt), 30.0) # Cap delay at 30s
             log_prefix = "Rate limit suspected" if is_rate_limit else "Error encountered"
             logger.info(f"  -> {log_prefix} for {engine_name} search '{query}'. Retrying attempt {attempt + 2}/{max_outer_retries + 1} in {delay:.1f}s...")
             time.sleep(delay)
        else:
             final_error_msg = f"Max retries reached for {engine_name} search '{query}'. Last error: {error_msg}"
             logger.error(final_error_msg)
             return {"engine": engine_name, "urls": [], "success": False, "error": final_error_msg}

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

    search_providers = [
        search_duckduckgo_provider,
    ]

    for i, provider_func in enumerate(search_providers):
        try:
            provider_result = provider_func(keywords, max_results=config.MAX_SEARCH_RESULTS_PER_ENGINE_STEP)
            engine = provider_result.get("engine", f"Unknown Engine {i+1}")
            if provider_result.get("success"):
                provider_urls = provider_result.get("urls", [])
                new_urls_found = len(set(provider_urls) - all_unique_urls)
                all_unique_urls.update(provider_urls)
                logger.debug(f"  -> {engine} contributed {len(provider_urls)} URLs ({new_urls_found} new). Total unique: {len(all_unique_urls)}")
            else:
                error_msg = provider_result.get("error", "Unknown search error")
                logger.warning(f"  -> {engine} search failed: {error_msg}")
                search_errors.append(f"{engine}: {error_msg}")
        except Exception as e:
            # Catch errors *during* the provider function call itself (like the previous AttributeError)
            engine_name = getattr(provider_func, '__name__', f'Provider_{i+1}')
            logger.error(f"Error executing search provider {engine_name}: {e}", exc_info=True)
            # Format a user-friendly error message
            error_msg = f"{engine_name}: Execution Error - {type(e).__name__}"
            search_errors.append(error_msg)


    final_url_list = sorted(list(all_unique_urls))
    logger.info(f"Web search for keywords '{' '.join(keywords)}' completed. Found {len(final_url_list)} unique URLs. Encountered {len(search_errors)} errors.")
    return final_url_list, search_errors


# --- Scraping Function ---
# [ scrape_url function remains unchanged from the previous correct version ]
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
    temp_filepath: Optional[str] = None
    original_size = 0
    sanitized_size = 0

    try:
        headers = {
            'User-Agent': config.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1'
            }
        response = requests.get(
            url,
            headers=headers,
            timeout=config.REQUEST_TIMEOUT,
            allow_redirects=True,
            stream=True
        )
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if not ('text/html' in content_type or 'application/xhtml+xml' in content_type):
            logger.debug(f"Skipping non-HTML URL (Content-Type: {content_type}): {log_url}")
            response.close()
            return None

        encoding_from_headers = response.encoding

        html_content_bytes = b""
        bytes_read = 0
        max_bytes = config.MAX_SCRAPE_CONTENT_BYTES

        for chunk in response.iter_content(chunk_size=8192):
            if not chunk: continue
            bytes_read += len(chunk)
            if bytes_read > max_bytes:
                logger.warning(f"Content download exceeds size limit ({max_bytes / 1024 / 1024:.1f} MB) for {log_url}. Skipping.")
                response.close()
                return None
            html_content_bytes += chunk

        response.close()
        original_size = bytes_read
        logger.debug(f"Downloaded {original_size / 1024:.1f} KB for {log_url}")

        if not html_content_bytes:
             logger.warning(f"Downloaded empty content for {log_url}.")
             return None

        html_content_str = ""
        final_encoding_used = None
        detected_encoding_from_bytes = None

        if not encoding_from_headers:
            try:
                detection_result = chardet.detect(html_content_bytes)
                if detection_result and detection_result['encoding'] and detection_result['confidence'] > 0.7:
                    detected_encoding_from_bytes = detection_result['encoding']
                    logger.debug(f"Encoding not in headers for {log_url}. Detected '{detected_encoding_from_bytes}' with confidence {detection_result['confidence']:.2f}")
                else:
                     logger.debug(f"Encoding not in headers for {log_url}. Chardet detection inconclusive: {detection_result}")
            except Exception as chardet_err:
                logger.warning(f"Chardet detection failed for {log_url}: {chardet_err}")

        encodings_to_try = []
        if encoding_from_headers:
            encodings_to_try.append(encoding_from_headers)
        if detected_encoding_from_bytes and (not encoding_from_headers or detected_encoding_from_bytes.lower() != encoding_from_headers.lower()):
             encodings_to_try.append(detected_encoding_from_bytes)
        fallbacks = ['utf-8', 'iso-8859-1', 'windows-1252']
        existing_lower = {e.lower() for e in encodings_to_try}
        for fb in fallbacks:
             if fb.lower() not in existing_lower:
                 encodings_to_try.append(fb)

        logger.debug(f"Attempting decode for {log_url} with encodings: {encodings_to_try}")

        for encoding in encodings_to_try:
             try:
                 html_content_str = html_content_bytes.decode(encoding, errors='replace')
                 final_encoding_used = encoding
                 logger.debug(f"Successfully decoded {log_url} using encoding: {encoding}")
                 break
             except (UnicodeDecodeError, LookupError) as decode_err:
                 logger.debug(f"Decoding attempt failed for {log_url} with encoding '{encoding}': {decode_err}")
        else:
            logger.error(f"Failed to decode content for {log_url} using attempted encodings: {encodings_to_try}")
            return None

        if not html_content_str.strip():
             logger.warning(f"Decoded content is empty or whitespace for {log_url} (used encoding: {final_encoding_used}).")
             return None

        sanitized_content = sanitize_scraped_content(html_content_str, url=log_url)
        sanitized_size = len(sanitized_content.encode('utf-8'))

        if not sanitized_content or len(sanitized_content) < config.MIN_SANITIZED_CONTENT_LENGTH:
            logger.debug(f"Content too short ({len(sanitized_content)} chars) or empty after sanitization for {log_url}. Min required: {config.MIN_SANITIZED_CONTENT_LENGTH}. Skipping.")
            return None

        try:
            fd, temp_filepath = tempfile.mkstemp(
                suffix=".txt", prefix="sanitized_", dir=config.TEMP_FILE_DIR, text=True
            )
            with os.fdopen(fd, 'w', encoding='utf-8') as temp_file_obj:
                temp_file_obj.write(sanitized_content)

            if not os.path.exists(temp_filepath) or os.path.getsize(temp_filepath) == 0:
                 logger.error(f"Failed to write or created empty temp file for {log_url} at {temp_filepath}")
                 if temp_filepath and os.path.exists(temp_filepath): os.remove(temp_filepath)
                 temp_filepath = None
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
            if temp_filepath and os.path.exists(temp_filepath):
                try: os.remove(temp_filepath)
                except OSError as e: logger.error(f"Failed to cleanup temp file {temp_filepath} after IO error: {e}")
            temp_filepath = None
            return None
        except Exception as file_err:
            logger.error(f"Unexpected error writing temp file for {log_url}: {file_err}", exc_info=True)
            if temp_filepath and os.path.exists(temp_filepath):
                try: os.remove(temp_filepath)
                except OSError as e: logger.error(f"Failed to cleanup temp file {temp_filepath} after unexpected error: {e}")
            temp_filepath = None
            return None

    except requests.exceptions.Timeout as e:
        logger.warning(f"Timeout error fetching URL {log_url}: {e}")
    except requests.exceptions.HTTPError as e:
        if 400 <= e.response.status_code < 500:
             logger.warning(f"HTTP Client Error {e.response.status_code} fetching URL {log_url}: {e}")
        else:
            logger.error(f"HTTP Server Error {e.response.status_code} fetching URL {log_url}: {e}")
    except requests.exceptions.ConnectionError as e:
         logger.warning(f"Connection Error fetching URL {log_url}: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request Error fetching URL {log_url}: {type(e).__name__} - {e}", exc_info=False)
    except Exception as e:
        logger.error(f"Unexpected Error processing URL {log_url}: {type(e).__name__} - {e}", exc_info=True)

    finally:
        if response is not None and hasattr(response, 'close') and not getattr(response, '_content_consumed', True):
            try: response.close()
            except Exception: pass

    return None