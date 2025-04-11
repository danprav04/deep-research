# web_research.py
import time
import re
import requests
import concurrent.futures
import tempfile
import os
import logging
from typing import List, Dict, Tuple, Optional, Set, Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup, SoupStrainer
from duckduckgo_search import DDGS

import config as config

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Search Functions ---
# search_duckduckgo_provider and perform_web_search remain unchanged from the previous fix
def search_duckduckgo_provider(keywords: List[str], max_results: int) -> Dict[str, Any]:
    """
    Performs a search on DuckDuckGo using the duckduckgo_search library with retry logic.

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
        return {"engine": engine_name, "urls": [], "success": True, "error": None}

    max_outer_retries = 2
    base_delay = config.DDGS_RETRY_DELAY_SECONDS

    logger.info(f"Searching {engine_name} for: '{query}' (max_results={max_results})")

    for attempt in range(max_outer_retries + 1):
        error_msg = "Unknown error" # Default error message for the attempt
        is_rate_limit = False # Flag for rate limit detection
        try:
            with DDGS(timeout=20) as ddgs:
                 results_iterator = ddgs.text(query, max_results=max_results)
                 urls = [r['href'] for r in results_iterator if r and isinstance(r, dict) and r.get('href')]
                 logger.info(f"{engine_name} search for '{query}' successful, found {len(urls)} URLs.")
                 return {"engine": engine_name, "urls": urls, "success": True, "error": None}

        except Exception as e:
            # Catch potential exceptions (network errors, timeouts, library internal errors)
            error_msg = f"Error searching {engine_name} for '{query}' (Attempt {attempt+1}): {type(e).__name__} - {e}"
            error_str_lower = str(e).lower()
            is_rate_limit = "rate limit" in error_str_lower or "429" in error_str_lower or "too many requests" in error_str_lower
            logger.warning(error_msg, exc_info=False) # Log as warning

        # --- Retry Logic ---
        if attempt < max_outer_retries:
             delay = base_delay * (2 ** attempt) # Exponential backoff
             log_prefix = "Rate limit suspected" if is_rate_limit else "Error encountered"
             logger.info(f"  -> {log_prefix} for {engine_name} search '{query}'. Retrying attempt {attempt + 2}/{max_outer_retries + 1} in {delay:.1f}s...")
             time.sleep(delay)
        else:
             # Max retries reached
             final_error_msg = f"Max retries reached for {engine_name} search '{query}'. Last error: {error_msg}"
             logger.error(final_error_msg)
             return {"engine": engine_name, "urls": [], "success": False, "error": final_error_msg}

    # Fallback if loop finishes unexpectedly
    logger.error(f"Unexpected exit from {engine_name} retry loop for query '{query}'.")
    return {"engine": engine_name, "urls": [], "success": False, "error": "Max retries reached unexpectedly."}


def perform_web_search(keywords: List[str]) -> Tuple[List[str], List[str]]:
    """
    Performs searches using configured providers and aggregates unique results.

    Args:
        keywords: List of keywords for the search query.

    Returns:
        A tuple containing:
        - List of unique URLs found across all providers.
        - List of error messages encountered during searches.
    """
    all_unique_urls: Set[str] = set()
    search_errors: List[str] = []

    # Define the search providers to use (currently only DuckDuckGo)
    search_providers = [
        search_duckduckgo_provider,
    ]

    for i, provider_func in enumerate(search_providers):
        provider_result = provider_func(keywords, max_results=config.MAX_SEARCH_RESULTS_PER_ENGINE_STEP)
        engine = provider_result.get("engine", "Unknown Engine")
        if provider_result.get("success"):
            provider_urls = provider_result.get("urls", [])
            new_urls_found = len(set(provider_urls) - all_unique_urls)
            all_unique_urls.update(provider_urls)
            logger.debug(f"  -> {engine} found {len(provider_urls)} URLs ({new_urls_found} new). Total unique: {len(all_unique_urls)}")
        else:
            error_msg = provider_result.get("error", "Unknown search error")
            logger.warning(f"  -> {engine} search failed: {error_msg}")
            search_errors.append(f"{engine}: {error_msg}")

    logger.info(f"Web search for keywords '{' '.join(keywords)}' completed. Found {len(all_unique_urls)} unique URLs. Encountered {len(search_errors)} errors.")
    return list(all_unique_urls), search_errors


# --- Scraping Function ---

def _clean_text(soup: BeautifulSoup) -> str:
    """Extracts and cleans text content from a BeautifulSoup object."""
    # Remove common clutter elements
    for element in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "input", "textarea", "select", "img", "figure", "iframe", "video", "audio", "picture", "source", "noscript", "meta", "link", "svg", "canvas", "map", "area", "label"]):
        element.decompose()

    # Attempt to find main content areas
    main_content_selectors = [
        'main', 'article', {'role': 'main'},
        {'class': re.compile(r'\b(content|main|post|entry|article[-_]body|body[-_]content)\b', re.I)},
        {'id': re.compile(r'\b(content|main|story|article)\b', re.I)}
    ]
    text_content = ""
    for selector in main_content_selectors:
        main_area = soup.find(selector)
        if main_area:
            text_content = main_area.get_text(separator='\n', strip=True)
            break
    else:
        body = soup.find('body')
        if body:
            text_content = body.get_text(separator='\n', strip=True)
        else:
            text_content = soup.get_text(separator='\n', strip=True)

    # Clean up extracted text
    lines = (line.strip() for line in text_content.splitlines())
    chunks = [chunk for line in lines for phrase in line.split("  ") if (chunk := phrase.strip())]
    cleaned_text = '\n'.join(chunk for chunk in chunks if len(chunk.split()) > 2)

    return cleaned_text


def scrape_url(url: str) -> Optional[Dict[str, str]]:
    """
    Scrapes text content from a given URL, saves it to a temporary file,
    and returns metadata including the temp file path.

    Args:
        url: The URL to scrape.

    Returns:
        A dictionary {'url': url, 'temp_filepath': path} on success,
        None if scraping fails, content is unsuitable, or an error occurs.
    """
    log_url = url[:75] + '...' if len(url) > 75 else url
    response: Optional[requests.Response] = None
    temp_file_obj = None
    temp_filepath: Optional[str] = None

    try:
        headers = {'User-Agent': config.USER_AGENT, 'Accept': 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.8'}
        response = requests.get(
            url,
            headers=headers,
            timeout=config.REQUEST_TIMEOUT,
            allow_redirects=True,
            stream=True
        )
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if not ('html' in content_type or 'text' in content_type or 'xml' in content_type):
            logger.debug(f"Skipping non-HTML/text URL ({content_type}): {log_url}")
            response.close()
            return None

        # --- Stream content download and check size limit ---
        html_content_bytes = b""
        bytes_read = 0
        max_bytes = config.MAX_SCRAPE_CONTENT_BYTES

        # ** Store encoding from headers before consuming stream **
        # response.encoding is based on headers (e.g., Content-Type: ...; charset=...)
        # It might be None if no charset is specified in headers.
        encoding_from_headers = response.encoding

        for chunk in response.iter_content(chunk_size=8192):
            if not chunk: continue
            bytes_read += len(chunk)
            if bytes_read > max_bytes:
                logger.warning(f"Content exceeds size limit ({max_bytes / 1024 / 1024:.1f} MB) for {log_url}. Skipping.")
                response.close()
                return None
            html_content_bytes += chunk

        # Close the response connection now that content is read
        response.close()
        logger.debug(f"Downloaded {bytes_read / 1024:.1f} KB for {log_url}")

        # --- Decode content carefully ---
        html_content_str = ""
        # ** Determine encoding to use **
        # Priority: 1. Encoding from headers, 2. Fallback to UTF-8
        final_encoding = encoding_from_headers if encoding_from_headers else 'utf-8'
        logger.debug(f"Attempting decode using final encoding: {final_encoding} for {log_url}")

        try:
            html_content_str = html_content_bytes.decode(final_encoding, errors='replace')
        except (UnicodeDecodeError, LookupError) as decode_err:
            logger.warning(f"Decoding error ({decode_err}) for {log_url} with encoding '{final_encoding}'.")
            # If the primary attempt failed (e.g., header encoding was wrong), try UTF-8 as a last resort if not already tried.
            if final_encoding != 'utf-8':
                logger.info(f"Trying UTF-8 fallback decoding for {log_url}")
                try:
                    html_content_str = html_content_bytes.decode('utf-8', errors='replace')
                    final_encoding = 'utf-8 (fallback)' # Note fallback was used
                except Exception as fallback_decode_err:
                    logger.error(f"UTF-8 fallback decoding also failed for {log_url}: {fallback_decode_err}")
                    return None # Cannot decode content
            else:
                # If UTF-8 was the primary attempt and failed, we can't decode
                logger.error(f"Primary decoding failed ({final_encoding}), no further fallback possible for {log_url}")
                return None


        if not html_content_str.strip():
             logger.warning(f"Decoded content is empty or whitespace for {log_url} (used encoding: {final_encoding}).")
             return None

        # --- Parse HTML using BeautifulSoup ---
        try:
            soup = BeautifulSoup(html_content_str, 'lxml')
        except Exception:
             try:
                 logger.debug(f"lxml parser failed or not available for {log_url}, trying html.parser.")
                 soup = BeautifulSoup(html_content_str, 'html.parser')
             except Exception as parse_err:
                 logger.error(f"Failed to parse HTML for {log_url} with both parsers: {parse_err}", exc_info=False)
                 return None

        # --- Extract and clean text ---
        cleaned_text = _clean_text(soup)

        # --- Check if meaningful content was extracted ---
        meaningful_word_count = len([word for word in cleaned_text.split() if len(word) > 1])
        if meaningful_word_count < config.MIN_MEANINGFUL_WORDS_PER_PAGE:
            logger.debug(f"Extracted content too short ({meaningful_word_count} words) for {log_url}. Skipping.")
            return None

        # --- Save cleaned text to a temporary file ---
        if cleaned_text:
            try:
                temp_file_obj = tempfile.NamedTemporaryFile(
                    mode='w', encoding='utf-8', delete=False,
                    suffix=".txt", prefix="scraped_", dir=config.TEMP_FILE_DIR
                )
                temp_filepath = temp_file_obj.name
                temp_file_obj.write(cleaned_text)
                temp_file_obj.close()

                file_size_kb = os.path.getsize(temp_filepath) / 1024
                logger.info(f"Successfully scraped {log_url} -> {os.path.basename(temp_filepath)} ({file_size_kb:.1f} KB)")
                return {'url': url, 'temp_filepath': temp_filepath}

            except Exception as file_err:
                logger.error(f"Error writing temp file for {log_url}: {file_err}", exc_info=True)
                if temp_filepath and os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                    except OSError: pass
                return None
        else:
            logger.debug(f"No meaningful content extracted after cleaning for {log_url}.")
            return None

    # --- Exception Handling ---
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout error fetching URL {log_url}")
    except requests.exceptions.HTTPError as e:
        if 400 <= e.response.status_code < 500:
             logger.warning(f"HTTP Client Error {e.response.status_code} fetching URL {log_url}: {e}")
        else:
            logger.error(f"HTTP Server Error {e.response.status_code} fetching URL {log_url}: {e}")
    except requests.exceptions.ConnectionError as e:
         logger.warning(f"Connection Error fetching URL {log_url}: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request Error fetching URL {log_url}: {e}", exc_info=False)
    except Exception as e:
        logger.error(f"Unexpected Error processing URL {log_url}: {type(e).__name__} - {e}", exc_info=True)

    # --- Cleanup in case of error ---
    finally:
        if response is not None:
            try: response.close()
            except Exception: pass
        if temp_file_obj is not None and not temp_file_obj.closed:
             try: temp_file_obj.close()
             except Exception: pass

    return None # Return None if any error prevented successful scraping and saving