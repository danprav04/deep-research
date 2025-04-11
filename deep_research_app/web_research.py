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
# Remove DuckDuckGoSearchException from import
from duckduckgo_search import DDGS

import config as config

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Search Functions ---

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

        # REMOVED the specific except DuckDuckGoSearchException block

        except Exception as e:
            # Catch other potential exceptions (network errors, timeouts, library internal errors)
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


# --- The rest of the web_research.py file remains the same ---
# perform_web_search, _clean_text, scrape_url functions etc.
# ... (keep the existing code below this point) ...

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
    # Add more functions like search_google_provider, search_bing_provider here if implemented
    search_providers = [
        search_duckduckgo_provider,
    ]

    # Can run providers concurrently if needed, but often sequential is fine for a few providers
    # and avoids hitting rate limits across multiple services simultaneously.
    for i, provider_func in enumerate(search_providers):
        # Pass the configured max results per step
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

        # Optional delay between different search providers if using multiple
        # if len(search_providers) > 1 and i < len(search_providers) - 1:
        #      time.sleep(config.INTER_PROVIDER_SEARCH_DELAY) # Add this to config if needed

    logger.info(f"Web search for keywords '{' '.join(keywords)}' completed. Found {len(all_unique_urls)} unique URLs. Encountered {len(search_errors)} errors.")
    return list(all_unique_urls), search_errors


# --- Scraping Function ---

def _clean_text(soup: BeautifulSoup) -> str:
    """Extracts and cleans text content from a BeautifulSoup object."""
    # Remove common clutter elements
    for element in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "input", "textarea", "select", "img", "figure", "iframe", "video", "audio", "picture", "source", "noscript", "meta", "link", "svg", "canvas", "map", "area", "label"]):
        element.decompose()

    # Attempt to find main content areas (adapt selectors as needed)
    main_content_selectors = [
        'main',
        'article',
        {'role': 'main'},
        {'class': re.compile(r'\b(content|main|post|entry|article[-_]body|body[-_]content)\b', re.I)},
        {'id': re.compile(r'\b(content|main|story|article)\b', re.I)}
    ]
    text_content = ""
    for selector in main_content_selectors:
        main_area = soup.find(selector)
        if main_area:
            # logger.debug(f"Found main content area using selector: {selector}")
            text_content = main_area.get_text(separator='\n', strip=True)
            break # Use the first main content area found
    else:
        # Fallback to using the whole body if no specific main area is found
        body = soup.find('body')
        if body:
            # logger.debug("No specific main content area found, falling back to body.")
            text_content = body.get_text(separator='\n', strip=True)
        else:
            # Final fallback: all text if body tag is missing (unlikely for valid HTML)
            logger.debug("No body tag found, falling back to all text.")
            text_content = soup.get_text(separator='\n', strip=True)


    # Clean up extracted text: remove excessive whitespace/newlines
    lines = (line.strip() for line in text_content.splitlines())
    # Further break down lines and remove empty chunks
    chunks = [chunk for line in lines for phrase in line.split("  ") if (chunk := phrase.strip())]
    # Join chunks with single newlines, filtering out very short chunks (likely noise)
    cleaned_text = '\n'.join(chunk for chunk in chunks if len(chunk.split()) > 2) # Keep chunks with more than 2 words

    return cleaned_text


def scrape_url(url: str) -> Optional[Dict[str, str]]:
    """
    Scrapes text content from a given URL, saves it to a temporary file,
    and returns metadata including the temp file path.

    Designed to be memory-efficient for large pages by streaming the download
    and limiting the total size read.

    Args:
        url: The URL to scrape.

    Returns:
        A dictionary {'url': url, 'temp_filepath': path} on success,
        None if scraping fails, content is unsuitable, or an error occurs.
        The caller is responsible for deleting the temporary file using the returned path.
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
            stream=True # <<< Critical for memory efficiency >>>
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Check Content-Type - Ensure it's likely HTML or text
        content_type = response.headers.get('content-type', '').lower()
        if not ('html' in content_type or 'text' in content_type or 'xml' in content_type):
            logger.debug(f"Skipping non-HTML/text URL ({content_type}): {log_url}")
            response.close() # Close connection
            return None

        # --- Stream content download and check size limit ---
        html_content_bytes = b""
        bytes_read = 0
        max_bytes = config.MAX_SCRAPE_CONTENT_BYTES

        for chunk in response.iter_content(chunk_size=8192): # Read in chunks
            if not chunk: # Handle empty chunks if they occur
                continue
            bytes_read += len(chunk)
            if bytes_read > max_bytes:
                logger.warning(f"Content exceeds size limit ({max_bytes / 1024 / 1024:.1f} MB) for {log_url}. Skipping.")
                response.close() # Important to close the connection
                return None
            html_content_bytes += chunk

        # Close the response connection now that content is read (or limit exceeded)
        response.close()
        logger.debug(f"Downloaded {bytes_read / 1024:.1f} KB for {log_url}")

        # --- Decode content carefully ---
        html_content_str = ""
        try:
            # Use apparent_encoding (from headers/meta tags) or fallback to utf-8
            detected_encoding = response.apparent_encoding or 'utf-8'
            html_content_str = html_content_bytes.decode(detected_encoding, errors='replace')
            logger.debug(f"Decoded content using {detected_encoding} for {log_url}")
        except (UnicodeDecodeError, LookupError) as decode_err:
            logger.warning(f"Decoding error ({decode_err}) for {log_url} with encoding '{detected_encoding}'. Trying utf-8 fallback.")
            try:
                html_content_str = html_content_bytes.decode('utf-8', errors='replace')
                logger.debug(f"Successfully decoded content using utf-8 fallback for {log_url}")
            except Exception as fallback_decode_err:
                logger.error(f"UTF-8 fallback decoding failed for {log_url}: {fallback_decode_err}")
                return None # Cannot decode content

        if not html_content_str.strip():
             logger.warning(f"Decoded content is empty or whitespace for {log_url}.")
             return None

        # --- Parse HTML using BeautifulSoup ---
        # Optional optimization: Use SoupStrainer to parse only specific parts if known
        # parse_only = SoupStrainer(['main', 'article', 'body']) # Example
        try:
            # Use lxml for speed if available, fallback to html.parser
            soup = BeautifulSoup(html_content_str, 'lxml') # , parse_only=parse_only)
        except Exception: # Catch errors during parsing (e.g., lxml not installed or malformed HTML)
             try:
                 logger.debug(f"lxml parser failed or not available for {log_url}, trying html.parser.")
                 soup = BeautifulSoup(html_content_str, 'html.parser') # , parse_only=parse_only)
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
                # Create a temporary file in the configured directory
                # delete=False means the file persists after closing, caller must delete it.
                temp_file_obj = tempfile.NamedTemporaryFile(
                    mode='w',
                    encoding='utf-8',
                    delete=False, # <<< We need the path, so don't auto-delete
                    suffix=".txt",
                    prefix="scraped_",
                    dir=config.TEMP_FILE_DIR # Use configured temp directory
                )
                temp_filepath = temp_file_obj.name
                temp_file_obj.write(cleaned_text)
                temp_file_obj.close() # Close the file handle immediately after writing

                file_size_kb = os.path.getsize(temp_filepath) / 1024
                logger.info(f"Successfully scraped {log_url} -> {os.path.basename(temp_filepath)} ({file_size_kb:.1f} KB)")
                return {'url': url, 'temp_filepath': temp_filepath}

            except Exception as file_err:
                logger.error(f"Error writing temp file for {log_url}: {file_err}", exc_info=True)
                # Clean up if file was created but write/close failed
                if temp_filepath and os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                        logger.info(f"Cleaned up partially created temp file: {temp_filepath}")
                    except OSError as cleanup_err:
                        logger.error(f"Failed to clean up temp file {temp_filepath} after write error: {cleanup_err}")
                return None # Failed to save content
        else:
            # No meaningful content extracted after cleaning
            logger.debug(f"No meaningful content extracted after cleaning for {log_url}.")
            return None

    # --- Exception Handling ---
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout error fetching URL {log_url}")
    except requests.exceptions.HTTPError as e:
        # Log client errors (4xx) as warnings, server errors (5xx) as errors
        if 400 <= e.response.status_code < 500:
             logger.warning(f"HTTP Client Error {e.response.status_code} fetching URL {log_url}: {e}")
        else:
            logger.error(f"HTTP Server Error {e.response.status_code} fetching URL {log_url}: {e}")
    except requests.exceptions.ConnectionError as e:
         logger.warning(f"Connection Error fetching URL {log_url}: {e}")
    except requests.exceptions.RequestException as e:
        # Catch other potential request errors (e.g., invalid URL, redirects)
        logger.error(f"Request Error fetching URL {log_url}: {e}", exc_info=False)
    except Exception as e:
        # Catch any other unexpected error during the process
        logger.error(f"Unexpected Error processing URL {log_url}: {type(e).__name__} - {e}", exc_info=True)

    # --- Cleanup in case of error ---
    finally:
        # Ensure response is closed if it exists
        if response is not None:
            try:
                response.close()
            except Exception:
                pass # Ignore errors during close
        # Ensure temp file handle is closed if opened but not closed due to error
        if temp_file_obj is not None and not temp_file_obj.closed:
             try:
                 temp_file_obj.close()
             except Exception: pass

    return None # Return None if any error prevented successful scraping and saving