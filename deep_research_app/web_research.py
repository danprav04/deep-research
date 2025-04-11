# web_research.py
import time
import re
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import concurrent.futures
import config
import tempfile # Added
import os       # Added

# --- Existing search functions (search_duckduckgo_provider, perform_web_search) remain the same ---

def search_duckduckgo_provider(keywords, max_results=config.MAX_SEARCH_RESULTS_PER_ENGINE_STEP):
    """Performs a search on DuckDuckGo with retry."""
    query = " ".join(keywords)
    urls = []
    if not query:
        return {"engine": "DuckDuckGo", "urls": [], "success": True, "error": None}

    # DDGS has its own retry/backoff, but we add a layer for explicit control/logging
    max_retries = 2 # Keep a local retry count for this specific function's logic
    for attempt in range(max_retries + 1):
        try:
            # Increase timeout slightly as DDGS can sometimes be slow
            with DDGS(timeout=20) as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                urls = [r['href'] for r in results if r and 'href' in r]
                return {"engine": "DuckDuckGo", "urls": urls, "success": True, "error": None}
        except Exception as e:
            # Check for common rate limit indicators
            is_rate_limit = "Ratelimit" in str(e) or "429" in str(e) or "202" in str(e)
            if is_rate_limit and attempt < max_retries:
                current_delay = config.DDGS_RETRY_DELAY_SECONDS * (2 ** attempt) # Exponential backoff
                print(f"  -> DuckDuckGo Rate limit detected for '{query}'. Retrying attempt {attempt + 1}/{max_retries} in {current_delay:.1f}s...")
                time.sleep(current_delay)
                continue
            else:
                err_msg = f"Error searching DuckDuckGo for '{query}' (Attempt {attempt+1}): {e}"
                if is_rate_limit:
                    err_msg = f"Max retries reached for DuckDuckGo rate limit on '{query}'."
                print(f"  -> {err_msg}")
                return {"engine": "DuckDuckGo", "urls": [], "success": False, "error": err_msg}

    # Should not be reached if logic is correct, but provides a fallback
    return {"engine": "DuckDuckGo", "urls": [], "success": False, "error": "Max retries reached unexpectedly."}

def perform_web_search(keywords):
    """
    Performs searches using configured providers and aggregates unique results.
    """
    all_unique_urls = set()
    search_errors = []

    # Define the search providers to use
    search_providers = [
        search_duckduckgo_provider,
        # Add other provider functions here if implemented
    ]

    for i, provider_func in enumerate(search_providers):
        # Pass the configured max results from config
        provider_result = provider_func(keywords, max_results=config.MAX_SEARCH_RESULTS_PER_ENGINE_STEP)

        if provider_result["success"]:
            new_urls = set(provider_result["urls"]) - all_unique_urls
            all_unique_urls.update(provider_result["urls"])
            # print(f"  -> {provider_result['engine']} found {len(provider_result['urls'])} URLs ({len(new_urls)} new).")
        else:
            print(f"  -> {provider_result['engine']} search failed: {provider_result['error']}")
            search_errors.append(f"{provider_result['engine']}: {provider_result['error']}")

        # Delay between different search providers if needed
        if len(search_providers) > 1 and i < len(search_providers) - 1:
             time.sleep(config.INTER_SEARCH_DELAY_SECONDS / 2) # Slightly shorter delay might be ok

    return list(all_unique_urls), search_errors

# --- Modified scrape_url function ---
def scrape_url(url):
    """
    Scrapes text content from a given URL. Handles common errors.
    Saves content to a temporary file.
    Returns a dictionary {'url': url, 'temp_filepath': path} on success, None on failure.
    """
    log_url = url[:75] + '...' if len(url) > 75 else url
    response = None
    temp_file = None
    temp_filepath = None

    try:
        headers = {'User-Agent': config.USER_AGENT}
        response = requests.get(url, headers=headers, timeout=config.REQUEST_TIMEOUT, allow_redirects=True, stream=True)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type and 'text' not in content_type:
            return None

        content_length_str = response.headers.get('content-length')
        max_bytes = config.MAX_SCRAPE_CONTENT_LENGTH_MB * 1024 * 1024

        if content_length_str and int(content_length_str) > max_bytes:
            return None

        html_content = b""
        for chunk in response.iter_content(chunk_size=8192):
            html_content += chunk
            if len(html_content) > max_bytes:
                return None
        response.close() # Close connection after reading

        # Decode carefully
        try:
            detected_encoding = response.apparent_encoding or 'utf-8' # Use detected or fallback
            html_content_str = html_content.decode(detected_encoding, errors='replace')
        except Exception as decode_err:
            print(f"Decoding error for {log_url}: {decode_err}. Trying utf-8 fallback.")
            html_content_str = html_content.decode('utf-8', errors='replace')


        try:
            soup = BeautifulSoup(html_content_str, 'lxml')
        except Exception:
            try:
                soup = BeautifulSoup(html_content_str, 'html.parser')
            except Exception as parse_err:
                print(f"Failed to parse HTML for {log_url}: {parse_err}")
                return None

        for element in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "input", "textarea", "select", "img", "figure", "iframe", "video", "audio", "picture", "source", "noscript", "meta", "link", "svg", "canvas", "map", "area"]):
            element.decompose()

        main_content = (
            soup.find('main') or
            soup.find('article') or
            soup.find('div', attrs={'role': 'main'}) or
            # Relaxed content selection slightly if specific tags aren't found
            soup.find('div', class_=re.compile(r'\b(content|main|post|entry|article[-_]body|body[-_]content)\b', re.I)) or
            soup.find('div', id=re.compile(r'\b(content|main)\b', re.I))
        )

        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            body = soup.find('body')
            text = body.get_text(separator='\n', strip=True) if body else soup.get_text(separator='\n', strip=True)

        lines = (line.strip() for line in text.splitlines())
        chunks = (' '.join(phrase.split()) for line in lines for phrase in line.split("  ") if phrase.strip())
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk and len(chunk.split()) > 2)

        meaningful_word_count = len([word for word in cleaned_text.split() if len(word) > 1])
        if meaningful_word_count < config.MIN_MEANINGFUL_WORDS_PER_PAGE:
            return None

        # --- Save to temporary file ---
        if cleaned_text:
            try:
                # Create a temporary file that won't be deleted automatically
                # Use 'w+' for writing and potentially reading later if needed, though we only write here
                temp_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix=".txt", prefix="scraped_")
                temp_filepath = temp_file.name
                temp_file.write(cleaned_text)
                temp_file.close() # Close the file handle immediately after writing
                # print(f"  -> Scraped {log_url} to {temp_filepath}") # Debug log
                return {'url': url, 'temp_filepath': temp_filepath}
            except Exception as file_err:
                print(f"Error writing temp file for {log_url}: {file_err}")
                # Clean up if file was created but write failed or other error occurred
                if temp_filepath and os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                    except OSError: pass # Ignore cleanup error
                return None
        else:
            # No meaningful content extracted
            return None

    except requests.exceptions.Timeout:
        print(f"Timeout error fetching URL {log_url}")
    except requests.exceptions.HTTPError as e:
        if 400 <= e.response.status_code < 500:
             pass # print(f"Client Error {e.response.status_code} fetching URL {log_url}")
        else:
            print(f"HTTP Server Error {e.response.status_code} fetching URL {log_url}: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request Error fetching URL {log_url}: {e}")
    except Exception as e:
        print(f"Unexpected Error processing URL {log_url}: {type(e).__name__} - {e}")
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        # Ensure temp file handle is closed if scrape_url exits unexpectedly before close
        if temp_file is not None and not temp_file.closed:
             try:
                 temp_file.close()
             except Exception: pass
             # If an error occurred *before* successfully returning the dict,
             # we should clean up the file if it exists.
             if temp_filepath and os.path.exists(temp_filepath):
                  # Check if the function is returning successfully or not.
                  # This check is tricky here. Best practice is manual cleanup in app.py.
                  # We *don't* remove it here if we plan to return the path.
                  pass


    return None # Return None if any error occurred before successful temp file write