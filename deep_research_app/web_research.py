# web_research.py
import time
import re
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import concurrent.futures
import config

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
            with DDGS(timeout=15) as ddgs:
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
        else:
            print(f"  -> {provider_result['engine']} search failed: {provider_result['error']}")
            search_errors.append(f"{provider_result['engine']}: {provider_result['error']}")

        # Delay between different search providers if needed
        if len(search_providers) > 1 and i < len(search_providers) - 1:
            time.sleep(config.INTER_SEARCH_DELAY_SECONDS)

    return list(all_unique_urls), search_errors


def scrape_url(url):
    """
    Scrapes text content from a given URL. Handles common errors.
    Returns a dictionary {'url': url, 'content': text} on success, None on failure.
    """
    log_url = url[:75] + '...' if len(url) > 75 else url
    response = None # Ensure response is defined for finally block
    try:
        headers = {'User-Agent': config.USER_AGENT}
        # Use stream=True to read headers first, then decide to download content
        response = requests.get(url, headers=headers, timeout=config.REQUEST_TIMEOUT, allow_redirects=True, stream=True)
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type and 'text' not in content_type:
            # print(f"Skipping non-HTML/text URL {log_url} (Content-Type: {content_type})")
            return None # Skip non-HTML/text content

        content_length_str = response.headers.get('content-length')
        max_bytes = config.MAX_SCRAPE_CONTENT_LENGTH_MB * 1024 * 1024

        # Check content length *before* downloading the whole body if possible
        if content_length_str and int(content_length_str) > max_bytes:
            # print(f"Skipping large file {log_url} (Content-Length: {content_length_str} bytes)")
            return None

        # Download content now (up to a limit if length wasn't available)
        html_content = b""
        for chunk in response.iter_content(chunk_size=8192):
             html_content += chunk
             if len(html_content) > max_bytes:
                  # print(f"Stopping download for {log_url} - exceeded {config.MAX_SCRAPE_CONTENT_LENGTH_MB}MB")
                  return None # Stop downloading if it exceeds limit

        # Close the response connection explicitly after consuming content or error
        response.close()

        # Parse the HTML
        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except Exception: # Fallback parser
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
            except Exception as parse_err:
                print(f"Failed to parse HTML for {log_url} with both parsers: {parse_err}")
                return None

        # Remove unwanted tags
        for element in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "input", "textarea", "select", "img", "figure", "iframe", "video", "audio", "picture", "source", "noscript", "meta", "link", "svg", "canvas", "map", "area"]):
            element.decompose()

        # Attempt to find main content area
        main_content = (
            soup.find('main') or
            soup.find('article') or
            soup.find('div', attrs={'role': 'main'}) or
            soup.find('section', id='content') or
            soup.find('div', id='content') or
            soup.find('div', class_=re.compile(r'\b(content|main|post|entry|article[-_]body|body[-_]content)\b', re.I)) or
            soup.find('div', id=re.compile(r'\b(content|main)\b', re.I))
        )

        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
             # Fallback to body if main content heuristics fail
             body = soup.find('body')
             text = body.get_text(separator='\n', strip=True) if body else soup.get_text(separator='\n', strip=True)


        # Clean up extracted text: remove excessive whitespace and very short lines
        lines = (line.strip() for line in text.splitlines())
        # Split by multiple spaces, handle phrases within lines, join with single space
        chunks = (' '.join(phrase.split()) for line in lines for phrase in line.split("  ") if phrase.strip())
        # Join non-empty chunks with newlines, filter short chunks (likely remnants)
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk and len(chunk.split()) > 2) # Keep lines with > 2 words

        # Final check for minimum meaningful content
        meaningful_word_count = len([word for word in cleaned_text.split() if len(word) > 1])
        if meaningful_word_count < config.MIN_MEANINGFUL_WORDS_PER_PAGE:
             # print(f"Skipping {log_url} - insufficient meaningful content ({meaningful_word_count} words)")
             return None

        return {'url': url, 'content': cleaned_text}

    except requests.exceptions.Timeout:
        print(f"Timeout error fetching URL {log_url}")
    except requests.exceptions.HTTPError as e:
        # Log client errors (4xx) potentially differently than server errors (5xx)
        if 400 <= e.response.status_code < 500:
            # print(f"Client Error {e.response.status_code} fetching URL {log_url}: {e}")
            pass # Often expected (404, 403), suppress verbose logging unless debugging
        else:
            print(f"HTTP Server Error {e.response.status_code} fetching URL {log_url}: {e}")
    except requests.exceptions.RequestException as e:
        # Catch other request errors (DNS, connection, etc.)
        print(f"Request Error fetching URL {log_url}: {e}")
    except Exception as e:
        # Catch-all for unexpected errors during scraping/parsing
        print(f"Unexpected Error processing URL {log_url}: {type(e).__name__} - {e}")
    finally:
        # Ensure the response connection is closed if it was opened
        if response is not None:
             try:
                  response.close()
             except Exception:
                  pass # Ignore errors during close

    return None # Return None if any error occurred