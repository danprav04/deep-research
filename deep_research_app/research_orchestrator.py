# --- File: deep_research_app/research_orchestrator.py ---
# research_orchestrator.py
import time
import json
import os
import traceback
import logging
import re
from typing import Dict, Any, List, Callable, Optional, Tuple, Generator, Set
from html import escape
import concurrent.futures
from urllib.parse import urlparse
from google.api_core import exceptions as google_api_exceptions # Import specific exception
from collections import defaultdict # Used for grouping sources by step

import config as config
from llm_interface import call_gemini, stream_gemini # Assume RateLimitExceededError might be raised by call_gemini if not handled internally
from web_research import perform_web_search, scrape_url
from utils import (
    parse_research_plan, generate_bibliography_map, convert_markdown_to_html
)

# Configure logger for this module
logger = logging.getLogger(__name__)

# Define yielded event structures for clarity
ProgressEvent = Dict[str, Any] # keys: type='progress', message, is_error, is_fatal
DataEvent = Dict[str, Any]     # keys: type='event', data={...}
ScrapeSuccessEvent = Dict[str, Any] # keys: type='scrape_success', metadata={...} (metadata now includes step_index)
LlmStatusEvent = Dict[str, Any] # keys: type='event', data={'type': 'llm_status', 'status': 'busy'|'retrying', 'message': str}

# Basic Topic Validation Pattern
VALID_TOPIC_PATTERN = re.compile(r"^[a-zA-Z0-9\s.,!?'\"\-():&]+$")
MAX_TOPIC_LENGTH = 300

def run_research_process(topic: str) -> Generator[Dict[str, Any], None, None]:
    """
    Executes the entire research process as a generator, yielding events.
    Includes persistent retry logic for LLM rate limit errors during plan generation.
    Context preparation for synthesis now attempts to balance sources across research steps.

    Yields dictionaries representing different stages and data:
    - {'type': 'progress', 'message': str, 'is_error': bool, 'is_fatal': bool}
    - {'type': 'event', 'data': Dict[str, Any]} (e.g., llm_chunk, stream_start, complete, llm_status)
    - {'type': 'scrape_success', 'metadata': Dict[str, str]} (incl. temp_filepath and step_index)

    Args:
        topic: The research topic.

    Returns:
        None. The function yields events until completion or fatal error.
    """
    # --- Research state variables ---
    # grouped_scraped_metadata: Dict[int, List[Dict[str, Any]]] = defaultdict(list) # Stores successful scrapes, grouped by step index
    research_plan: List[Dict[str, Any]] = []
    accumulated_synthesis_md: str = ""
    final_report_markdown: str = ""
    # url_to_index_map: Dict[str, int] = {} # Will be generated later from *all* successful scrapes
    all_successful_scrapes: List[Dict[str, Any]] = [] # Flat list of all successful scrape metadata dicts
    start_time_total = time.time()
    fatal_error_occurred = False

    # --- Topic Validation ---
    if not topic or not topic.strip():
        logger.error("Orchestrator received empty topic.")
        yield {'type': 'progress', 'message': "Fatal Error: Research topic cannot be empty.", 'is_error': True, 'is_fatal': True}
        return
    if len(topic) > MAX_TOPIC_LENGTH:
        logger.warning(f"Topic length ({len(topic)}) exceeds max ({MAX_TOPIC_LENGTH}). Truncating.")
        topic = topic[:MAX_TOPIC_LENGTH].strip()
        yield {'type': 'progress', 'message': f"Warning: Topic was too long and has been truncated to: '{topic}'", 'is_error': True, 'is_fatal': False}

    logger.info(f"Starting research process for topic: '{topic}'")

    try: # <<< START OF MAIN TRY BLOCK >>>

        # === Step 1: Generate Research Plan (with Rate Limit Retry) ===
        yield {'type': 'progress', 'message': f"Generating research plan for: '{topic}'...", 'is_error': False, 'is_fatal': False}
        plan_system_prompt = """
        You are a research assistant. Your task is to create a structured research plan based on a given topic.
        Output the plan STRICTLY as a JSON list of objects within a ```json ... ``` markdown block.
        Each object must represent a distinct research step and contain:
        - "step": A string describing the research question or area for that step.
        - "keywords": A list of 2-3 relevant and specific search query strings for that step.
        Aim for 5-7 logical steps suitable for generating a concise research report.
        Focus on breaking down the topic into manageable parts.
        Do not include any explanatory text before or after the JSON block.
        """
        plan_user_prompt = f"""
        Generate the research plan for the topic: "{topic}"

        Example for "Impact of AI on Journalism":
        ```json
        [
          {{"step": "Define AI in the context of journalism", "keywords": ["AI journalism definition", "AI tools for newsrooms"]}},
          {{"step": "Analyze AI's impact on news gathering", "keywords": ["AI automated news writing", "AI investigative journalism tools"]}},
          {{"step": "Examine ethical concerns of AI in journalism", "keywords": ["AI journalism ethics", "bias in AI news algorithms", "deepfakes journalism impact"]}},
          {{"step": "Explore AI's effect on journalist roles", "keywords": ["AI journalism job displacement", "AI skills for journalists"]}},
          {{"step": "Investigate AI's influence on news consumption", "keywords": ["AI news personalization", "filter bubbles AI news", "AI impact on media trust"]}},
          {{"step": "Summarize future trends of AI in journalism", "keywords": ["future of AI journalism", "AI journalism predictions"]}}
        ]
        ```
        Now, generate the plan for: "{topic}"
        """

        plan_response = None
        rate_limit_retries = 0
        while rate_limit_retries <= config.LLM_MAX_RATE_LIMIT_RETRIES:
            try:
                plan_response = call_gemini(prompt=plan_user_prompt, system_prompt=plan_system_prompt)
                # If successful, break the rate limit retry loop
                break
            except google_api_exceptions.ResourceExhausted as e:
                rate_limit_retries += 1
                logger.warning(f"LLM Rate Limit Hit (Plan Generation - Attempt {rate_limit_retries}/{config.LLM_MAX_RATE_LIMIT_RETRIES}): {e}")
                if rate_limit_retries > config.LLM_MAX_RATE_LIMIT_RETRIES:
                    error_msg = f"Fatal Error: Exceeded max rate limit retries ({config.LLM_MAX_RATE_LIMIT_RETRIES}) while generating research plan. Last error: {e}"
                    logger.error(error_msg)
                    yield {'type': 'progress', 'message': error_msg, 'is_error': True, 'is_fatal': True}
                    fatal_error_occurred = True
                    return # Stop generation

                # Yield status update to frontend
                delay = config.LLM_RATE_LIMIT_RETRY_DELAY * (rate_limit_retries) # Increase delay
                wait_message = f"LLM rate limit hit generating plan. Waiting {delay:.0f}s before retrying (Attempt {rate_limit_retries}/{config.LLM_MAX_RATE_LIMIT_RETRIES})..."
                # Yield as a progress message, easily handled by frontend
                yield {'type': 'progress', 'message': wait_message, 'is_error': True, 'is_fatal': False} # Error but not fatal yet
                time.sleep(delay)
                # Continue loop to retry call_gemini

            except (ValueError, RuntimeError, Exception) as e:
                 # Handle non-rate-limit errors immediately
                 logger.error(f"LLM Error generating research plan: {e}", exc_info=True)
                 error_msg = f"Fatal Error: An issue occurred while generating the research plan ({type(e).__name__}). Please check LLM configuration and logs."
                 yield {'type': 'progress', 'message': error_msg, 'is_error': True, 'is_fatal': True}
                 fatal_error_occurred = True
                 return # Stop generation

        # --- Parse the plan (assuming call_gemini succeeded or loop exited) ---
        if plan_response is None and not fatal_error_occurred: # Should only happen if rate limit retries exhausted
             logger.error("Plan generation failed after exhausting rate limit retries.")
             yield {'type': 'progress', 'message': "Fatal Error: Failed to generate research plan due to persistent LLM rate limits.", 'is_error': True, 'is_fatal': True}
             fatal_error_occurred = True
             return

        try:
            research_plan = parse_research_plan(plan_response)
            if not research_plan or not isinstance(research_plan, list) or not research_plan[0].get("step") or research_plan[0]["step"].startswith("Failed"):
                 fail_reason = research_plan[0]["step"] if (research_plan and isinstance(research_plan, list) and research_plan[0].get("step")) else "Could not parse plan structure from LLM."
                 raw_snippet = f" Raw Response Snippet: '{plan_response[:150].strip()}...'" if plan_response else " (LLM Response was empty)"
                 logger.error(f"Failed to create/parse research plan. Reason: {fail_reason}.{raw_snippet}")
                 yield {'type': 'progress', 'message': f"Fatal Error: Failed to create or parse research plan. Please try rephrasing the topic or check LLM status. Reason: {escape(fail_reason)}", 'is_error': True, 'is_fatal': True}
                 fatal_error_occurred = True
                 return
        except Exception as parse_err: # Catch potential errors in parse_research_plan itself
             logger.error(f"Error parsing plan response: {parse_err}", exc_info=True)
             yield {'type': 'progress', 'message': f"Fatal Error: Internal error parsing research plan structure: {escape(str(parse_err))}", 'is_error': True, 'is_fatal': True}
             fatal_error_occurred = True
             return

        yield {'type': 'progress', 'message': f"Generated {len(research_plan)} step plan.", 'is_error': False, 'is_fatal': False}
        for i, step in enumerate(research_plan):
             step_desc = step.get('step', 'Unnamed Step')
             step_keywords = step.get('keywords', [])
             yield {'type': 'progress', 'message': f"  Step {i+1}: {step_desc[:100]}{'...' if len(step_desc)>100 else ''} (Keywords: {step_keywords})", 'is_error': False, 'is_fatal': False}


        # === Step 2a: Search and Collect URLs (Track Step Index) ===
        yield {'type': 'progress', 'message': "Starting web search...", 'is_error': False, 'is_fatal': False}
        start_search_time = time.time()
        # Store tuples of (url, step_index)
        urls_with_step_index: List[Tuple[str, int]] = []
        all_unique_urls_found: Set[str] = set() # Track uniqueness across all steps
        total_search_errors = 0
        total_search_queries_attempted = 0

        for i, step in enumerate(research_plan):
            step_index = i
            step_desc = step.get('step', f'Unnamed Step {step_index+1}')
            keywords = step.get('keywords', [])
            progress_msg = f"Searching - Step {step_index+1}/{len(research_plan)}: '{step_desc[:70]}{'...' if len(step_desc)>70 else ''}'"
            yield {'type': 'progress', 'message': progress_msg, 'is_error': False, 'is_fatal': False}

            if not keywords or not isinstance(keywords, list) or not any(kw.strip() for kw in keywords):
                yield {'type': 'progress', 'message': "  -> No valid keywords provided for this step, skipping search.", 'is_error': False, 'is_fatal': False}
                continue

            valid_keywords = [kw for kw in keywords if isinstance(kw, str) and kw.strip()]
            if not valid_keywords:
                 yield {'type': 'progress', 'message': "  -> Keywords list contained only invalid entries, skipping search.", 'is_error': False, 'is_fatal': False}
                 continue

            total_search_queries_attempted += 1
            step_urls, step_errors = perform_web_search(valid_keywords)

            if step_errors:
                total_search_errors += len(step_errors)
                for err in step_errors:
                    yield {'type': 'progress', 'message': f"    -> Search Warning: {escape(err[:200])}", 'is_error': True, 'is_fatal': False}

            new_urls_added_count = 0
            for url in step_urls:
                 if url not in all_unique_urls_found:
                      urls_with_step_index.append((url, step_index))
                      all_unique_urls_found.add(url)
                      new_urls_added_count += 1

            yield {'type': 'progress', 'message': f"  -> Found {len(step_urls)} URLs ({new_urls_added_count} new unique added for this step). Total unique: {len(all_unique_urls_found)}.", 'is_error': False, 'is_fatal': False}

            if i < len(research_plan) - 1:
                 time.sleep(config.INTER_SEARCH_DELAY_SECONDS)

        search_duration = time.time() - start_search_time
        yield {'type': 'progress', 'message': f"Search phase completed in {search_duration:.2f}s.", 'is_error': False, 'is_fatal': False}
        if total_search_errors > 0:
            yield {'type': 'progress', 'message': f"Collected {len(all_unique_urls_found)} total unique URLs from {total_search_queries_attempted} queries ({total_search_errors} search errors occurred).", 'is_error': True, 'is_fatal': False}
        else:
            yield {'type': 'progress', 'message': f"Collected {len(all_unique_urls_found)} total unique URLs from {total_search_queries_attempted} queries.", 'is_error': False, 'is_fatal': False}

        # --- Filter URLs (Now working with (url, step_index) tuples) ---
        urls_to_scrape_with_step: List[Tuple[str, int]] = []
        skipped_urls_count = 0
        yield {'type': 'progress', 'message': "Filtering URLs for scraping...", 'is_error': False, 'is_fatal': False}
        common_non_content_domains = {
            'youtube.com', 'youtu.be', 'vimeo.com', # Video platforms
            'facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com', 'pinterest.com', 'reddit.com', # Social media
            'amazon.', 'ebay.', # E-commerce product pages often not useful
            'google.com/search', 'bing.com', 'duckduckgo.com', # Search results themselves
            'archive.org', # Web archives (can be noisy, consider allowing if needed)
            'wikipedia.org', # Often good, but can dominate results - maybe filter later if needed?
            'login', 'signin', 'register', 'account', # Auth pages
            'tel:', 'mailto:', 'javascript:', 'ftp:', 'file:', 'data:', # Unwanted schemes
            'localhost', '127.0.0.1' # Local addresses
        }
        common_file_extensions = ('.pdf', '.jpg', '.png', '.gif', '.zip', '.mp4', '.mp3', '.docx', '.xlsx', '.pptx', '.webp', '.svg', '.xml', '.css', '.js', '.jpeg', '.doc', '.xls', '.ppt', '.txt', '.exe', '.dmg', '.iso', '.rar', '.gz', '.tar', '.bz2', '.7z', '.json', '.csv', '.woff', '.woff2', '.ttf', '.eot', '.map')

        # Use the combined list from the search phase
        sorted_urls_with_step = sorted(urls_with_step_index, key=lambda item: item[1]) # Sort primarily by step index

        for url, step_index in sorted_urls_with_step:
             if len(urls_to_scrape_with_step) >= config.MAX_TOTAL_URLS_TO_SCRAPE:
                  yield {'type': 'progress', 'message': f"  -> Reached URL scraping limit ({config.MAX_TOTAL_URLS_TO_SCRAPE}). Skipping remaining URLs.", 'is_error': False, 'is_fatal': False}
                  skipped_urls_count += (len(sorted_urls_with_step) - len(urls_to_scrape_with_step)) # Count remaining as skipped
                  break
             try:
                 lower_url = url.lower()
                 parsed_url = urlparse(url)
                 domain = parsed_url.netloc
                 is_unwanted_scheme = not lower_url.startswith(('http://', 'https://'))
                 path_part = parsed_url.path.lower()
                 is_file_extension = path_part.endswith(common_file_extensions)
                 is_common_non_content = any(skip_domain in domain for skip_domain in common_non_content_domains if skip_domain.endswith('.')) or \
                                          any(domain == skip_domain for skip_domain in common_non_content_domains if not skip_domain.endswith('.')) or \
                                          any(auth_path in lower_url for auth_path in ['login', 'signin', 'register', 'account'])
                 is_local = domain in ['localhost', '127.0.0.1']
                 if not any([is_unwanted_scheme, is_file_extension, is_common_non_content, is_local]):
                      urls_to_scrape_with_step.append((url, step_index))
                 else:
                      skipped_urls_count += 1
                      reason = "unwanted scheme" if is_unwanted_scheme else \
                               "file extension" if is_file_extension else \
                               "common non-content domain/path" if is_common_non_content else \
                               "local address" if is_local else "unknown filter"
                      logger.debug(f"Filtering: Skipped URL '{url[:70]}...' (Step {step_index+1}) Reason: {reason}")
             except Exception as filter_err:
                 logger.warning(f"Error during URL filtering for '{url[:70]}...' (Step {step_index+1}): {filter_err}", exc_info=False)
                 skipped_urls_count += 1

        yield {'type': 'progress', 'message': f"Selected {len(urls_to_scrape_with_step)} URLs for scraping after filtering ({skipped_urls_count} skipped, limit was {config.MAX_TOTAL_URLS_TO_SCRAPE}).", 'is_error': False, 'is_fatal': False}

        if not urls_to_scrape_with_step:
             logger.error("No suitable URLs found to scrape after search and filtering.")
             yield {'type': 'progress', 'message': "Fatal Error: No suitable URLs found to scrape. Cannot proceed. Try broadening the topic or checking search engine status.", 'is_error': True, 'is_fatal': True}
             fatal_error_occurred = True
             return # Stop generation


        # === Step 2b: Scrape URLs Concurrently (Track Step Index) ===
        yield {'type': 'progress', 'message': f"Starting concurrent scraping ({config.MAX_WORKERS} workers)...", 'is_error': False, 'is_fatal': False}
        start_scrape_time = time.time()
        grouped_scraped_metadata: Dict[int, List[Dict[str, Any]]] = defaultdict(list) # Group by step index
        all_successful_scrapes = [] # Reset flat list
        processed_scrape_count = 0
        successful_scrape_count = 0
        scrape_errors = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS, thread_name_prefix="Scraper") as executor:
            # Map future back to (url, step_index)
            future_to_info = {executor.submit(scrape_url, url): (url, step_idx) for url, step_idx in urls_to_scrape_with_step}

            for future in concurrent.futures.as_completed(future_to_info):
                url, step_index = future_to_info[future]
                log_url_snippet = f"{url[:60]}... (Step {step_index+1})"
                processed_scrape_count += 1
                try:
                    result_dict = future.result()
                    if result_dict and 'temp_filepath' in result_dict and result_dict['temp_filepath']:
                        if os.path.exists(result_dict['temp_filepath']):
                            # Add step_index to the result metadata before yielding/storing
                            result_dict['step_index'] = step_index
                            grouped_scraped_metadata[step_index].append(result_dict)
                            all_successful_scrapes.append(result_dict) # Add to flat list as well
                            successful_scrape_count += 1
                            yield {'type': 'scrape_success', 'metadata': result_dict} # Yield includes step_index now
                        else:
                            logger.warning(f"Scrape task for {log_url_snippet} reported success but temp file missing: {result_dict['temp_filepath']}")
                            scrape_errors.append(f"Temp file missing for {log_url_snippet}")
                except Exception as exc:
                    logger.error(f"Unexpected error in scraping task future for {log_url_snippet}: {exc}", exc_info=True)
                    err_msg = f"Scrape task failed for {log_url_snippet}: {type(exc).__name__}"
                    scrape_errors.append(err_msg)
                    yield {'type': 'progress', 'message': f"    -> {err_msg}", 'is_error': True, 'is_fatal': False}

                if processed_scrape_count % 5 == 0 or processed_scrape_count == len(urls_to_scrape_with_step):
                      progress_perc = (processed_scrape_count * 100) // len(urls_to_scrape_with_step)
                      yield {'type': 'progress', 'message': f"  -> Scraping Progress: {processed_scrape_count}/{len(urls_to_scrape_with_step)} URLs processed ({progress_perc}%). Successful: {successful_scrape_count}", 'is_error': False, 'is_fatal': False}

        scrape_duration = time.time() - start_scrape_time
        yield {'type': 'progress', 'message': f"Scraping finished in {scrape_duration:.2f}s. Successfully scraped and sanitized content from {successful_scrape_count} URLs.", 'is_error': False, 'is_fatal': False}
        if scrape_errors:
             yield {'type': 'progress', 'message': f"  -> Encountered {len(scrape_errors)} errors during scraping.", 'is_error': True, 'is_fatal': False}

        if not all_successful_scrapes: # Check if the flat list is empty
            logger.error("Failed to scrape any usable content successfully after sanitization and filtering.")
            yield {'type': 'progress', 'message': "Fatal Error: Failed to gather sufficient web content after scraping and filtering. Cannot proceed with synthesis.", 'is_error': True, 'is_fatal': True}
            fatal_error_occurred = True
            return # Stop generation


        # === Step 3: Generate Bibliography Map ===
        # Use the flat list of all successful scrapes for bibliography
        url_to_index_map, bibliography_prompt_list = generate_bibliography_map(all_successful_scrapes)
        yield {'type': 'progress', 'message': f"Generated bibliography map for {len(url_to_index_map)} successfully processed sources.", 'is_error': False, 'is_fatal': False}


        # === Step 4: Synthesize Information (Streaming, RAM Optimized, Step-Balanced Context) ===
        yield {'type': 'progress', 'message': f"Synthesizing information from {successful_scrape_count} sources using {config.GOOGLE_MODEL_NAME}...", 'is_error': False, 'is_fatal': False}
        yield {'type': 'event', 'data': {'type': 'stream_start', 'target': 'synthesis'}}

        # --- Prepare Context (NEW: Round-Robin across steps) ---
        context_for_llm_parts = []
        current_total_chars = 0
        sources_included_count = 0
        skipped_sources_count = 0
        context_build_start_time = time.time()

        yield {'type': 'progress', 'message': f"  -> Preparing context for synthesis (limit ~{config.MAX_CONTEXT_CHARS // 1000}k chars, balancing across {len(grouped_scraped_metadata)} steps)...", 'is_error': False, 'is_fatal': False}

        # Determine the order of steps (e.g., 0, 1, 2...)
        step_indices_with_sources = sorted(grouped_scraped_metadata.keys())
        # Track the next source index to pick from each step's list
        next_source_index_per_step = {step_idx: 0 for step_idx in step_indices_with_sources}
        # Track how many sources remain for each step
        remaining_sources_per_step = {step_idx: len(grouped_scraped_metadata[step_idx]) for step_idx in step_indices_with_sources}
        total_sources_available = sum(remaining_sources_per_step.values())
        context_limit_reached = False

        # Loop while context limit not reached AND there are sources left to consider
        while not context_limit_reached and sum(remaining_sources_per_step.values()) > 0:
            made_progress_this_round = False # Track if we added anything in a full pass
            for step_idx in step_indices_with_sources:
                if context_limit_reached: break # Stop immediately if limit hit mid-round

                current_source_idx = next_source_index_per_step[step_idx]
                if current_source_idx < len(grouped_scraped_metadata[step_idx]):
                    # There's a source available for this step
                    source_metadata = grouped_scraped_metadata[step_idx][current_source_idx]
                    filepath = source_metadata.get('temp_filepath')
                    url = source_metadata.get('url')
                    log_url_snippet = f"{url[:60]}... (Step {step_idx+1})"

                    if not filepath or not url or not os.path.exists(filepath):
                        logger.warning(f"Skipping source for synthesis (missing temp file/metadata). URL: {log_url_snippet}, Path: {filepath}")
                        # Don't yield progress here to avoid flooding, logged above
                        skipped_sources_count += 1
                        remaining_sources_per_step[step_idx] -= 1 # Decrement remaining for this step
                        next_source_index_per_step[step_idx] += 1 # Move to next source index for this step
                        continue

                    try:
                        file_size = os.path.getsize(filepath)
                        if file_size == 0:
                            logger.warning(f"Skipping empty temp file {os.path.basename(filepath)} for {log_url_snippet}")
                            skipped_sources_count += 1
                            remaining_sources_per_step[step_idx] -= 1
                            next_source_index_per_step[step_idx] += 1
                            continue

                        # Estimate characters: URL len + file content size + small overhead
                        # Note: File size is bytes, roughly equivalent to chars for ASCII/UTF-8 common chars
                        estimated_addition = len(url) + file_size + 50

                        if (current_total_chars + estimated_addition) <= config.MAX_CONTEXT_CHARS:
                            # Read content (only if it fits)
                            content = None
                            try:
                                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                if content and content.strip():
                                    context_for_llm_parts.append({'url': url, 'content': content})
                                    current_total_chars += estimated_addition # Use estimate for consistency
                                    sources_included_count += 1
                                    next_source_index_per_step[step_idx] += 1 # Successfully added, move to next source
                                    remaining_sources_per_step[step_idx] -= 1
                                    made_progress_this_round = True
                                    logger.debug(f"Added source {log_url_snippet} to context. Total chars ~{current_total_chars // 1000}k.")
                                else:
                                    logger.warning(f"Read empty or whitespace-only content from temp file {os.path.basename(filepath)} for {log_url_snippet} (Post-Sanitization Check)")
                                    skipped_sources_count += 1
                                    remaining_sources_per_step[step_idx] -= 1
                                    next_source_index_per_step[step_idx] += 1 # Skip this source
                            except Exception as read_err:
                                logger.error(f"Error reading temp file {os.path.basename(filepath)} for {log_url_snippet}: {read_err}", exc_info=False)
                                skipped_sources_count += 1
                                remaining_sources_per_step[step_idx] -= 1
                                next_source_index_per_step[step_idx] += 1 # Skip this source
                            finally:
                                del content # Free memory from content string
                        else:
                            # This source doesn't fit, context limit is effectively reached
                            logger.warning(f"Context limit ({config.MAX_CONTEXT_CHARS // 1000}k chars) reached. Could not add source {log_url_snippet} (needs ~{estimated_addition // 1000}k chars, have ~{(config.MAX_CONTEXT_CHARS - current_total_chars) // 1000}k left).")
                            context_limit_reached = True
                            # Don't increment next_source_index_per_step - we didn't process this one
                            # Don't decrement remaining_sources_per_step - it's still available if limit increases
                            break # Break inner loop (steps)

                    except OSError as e:
                        logger.error(f"Error accessing temp file metadata {os.path.basename(filepath)} for {log_url_snippet}: {e}", exc_info=False)
                        skipped_sources_count += 1
                        remaining_sources_per_step[step_idx] -= 1
                        next_source_index_per_step[step_idx] += 1
                    except Exception as e:
                        logger.error(f"Unexpected error processing temp file {os.path.basename(filepath)} for {log_url_snippet}: {e}", exc_info=False)
                        skipped_sources_count += 1
                        remaining_sources_per_step[step_idx] -= 1
                        next_source_index_per_step[step_idx] += 1
                # else: No more sources left for this step_idx

            # End of round-robin pass through steps
            if not made_progress_this_round and not context_limit_reached:
                # If we went through all steps and added nothing, and the limit isn't reached,
                # it means all remaining sources were individually too large to fit.
                logger.warning("Context preparation finished: No remaining sources could fit within the context limit.")
                break # Exit the while loop

        context_build_duration = time.time() - context_build_start_time
        logger.info(f"Context preparation took {context_build_duration:.2f}s.")

        skipped_sources_count += sum(remaining_sources_per_step.values()) # Add remaining unprocessed sources to skipped count

        if sources_included_count == 0:
             logger.error("No source content could be prepared for synthesis after reading temp files.")
             yield {'type': 'progress', 'message': "Fatal Error: No source content available for synthesis. Cannot proceed.", 'is_error': True, 'is_fatal': True}
             fatal_error_occurred = True
             return # Stop generation

        if skipped_sources_count > 0:
             yield {'type': 'progress', 'message': f"  -> Context includes {sources_included_count} sources. Skipped {skipped_sources_count}/{total_sources_available} sources due to errors, size limits, or empty content.", 'is_error': True, 'is_fatal': False}
        estimated_tokens = current_total_chars / config.CHARS_PER_TOKEN_ESTIMATE
        yield {'type': 'progress', 'message': f"  -> Prepared synthesis context using {sources_included_count} sources (~{current_total_chars // 1000}k chars / ~{estimated_tokens / 1000:.1f}k est. tokens).", 'is_error': False, 'is_fatal': False}

        try:
            context_json_str = json.dumps(context_for_llm_parts, indent=None, ensure_ascii=False, separators=(',', ':'))
            del context_for_llm_parts # Free memory
            actual_context_chars = len(context_json_str)
            logger.info(f"Actual serialized context size: {actual_context_chars // 1000}k chars.")
            if actual_context_chars > config.MAX_CONTEXT_CHARS * 1.1:
                 logger.warning(f"Serialized context ({actual_context_chars // 1000}k) significantly exceeds estimated limit ({config.MAX_CONTEXT_CHARS // 1000}k). LLM call might fail.")
                 yield {'type': 'progress', 'message': f"  -> Warning: Prepared context size ({actual_context_chars // 1000}k chars) exceeds target limit. LLM call might fail due to size.", 'is_error': True, 'is_fatal': False}
        except (TypeError, OverflowError, Exception) as json_err:
            logger.error(f"Failed to serialize context parts to JSON: {json_err}", exc_info=True)
            yield {'type': 'progress', 'message': f"Fatal Error: Could not prepare context data for LLM due to serialization issue: {escape(str(json_err))}", 'is_error': True, 'is_fatal': True}
            fatal_error_occurred = True
            return

        # --- Synthesis Prompt ---
        synthesis_system_prompt = f"""
        You are a research synthesizer. Your goal is to extract and organize information relevant to a specific research plan, based *only* on the provided web content snippets.
        You must cite every piece of information accurately using the specified format.
        Output the synthesis in Markdown format, structured strictly by the research plan steps.
        """
        synthesis_user_prompt = f"""
        Analyze the following web content snippets related to the topic "{escape(topic)}".
        Synthesize the key information relevant to each step of the research plan below.

        Research Topic: {escape(topic)}

        Research Plan:
        ```json
        {json.dumps(research_plan, indent=2)}
        ```

        Source Content Snippets ({sources_included_count} sources):
        ```json
        {context_json_str}
        ```

        **Instructions:**
        1.  Go through each step in the "Research Plan" sequentially.
        2.  For each step, find relevant information within the "Source Content Snippets". Snippets are provided as a JSON list, where each item has a 'url' and 'content' (the sanitized text from that URL).
        3.  Present the findings for each step under a Markdown heading: `### Step X: <Step Description from Plan>`.
        4.  Synthesize the information concisely. Use bullet points or short paragraphs.
        5.  **CRITICAL CITATION REQUIREMENT:** Immediately after *every* sentence or piece of information derived from a specific source, you MUST cite it using the exact format: `[Source URL: <full_url_here>]`. Use the full URL provided in the 'url' field of the corresponding source snippet.
        6.  If multiple sources support the same point, list citations consecutively: `[Source URL: <url1>][Source URL: <url2>]`. Do NOT combine information without citing each source used.
        7.  If no relevant information is found for a specific plan step within the provided snippets, state clearly under the step heading: "No specific information found for this step in the provided sources."
        8.  Use `---` as a separator ONLY between the synthesized content for different plan steps.
        9.  **Output ONLY the synthesized Markdown content, starting directly with the first step's heading (`### Step 1:...`).** Do NOT include an introduction, conclusion, summary, bibliography, or any other text outside the step-by-step synthesis.

        Begin the Markdown synthesis now.
        """
        del context_json_str # Free memory

        # --- Execute Synthesis Stream (Now handles rate limit retries internally) ---
        accumulated_synthesis_md = ""
        synthesis_stream_error = None
        synthesis_stream_fatal = False
        try:
            stream_generator = stream_gemini(prompt=synthesis_user_prompt, system_prompt=synthesis_system_prompt)
            for result in stream_generator:
                event_type = result.get('type')
                if event_type == 'chunk':
                    content = result.get('content', '')
                    yield {'type': 'event', 'data': {'type': 'llm_chunk', 'content': content, 'target': 'synthesis'}}
                    accumulated_synthesis_md += content
                elif event_type == 'llm_status': # Handle new status event
                    yield {'type': 'progress', 'message': result.get('message', 'LLM status update.'), 'is_error': True, 'is_fatal': False} # Show as non-fatal error/warning
                elif event_type == 'stream_error':
                    synthesis_stream_error = result.get('message', 'Unknown stream error')
                    synthesis_stream_fatal = result.get('is_fatal', False)
                    logger.error(f"LLM stream error during synthesis: {synthesis_stream_error}")
                    yield {'type': 'progress', 'message': f"LLM Error (Synthesis): {escape(synthesis_stream_error)}", 'is_error': True, 'is_fatal': synthesis_stream_fatal}
                    if synthesis_stream_fatal:
                        fatal_error_occurred = True
                        return # Stop generation
                    # Don't break here if not fatal, allow process to continue to report phase maybe?
                elif event_type == 'stream_warning':
                     msg = result.get('message', 'Unknown stream warning')
                     logger.warning(f"LLM Stream Warning (Synthesis): {msg}")
                     yield {'type': 'progress', 'message': f"LLM Warning (Synthesis): {escape(msg)}", 'is_error': True, 'is_fatal': False}
                elif event_type == 'stream_end':
                     reason = result.get('finish_reason', 'N/A')
                     logger.info(f"LLM synthesis stream finished. Finish Reason: {reason}")
                     if reason not in ["STOP", "MAX_TOKENS", "1", "2", None, 0, "FINISH_REASON_UNSPECIFIED"]: # Added 0/UNSPECIFIED
                         logger.warning(f"Synthesis stream finished with non-standard reason: {reason}")
                         yield {'type': 'progress', 'message': f"Warning: Synthesis stream finished with reason: {reason}. Output might be incomplete.", 'is_error': True, 'is_fatal': False}
                     break # Exit loop on stream end
        except Exception as e:
             logger.error(f"Fatal error processing LLM synthesis stream: {e}", exc_info=True)
             yield {'type': 'progress', 'message': f"Fatal Error: Unexpected issue during information synthesis ({type(e).__name__}).", 'is_error': True, 'is_fatal': True}
             fatal_error_occurred = True
             return

        yield {'type': 'progress', 'message': "Synthesis generation finished." + (" (Errors occurred)" if synthesis_stream_error else ""), 'is_error': bool(synthesis_stream_error), 'is_fatal': False}

        if not accumulated_synthesis_md.strip():
             if synthesis_stream_error:
                 logger.error(f"Synthesis resulted in empty content, likely due to stream error: {synthesis_stream_error}")
                 yield {'type': 'progress', 'message': "Warning: Synthesis was empty, potentially due to an earlier LLM error.", 'is_error': True, 'is_fatal': False}
             else:
                 logger.warning("Synthesis resulted in empty content, but no stream error reported. LLM might have found no relevant info.")
                 yield {'type': 'progress', 'message': "Note: Synthesis phase produced no content. This might be expected if sources lacked relevant information.", 'is_error': False, 'is_fatal': False}


        # === Step 5: Generate Final Report (Streaming) ===
        yield {'type': 'progress', 'message': f"Generating final report using {config.GOOGLE_MODEL_NAME}...", 'is_error': False, 'is_fatal': False}
        yield {'type': 'event', 'data': {'type': 'stream_start', 'target': 'report'}}

        # --- Prepare Report Prompt (Check context size for synthesis part) ---
        truncated_synthesis_md = ""
        try:
            # Estimate size of fixed parts of the prompt
            plan_json_compact = json.dumps(research_plan, separators=(',', ':'))
            # Estimate base length: topic, plan, bibliography string, fixed instruction text (~3k)
            base_prompt_elements_len = (
                len(topic) + len(plan_json_compact) + len(bibliography_prompt_list) + 3000
            )

            # Calculate space available for the synthesized markdown
            available_chars_for_synthesis_in_report = config.MAX_CONTEXT_CHARS - base_prompt_elements_len

            if len(accumulated_synthesis_md) > available_chars_for_synthesis_in_report:
                # Ensure chars_to_keep is not negative
                chars_to_keep = max(0, available_chars_for_synthesis_in_report)
                truncated_synthesis_md = accumulated_synthesis_md[:chars_to_keep] + "\n\n... [Synthesis truncated due to context limits for final report generation]"
                logger.warning(f"Truncating synthesis markdown (from {len(accumulated_synthesis_md)} to {len(truncated_synthesis_md)} chars) for report prompt.")
                yield {'type': 'progress', 'message': "  -> Warning: Synthesized text truncated for final report generation due to context limits. Report might be incomplete.", 'is_error': True, 'is_fatal': False}
            else:
                truncated_synthesis_md = accumulated_synthesis_md
        except Exception as e:
             logger.error(f"Error preparing report prompt (truncation check): {e}", exc_info=True)
             yield {'type': 'progress', 'message': f"Error preparing report prompt: {escape(str(e))}", 'is_error': True, 'is_fatal': False}
             # Fallback: Use potentially untruncated synthesis if calculation failed
             truncated_synthesis_md = accumulated_synthesis_md


        # --- Report Prompt ---
        report_system_prompt = f"""
        You are a research report writer. Your task is to compile a comprehensive research report in Markdown format based on the provided research plan, synthesized information, and bibliography map.
        Pay close attention to formatting instructions and accurately replace URL citations with footnote markers.
        The final output must be only the Markdown report content.
        """
        report_user_prompt = f"""
        Create a comprehensive research report in Markdown format on the topic: "{escape(topic)}".

        Use the following inputs:

        1.  **Original Research Plan:**
            ```json
            {json.dumps(research_plan, indent=2)}
            ```

        2.  **Synthesized Information (Markdown with raw URL citations):**
            ```markdown
            {truncated_synthesis_md if truncated_synthesis_md.strip() else "No synthesized information was available or provided."}
            ```

        3.  **Bibliography Map (URL to Citation Number):**
            ```
            {bibliography_prompt_list if bibliography_prompt_list else "No sources available for bibliography."}
            ```

        **Instructions for Generating the Final Markdown Report:**

        1.  **Structure:** Create the report with these Markdown sections:
            *   `# Research Report: {escape(topic)}` (Main Title)
            *   `## Introduction`
            *   `## Findings` (Contains subsections for each plan step)
            *   `## Conclusion`
            *   `## Bibliography`

        2.  **Introduction:** Briefly introduce the research topic "{escape(topic)}". State the report's purpose (to synthesize information based on the plan and sources). Optionally list the key steps from the Research Plan. Keep concise (2-4 sentences).

        3.  **Findings:**
            *   Under `## Findings`, create a subsection for each step in the original Research Plan using `### Step X: <Step Description>`. Use the exact step description from the plan.
            *   Integrate the relevant information for each step from the "Synthesized Information" section. Rephrase and structure for clarity and flow.
            *   **Citation Replacement (Crucial):** Find every raw URL citation `[Source URL: <full_url_here>]` in the Synthesized Information you use. Replace **each complete instance** with the corresponding Markdown footnote reference `[^N]`, where `N` is the number associated with `<full_url_here>` in the provided "Bibliography Map".
            *   **Handling Missing Citations:** If a URL inside `[Source URL: ...]` is NOT found as a key in the Bibliography Map (this indicates an error upstream, but handle gracefully), **OMIT** the citation marker entirely for that instance. Do not invent numbers or leave partial tags (e.g., remove `[Source URL: http://example.com/not/in/map]`). Log a warning internally if possible (though you are an LLM).
            *   If "Synthesized Information" was empty or lacked content for a step, state this clearly under the relevant `### Step X:` heading (e.g., "No specific findings were synthesized for this step based on the available sources.").

        4.  **Conclusion:** Summarize the key findings (or lack thereof). Briefly mention limitations encountered (e.g., limited sources, synthesis truncation, information gaps noted in Findings). Keep concise (2-4 sentences). Do not introduce new information.

        5.  **Bibliography:**
            *   Under `## Bibliography`, list all sources from the "Bibliography Map".
            *   Use the standard Markdown footnote definition format: `[^N]: <full_url_here>`
            *   Each definition **must** be on its own line. Ensure numbers `N` match those used in Findings.
            *   If the Bibliography Map was empty, state "No sources were cited in this report." under the heading.

        6.  **Formatting:** Use standard Markdown (headings, lists, bold, italics). Ensure proper spacing. Use `*` or `-` for bullet points. Use `---` only if naturally needed for separation, not mandatorily between sections.

        7.  **Output:** Generate **ONLY** the complete Markdown report. Start directly with `# Research Report: {escape(topic)}`. Do not include preamble, explanations, or text outside the defined report structure.

        Generate the final Markdown report now.
        """
        del accumulated_synthesis_md
        del truncated_synthesis_md
        # Keep research_plan and bibliography_prompt_list if needed for debugging, otherwise del
        # del research_plan
        # del bibliography_prompt_list

        # --- Execute Report Stream (Now handles rate limit retries internally) ---
        final_report_markdown = ""
        report_stream_error = None
        report_stream_fatal = False
        try:
            stream_generator = stream_gemini(prompt=report_user_prompt, system_prompt=report_system_prompt)
            for result in stream_generator:
                event_type = result.get('type')
                if event_type == 'chunk':
                    content = result.get('content', '')
                    yield {'type': 'event', 'data': {'type': 'llm_chunk', 'content': content, 'target': 'report'}}
                    final_report_markdown += content
                elif event_type == 'llm_status': # Handle new status event
                     yield {'type': 'progress', 'message': result.get('message', 'LLM status update.'), 'is_error': True, 'is_fatal': False} # Show as non-fatal error/warning
                elif event_type == 'stream_error':
                    report_stream_error = result.get('message', 'Unknown stream error')
                    report_stream_fatal = result.get('is_fatal', False)
                    logger.error(f"LLM stream error during report generation: {report_stream_error}")
                    yield {'type': 'progress', 'message': f"LLM Error (Report): {escape(report_stream_error)}", 'is_error': True, 'is_fatal': report_stream_fatal}
                    if report_stream_fatal:
                        fatal_error_occurred = True
                        return # Stop generation
                    # Don't break if not fatal
                elif event_type == 'stream_warning':
                     msg = result.get('message', 'Unknown stream warning')
                     logger.warning(f"LLM Stream Warning (Report): {msg}")
                     yield {'type': 'progress', 'message': f"LLM Warning (Report): {escape(msg)}", 'is_error': True, 'is_fatal': False}
                elif event_type == 'stream_end':
                     reason = result.get('finish_reason', 'N/A')
                     logger.info(f"LLM report stream finished. Finish Reason: {reason}")
                     if reason not in ["STOP", "MAX_TOKENS", "1", "2", None, 0, "FINISH_REASON_UNSPECIFIED"]: # Added 0/UNSPECIFIED
                         logger.warning(f"Report stream finished with non-standard reason: {reason}")
                         yield {'type': 'progress', 'message': f"Warning: Report stream finished with reason: {reason}. Output might be incomplete.", 'is_error': True, 'is_fatal': False}
                     break # Exit loop on stream end
        except Exception as e:
             logger.error(f"Fatal error processing LLM report stream: {e}", exc_info=True)
             yield {'type': 'progress', 'message': f"Fatal Error: Unexpected issue during final report generation ({type(e).__name__}).", 'is_error': True, 'is_fatal': True}
             fatal_error_occurred = True
             return

        yield {'type': 'progress', 'message': "Report generation finished." + (" (Errors occurred)" if report_stream_error else ""), 'is_error': bool(report_stream_error), 'is_fatal': False}

        if not final_report_markdown.strip():
            if report_stream_error:
                 logger.error(f"Final report generation resulted in empty content due to stream error: {report_stream_error}")
                 final_report_markdown = f"# Research Report: {escape(topic)}\n\n## Error\n\nThe final report could not be generated due to an error during the process:\n\n```\n{escape(report_stream_error)}\n```"
                 yield {'type': 'progress', 'message': "Error: Final report generation failed due to an LLM error.", 'is_error': True, 'is_fatal': False}
            else:
                 logger.error("Final report generation resulted in empty content, but no stream error reported.")
                 final_report_markdown = f"# Research Report: {escape(topic)}\n\n## Error\n\nReport generation failed unexpectedly and produced no content. Please check the logs."
                 yield {'type': 'progress', 'message': "Error: Final report generation resulted in empty content.", 'is_error': True, 'is_fatal': False}


        # === Step 6: Final Processing and Completion ===
        yield {'type': 'progress', 'message': "Processing final report for display...", 'is_error': False, 'is_fatal': False}

        # --- Strip potential Markdown code fences ---
        cleaned_report_markdown = final_report_markdown.strip()
        if cleaned_report_markdown.startswith("```markdown") and cleaned_report_markdown.endswith("```"):
             cleaned_report_markdown = cleaned_report_markdown[len("```markdown"): -len("```")].strip()
             logger.info("Stripped surrounding ```markdown fences from the final report.")
        elif cleaned_report_markdown.startswith("```") and cleaned_report_markdown.endswith("```"):
             possible_content = cleaned_report_markdown[3:-3].strip()
             # Heuristic: If the content inside starts like a report, keep the fences, otherwise strip.
             if not possible_content.startswith(('#', '##')): # Check if it starts with markdown headings
                  logger.info("Stripped surrounding ``` fences from the final report (content didn't start with heading).")
                  cleaned_report_markdown = possible_content
             else:
                  logger.info("Detected ``` fences, but content looks like the report itself. Keeping fences.")
        final_report_markdown = cleaned_report_markdown

        # --- Convert Markdown to HTML ---
        report_html = convert_markdown_to_html(final_report_markdown)
        if report_html.strip().lower().startswith(('<p><em>report content is empty', '<pre><strong>error', '<p><em>markdown conversion resulted', 'report generation failed', '<h2>report display error</h2>')):
            logger.error("Final Markdown report content seems invalid or conversion failed.")
            yield {'type': 'progress', 'message': "Error preparing final report for display. Check logs.", 'is_error': True, 'is_fatal': False}
            # Ensure error HTML is used if conversion failed
            if not report_html.strip() or "conversion resulted in empty html" in report_html.lower():
                 report_html = f"<h2>Report Display Error</h2><p>Could not convert report Markdown to HTML. Raw Markdown content:</p><pre><code>{escape(final_report_markdown)}</code></pre>"
        elif not report_html.strip():
             logger.error("Markdown conversion resulted in empty HTML unexpectedly.")
             yield {'type': 'progress', 'message': "Error: Markdown conversion resulted in empty HTML.", 'is_error': True, 'is_fatal': False}
             report_html = f"<h2>Report Display Error</h2><p>Markdown conversion resulted in empty content. Raw Markdown content:</p><pre><code>{escape(final_report_markdown)}</code></pre>"


        # --- Yield Final Data Package ---
        yield {'type': 'progress', 'message': "Sending final results to client...", 'is_error': False, 'is_fatal': False}
        final_data = {
            'type': 'complete',
            'report_html': report_html
        }
        yield {'type': 'event', 'data': final_data}

        end_time_total = time.time()
        total_duration = end_time_total - start_time_total
        logger.info(f"Research process for '{topic}' completed in {total_duration:.2f} seconds.")
        yield {'type': 'progress', 'message': f"Research process completed successfully in {total_duration:.2f} seconds.", 'is_error': False, 'is_fatal': False}

    except Exception as e:
        # Catch-all for unexpected errors during orchestration
        logger.error(f"FATAL: Unhandled exception in research orchestrator for topic '{topic}': {e}", exc_info=True)
        error_msg = "An unexpected server error occurred during research orchestration. Please check server logs for details."
        if not fatal_error_occurred:
            try:
                 # Yield fatal error only if one wasn't already yielded
                 yield {'type': 'progress', 'message': error_msg, 'is_error': True, 'is_fatal': True}
            except Exception as callback_err:
                 # Log failure to yield, but avoid raising further errors
                 logger.critical(f"Failed to yield final fatal error message in orchestrator: {callback_err}")
        fatal_error_occurred = True # Ensure flag is set
    finally:
        # Cleanup (temp files) is handled by app.py using scrape_success events & tracking filepaths
        if fatal_error_occurred:
            logger.error(f"Orchestrator generator stopped prematurely due to fatal error for topic: '{topic}'.")
        else:
            logger.info(f"Orchestrator generator finished executing for topic: '{topic}'.")
        # Signal generator termination (useful for cleanup coordination if needed later)
        # The app.py stream handler now sends a 'stream_terminated' event in its finally block.