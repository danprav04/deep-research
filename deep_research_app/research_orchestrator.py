# research_orchestrator.py
import time
import json
import os
import traceback
import logging
import re
from typing import Dict, Any, List, Callable, Optional, Tuple, Generator
from html import escape
import concurrent.futures
from urllib.parse import urlparse

import config as config
from llm_interface import call_gemini, stream_gemini
from web_research import perform_web_search, scrape_url
from utils import (
    parse_research_plan, generate_bibliography_map, convert_markdown_to_html
)

# Configure logger for this module
logger = logging.getLogger(__name__)

# Define yielded event structures for clarity
ProgressEvent = Dict[str, Any] # keys: type='progress', message, is_error, is_fatal
DataEvent = Dict[str, Any]     # keys: type='event', data={...}
ScrapeSuccessEvent = Dict[str, Any] # keys: type='scrape_success', metadata={...}

# Basic Topic Validation Pattern (Example: Allow letters, numbers, spaces, common punctuation)
# Adjust as needed for stricter or more permissive validation.
VALID_TOPIC_PATTERN = re.compile(r"^[a-zA-Z0-9\s.,!?'\"\-():&]+$")
MAX_TOPIC_LENGTH = 300

def run_research_process(topic: str) -> Generator[Dict[str, Any], None, None]:
    """
    Executes the entire research process as a generator, yielding events.

    Yields dictionaries representing different stages and data:
    - {'type': 'progress', 'message': str, 'is_error': bool, 'is_fatal': bool}
    - {'type': 'event', 'data': Dict[str, Any]} (e.g., for llm_chunk, stream_start, complete)
    - {'type': 'scrape_success', 'metadata': Dict[str, str]} (incl. temp_filepath)

    Args:
        topic: The research topic.

    Returns:
        None. The function yields events until completion or fatal error.
    """
    # --- Research state variables ---
    scraped_source_metadata_list: List[Dict[str, Any]] = []
    research_plan: List[Dict[str, Any]] = []
    accumulated_synthesis_md: str = ""
    final_report_markdown: str = ""
    url_to_index_map: Dict[str, int] = {}
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
    # Optional: More stringent character validation
    # if not VALID_TOPIC_PATTERN.match(topic):
    #     logger.error(f"Topic contains invalid characters: {topic}")
    #     yield {'type': 'progress', 'message': "Fatal Error: Research topic contains invalid characters.", 'is_error': True, 'is_fatal': True}
    #     return

    logger.info(f"Starting research process for topic: '{topic}'")

    try: # <<< START OF MAIN TRY BLOCK >>>

        # === Step 1: Generate Research Plan ===
        yield {'type': 'progress', 'message': f"Generating research plan for: '{topic}'...", 'is_error': False, 'is_fatal': False}
        # System prompt guides the LLM's role and output format.
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
        try:
            plan_response = call_gemini(prompt=plan_user_prompt, system_prompt=plan_system_prompt)
            research_plan = parse_research_plan(plan_response) # Utils function handles JSON block extraction and parsing

            if not research_plan or not isinstance(research_plan, list) or not research_plan[0].get("step") or research_plan[0]["step"].startswith("Failed"):
                 fail_reason = research_plan[0]["step"] if (research_plan and isinstance(research_plan, list) and research_plan[0].get("step")) else "Could not parse plan structure from LLM."
                 # Avoid logging potentially huge raw response in case of failure
                 raw_snippet = f" Raw Response Snippet: '{plan_response[:150].strip()}...'" if plan_response else " (LLM Response was empty)"
                 logger.error(f"Failed to create/parse research plan. Reason: {fail_reason}.{raw_snippet}")
                 yield {'type': 'progress', 'message': f"Fatal Error: Failed to create or parse research plan. Please try rephrasing the topic or check LLM status. Reason: {escape(fail_reason)}", 'is_error': True, 'is_fatal': True}
                 fatal_error_occurred = True
                 return # Stop generation
        except (ValueError, RuntimeError, Exception) as e:
             logger.error(f"LLM Error generating research plan: {e}", exc_info=True)
             # Provide a user-friendly error, avoid leaking raw exception details
             error_msg = f"Fatal Error: An issue occurred while generating the research plan ({type(e).__name__}). Please check LLM configuration and logs."
             yield {'type': 'progress', 'message': error_msg, 'is_error': True, 'is_fatal': True}
             fatal_error_occurred = True
             return # Stop generation

        yield {'type': 'progress', 'message': f"Generated {len(research_plan)} step plan.", 'is_error': False, 'is_fatal': False}
        for i, step in enumerate(research_plan):
             step_desc = step.get('step', 'Unnamed Step')
             step_keywords = step.get('keywords', [])
             yield {'type': 'progress', 'message': f"  Step {i+1}: {step_desc[:100]}{'...' if len(step_desc)>100 else ''} (Keywords: {step_keywords})", 'is_error': False, 'is_fatal': False}


        # === Step 2a: Search and Collect URLs ===
        yield {'type': 'progress', 'message': "Starting web search...", 'is_error': False, 'is_fatal': False}
        start_search_time = time.time()
        all_urls_from_search_step = set()
        total_search_errors = 0
        total_search_queries_attempted = 0

        for i, step in enumerate(research_plan):
            step_desc = step.get('step', f'Unnamed Step {i+1}')
            keywords = step.get('keywords', [])
            progress_msg = f"Searching - Step {i+1}/{len(research_plan)}: '{step_desc[:70]}{'...' if len(step_desc)>70 else ''}'"
            yield {'type': 'progress', 'message': progress_msg, 'is_error': False, 'is_fatal': False}

            if not keywords or not isinstance(keywords, list) or not any(kw.strip() for kw in keywords):
                yield {'type': 'progress', 'message': "  -> No valid keywords provided for this step, skipping search.", 'is_error': False, 'is_fatal': False}
                continue

            # Use only valid, non-empty keywords
            valid_keywords = [kw for kw in keywords if isinstance(kw, str) and kw.strip()]
            if not valid_keywords:
                 yield {'type': 'progress', 'message': "  -> Keywords list contained only invalid entries, skipping search.", 'is_error': False, 'is_fatal': False}
                 continue

            total_search_queries_attempted += 1
            step_urls, step_errors = perform_web_search(valid_keywords) # web_research handles internal logic/retries

            if step_errors:
                total_search_errors += len(step_errors)
                for err in step_errors:
                    # Escape error message for safe display
                    yield {'type': 'progress', 'message': f"    -> Search Warning: {escape(err[:200])}", 'is_error': True, 'is_fatal': False}

            new_urls_count = len(set(step_urls) - all_urls_from_search_step)
            all_urls_from_search_step.update(step_urls)
            yield {'type': 'progress', 'message': f"  -> Found {len(step_urls)} URLs ({new_urls_count} new). Total unique: {len(all_urls_from_search_step)}.", 'is_error': False, 'is_fatal': False}

            # Avoid hammering search engine if many steps
            if i < len(research_plan) - 1:
                 time.sleep(config.INTER_SEARCH_DELAY_SECONDS)

        search_duration = time.time() - start_search_time
        yield {'type': 'progress', 'message': f"Search phase completed in {search_duration:.2f}s.", 'is_error': False, 'is_fatal': False}
        if total_search_errors > 0:
            yield {'type': 'progress', 'message': f"Collected {len(all_urls_from_search_step)} total unique URLs from {total_search_queries_attempted} queries ({total_search_errors} search errors occurred).", 'is_error': True, 'is_fatal': False}
        else:
            yield {'type': 'progress', 'message': f"Collected {len(all_urls_from_search_step)} total unique URLs from {total_search_queries_attempted} queries.", 'is_error': False, 'is_fatal': False}


        # --- Filter URLs ---
        # Apply filtering *before* scraping to save resources
        urls_to_scrape_list = []
        skipped_urls = set()
        yield {'type': 'progress', 'message': "Filtering URLs for scraping...", 'is_error': False, 'is_fatal': False}

        # More robust filtering (lowercase once, check common non-content domains/patterns)
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

        for url in sorted(list(all_urls_from_search_step)):
             if len(urls_to_scrape_list) >= config.MAX_TOTAL_URLS_TO_SCRAPE:
                  yield {'type': 'progress', 'message': f"  -> Reached URL scraping limit ({config.MAX_TOTAL_URLS_TO_SCRAPE}). Skipping remaining URLs.", 'is_error': False, 'is_fatal': False}
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
                      urls_to_scrape_list.append(url)
                 else:
                      skipped_urls.add(url)
                      # Log reason for skipping (optional, can be verbose)
                      reason = "unwanted scheme" if is_unwanted_scheme else \
                               "file extension" if is_file_extension else \
                               "common non-content domain/path" if is_common_non_content else \
                               "local address" if is_local else "unknown filter"
                      logger.debug(f"Filtering: Skipped URL '{url[:70]}...' Reason: {reason}")

             except Exception as filter_err:
                 logger.warning(f"Error during URL filtering for '{url[:70]}...': {filter_err}", exc_info=False)
                 skipped_urls.add(url)


        yield {'type': 'progress', 'message': f"Selected {len(urls_to_scrape_list)} URLs for scraping after filtering ({len(skipped_urls)} skipped, limit was {config.MAX_TOTAL_URLS_TO_SCRAPE}).", 'is_error': False, 'is_fatal': False}

        if not urls_to_scrape_list:
             logger.error("No suitable URLs found to scrape after search and filtering.")
             yield {'type': 'progress', 'message': "Fatal Error: No suitable URLs found to scrape. Cannot proceed. Try broadening the topic or checking search engine status.", 'is_error': True, 'is_fatal': True}
             fatal_error_occurred = True
             return # Stop generation


        # === Step 2b: Scrape URLs Concurrently ===
        yield {'type': 'progress', 'message': f"Starting concurrent scraping ({config.MAX_WORKERS} workers)...", 'is_error': False, 'is_fatal': False}
        start_scrape_time = time.time()
        scraped_source_metadata_list = [] # Reset before scraping
        processed_scrape_count = 0
        successful_scrape_count = 0
        scrape_errors = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS, thread_name_prefix="Scraper") as executor:
            future_to_url = {executor.submit(scrape_url, url): url for url in urls_to_scrape_list}

            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                processed_scrape_count += 1
                try:
                    # scrape_url now returns None on failure or if content is unsuitable (e.g., too short after sanitizing)
                    result_dict = future.result()
                    if result_dict and 'temp_filepath' in result_dict and result_dict['temp_filepath']:
                        # Basic validation of returned dict
                        if os.path.exists(result_dict['temp_filepath']):
                            scraped_source_metadata_list.append(result_dict)
                            successful_scrape_count += 1
                            yield {'type': 'scrape_success', 'metadata': result_dict} # Signal success for cleanup tracking
                        else:
                            logger.warning(f"Scrape task for {url[:70]}... reported success but temp file missing: {result_dict['temp_filepath']}")
                            scrape_errors.append(f"Temp file missing for {url[:70]}...")
                    # else: No result means scraping failed or content was filtered by scrape_url itself, already logged there.

                except Exception as exc:
                    # Catch errors raised by the future itself (e.g., worker exceptions not caught in scrape_url)
                    logger.error(f"Unexpected error in scraping task future for {url[:70]}...: {exc}", exc_info=True)
                    err_msg = f"Scrape task failed for {url[:60]}...: {type(exc).__name__}"
                    scrape_errors.append(err_msg)
                    yield {'type': 'progress', 'message': f"    -> {err_msg}", 'is_error': True, 'is_fatal': False}

                # Update progress periodically
                if processed_scrape_count % 5 == 0 or processed_scrape_count == len(urls_to_scrape_list):
                      progress_perc = (processed_scrape_count * 100) // len(urls_to_scrape_list)
                      yield {'type': 'progress', 'message': f"  -> Scraping Progress: {processed_scrape_count}/{len(urls_to_scrape_list)} URLs processed ({progress_perc}%). Successful: {successful_scrape_count}", 'is_error': False, 'is_fatal': False}

        scrape_duration = time.time() - start_scrape_time
        yield {'type': 'progress', 'message': f"Scraping finished in {scrape_duration:.2f}s. Successfully scraped and sanitized content from {successful_scrape_count} URLs.", 'is_error': False, 'is_fatal': False}
        if scrape_errors:
             yield {'type': 'progress', 'message': f"  -> Encountered {len(scrape_errors)} errors during scraping.", 'is_error': True, 'is_fatal': False}


        if not scraped_source_metadata_list:
            logger.error("Failed to scrape any usable content successfully after sanitization and filtering.")
            yield {'type': 'progress', 'message': "Fatal Error: Failed to gather sufficient web content after scraping and filtering. Cannot proceed with synthesis.", 'is_error': True, 'is_fatal': True}
            fatal_error_occurred = True
            return # Stop generation

        # Ensure scraped list maintains the order of urls_to_scrape_list for consistency
        scraped_url_map = {item['url']: item for item in scraped_source_metadata_list}
        ordered_scraped_metadata_list = [scraped_url_map[url] for url in urls_to_scrape_list if url in scraped_url_map]
        scraped_source_metadata_list = ordered_scraped_metadata_list


        # === Step 3: Generate Bibliography Map ===
        # This map is crucial for replacing URLs with footnote markers later
        url_to_index_map, bibliography_prompt_list = generate_bibliography_map(scraped_source_metadata_list)
        yield {'type': 'progress', 'message': f"Generated bibliography map for {len(url_to_index_map)} successfully processed sources.", 'is_error': False, 'is_fatal': False}


        # === Step 4: Synthesize Information (Streaming, RAM Optimized) ===
        yield {'type': 'progress', 'message': f"Synthesizing information from {len(scraped_source_metadata_list)} sources using {config.GOOGLE_MODEL_NAME}...", 'is_error': False, 'is_fatal': False}
        yield {'type': 'event', 'data': {'type': 'stream_start', 'target': 'synthesis'}}

        context_for_llm_parts = []
        current_total_chars = 0
        sources_included_count = 0
        skipped_sources_count = 0

        yield {'type': 'progress', 'message': f"  -> Preparing context for synthesis (limit ~{config.MAX_CONTEXT_CHARS // 1000}k chars)...", 'is_error': False, 'is_fatal': False}

        # Build context by reading *sanitized* content from temp files
        for source_metadata in scraped_source_metadata_list:
            filepath = source_metadata.get('temp_filepath')
            url = source_metadata.get('url')
            content = None # Initialize content as None

            if not filepath or not url or not os.path.exists(filepath):
                 logger.warning(f"Skipping source for synthesis (missing temp file/metadata). URL: {url or 'Unknown'}, Path: {filepath}")
                 yield {'type': 'progress', 'message': f"  -> Warning: Skipping source, missing temp file or metadata for URL {url or 'Unknown'}", 'is_error': True, 'is_fatal': False}
                 skipped_sources_count += 1
                 continue

            try:
                 file_size = os.path.getsize(filepath)
                 if file_size == 0:
                      logger.warning(f"Skipping empty temp file {os.path.basename(filepath)} for {url[:60]}...")
                      yield {'type': 'progress', 'message': f"  -> Warning: Skipping empty temp file for {url[:60]}...", 'is_error': True, 'is_fatal': False}
                      skipped_sources_count += 1
                      continue

                 # Read content, check if adding exceeds limit *before* appending
                 # Estimate size contribution (URL length + content length + JSON overhead)
                 estimated_addition = len(url) + file_size + 50 # Rough estimate
                 if (current_total_chars + estimated_addition) <= config.MAX_CONTEXT_CHARS:
                     try:
                         with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                             content = f.read() # Read the sanitized content

                         # Ensure content isn't just whitespace (should be handled by scrape_url, but double-check)
                         if content and content.strip():
                             context_for_llm_parts.append({'url': url, 'content': content})
                             current_total_chars += estimated_addition # Use estimate for tracking
                             sources_included_count += 1
                         else:
                             logger.warning(f"Read empty or whitespace-only content from temp file {os.path.basename(filepath)} for {url[:60]}... (Post-Sanitization Check)")
                             yield {'type': 'progress', 'message': f"  -> Warning: Read empty content from temp file for {url[:60]}...", 'is_error': True, 'is_fatal': False}
                             skipped_sources_count += 1

                     except Exception as read_err:
                         logger.error(f"Error reading temp file {os.path.basename(filepath)} for {url[:60]}: {read_err}", exc_info=False)
                         yield {'type': 'progress', 'message': f"  -> Error reading temp file for {url[:60]}...: {escape(str(read_err))}", 'is_error': True, 'is_fatal': False}
                         skipped_sources_count += 1
                 else:
                      # Context limit reached, stop adding sources
                      logger.warning(f"Context limit ({config.MAX_CONTEXT_CHARS // 1000}k chars) reached. Stopping context build. Included {sources_included_count} sources.")
                      yield {'type': 'progress', 'message': f"  -> Context limit reached. Synthesizing based on first {sources_included_count}/{len(scraped_source_metadata_list)} sources (~{current_total_chars // 1000}k chars).", 'is_error': True, 'is_fatal': False}
                      skipped_sources_count += (len(scraped_source_metadata_list) - sources_included_count)
                      break # Stop processing further sources

            except OSError as e:
                 logger.error(f"Error accessing temp file {os.path.basename(filepath)} for {url[:60]}...: {e}", exc_info=False)
                 yield {'type': 'progress', 'message': f"  -> Error accessing temp file metadata for {url[:60]}...: {escape(str(e))}", 'is_error': True, 'is_fatal': False}
                 skipped_sources_count += 1
            except Exception as e:
                 logger.error(f"Unexpected error processing temp file {os.path.basename(filepath)} for {url[:60]}...: {e}", exc_info=False)
                 yield {'type': 'progress', 'message': f"  -> Unexpected error processing temp file for {url[:60]}...: {escape(str(e))}", 'is_error': True, 'is_fatal': False}
                 skipped_sources_count += 1
            finally:
                 # Release memory if content was read but not used (or used)
                 del content

        if sources_included_count == 0:
             logger.error("No source content could be prepared for synthesis after reading temp files.")
             yield {'type': 'progress', 'message': "Fatal Error: No source content available for synthesis. Cannot proceed.", 'is_error': True, 'is_fatal': True}
             fatal_error_occurred = True
             return # Stop generation

        # Provide feedback on skipped sources if any
        if skipped_sources_count > 0:
             yield {'type': 'progress', 'message': f"  -> Note: Skipped {skipped_sources_count} sources during context preparation due to errors, size limits, or missing files.", 'is_error': True, 'is_fatal': False}

        estimated_tokens = current_total_chars / config.CHARS_PER_TOKEN_ESTIMATE
        yield {'type': 'progress', 'message': f"  -> Prepared synthesis context using {sources_included_count} sources (~{current_total_chars // 1000}k chars / ~{estimated_tokens / 1000:.1f}k est. tokens).", 'is_error': False, 'is_fatal': False}

        # Prepare context for LLM (serialize the list of dicts)
        try:
            context_json_str = json.dumps(context_for_llm_parts, indent=None, ensure_ascii=False, separators=(',', ':')) # Compact JSON
            del context_for_llm_parts # Free memory
            actual_context_chars = len(context_json_str)
            logger.info(f"Actual serialized context size: {actual_context_chars // 1000}k chars.")
            if actual_context_chars > config.MAX_CONTEXT_CHARS * 1.1: # Check if estimate was way off
                 logger.warning(f"Serialized context ({actual_context_chars // 1000}k) significantly exceeds estimated limit ({config.MAX_CONTEXT_CHARS // 1000}k). LLM call might fail.")
                 yield {'type': 'progress', 'message': f"  -> Warning: Prepared context size ({actual_context_chars // 1000}k chars) exceeds target limit. LLM call might fail due to size.", 'is_error': True, 'is_fatal': False}

        except (TypeError, OverflowError, Exception) as json_err:
            logger.error(f"Failed to serialize context parts to JSON: {json_err}", exc_info=True)
            yield {'type': 'progress', 'message': f"Fatal Error: Could not prepare context data for LLM due to serialization issue: {escape(str(json_err))}", 'is_error': True, 'is_fatal': True}
            fatal_error_occurred = True
            return

        # --- Synthesis Prompt ---
        # Use clear system and user prompts
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

        # --- Execute Synthesis Stream ---
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
                elif event_type == 'stream_error':
                    synthesis_stream_error = result.get('message', 'Unknown stream error')
                    synthesis_stream_fatal = result.get('is_fatal', False)
                    logger.error(f"LLM stream error during synthesis: {synthesis_stream_error}")
                    # Pass error to client progress
                    yield {'type': 'progress', 'message': f"LLM Error (Synthesis): {escape(synthesis_stream_error)}", 'is_error': True, 'is_fatal': synthesis_stream_fatal}
                    if synthesis_stream_fatal:
                        fatal_error_occurred = True
                        return # Stop generation
                    # Break on non-fatal stream error? Or continue accumulating? Let's break.
                    break
                elif event_type == 'stream_warning':
                     msg = result.get('message', 'Unknown stream warning')
                     logger.warning(f"LLM Stream Warning (Synthesis): {msg}")
                     yield {'type': 'progress', 'message': f"LLM Warning (Synthesis): {escape(msg)}", 'is_error': True, 'is_fatal': False}
                elif event_type == 'stream_end':
                     reason = result.get('finish_reason', 'N/A')
                     logger.info(f"LLM synthesis stream finished. Finish Reason: {reason}")
                     # Handle potential non-ideal finish reasons if needed
                     if reason not in ["STOP", "MAX_TOKENS", "1", "2", None]: # Check for unexpected reasons
                         logger.warning(f"Synthesis stream finished with non-standard reason: {reason}")
                         yield {'type': 'progress', 'message': f"Warning: Synthesis stream finished with reason: {reason}. Output might be incomplete.", 'is_error': True, 'is_fatal': False}
                     break # Normal end or handled warning
        except Exception as e:
             logger.error(f"Fatal error processing LLM synthesis stream: {e}", exc_info=True)
             yield {'type': 'progress', 'message': f"Fatal Error: Unexpected issue during information synthesis ({type(e).__name__}).", 'is_error': True, 'is_fatal': True}
             fatal_error_occurred = True
             return

        yield {'type': 'progress', 'message': "Synthesis generation finished." + (" (Errors occurred)" if synthesis_stream_error else ""), 'is_error': bool(synthesis_stream_error), 'is_fatal': False}

        if not accumulated_synthesis_md.strip():
             # If synthesis is empty AND an error occurred, the error is the likely cause.
             # If synthesis is empty but NO error occurred, the LLM produced nothing (maybe no relevant info found).
             if synthesis_stream_error:
                 logger.error(f"Synthesis resulted in empty content, likely due to stream error: {synthesis_stream_error}")
                 # Don't necessarily make it fatal yet, maybe report can still be generated
                 yield {'type': 'progress', 'message': "Warning: Synthesis was empty, potentially due to an earlier LLM error.", 'is_error': True, 'is_fatal': False}
             else:
                 logger.warning("Synthesis resulted in empty content, but no stream error reported. LLM might have found no relevant info.")
                 yield {'type': 'progress', 'message': "Note: Synthesis phase produced no content. This might be expected if sources lacked relevant information.", 'is_error': False, 'is_fatal': False}
                 # Proceed, the report should handle this lack of input.


        # === Step 5: Generate Final Report (Streaming) ===
        yield {'type': 'progress', 'message': f"Generating final report using {config.GOOGLE_MODEL_NAME}...", 'is_error': False, 'is_fatal': False}
        yield {'type': 'event', 'data': {'type': 'stream_start', 'target': 'report'}}

        # --- Prepare Report Prompt ---
        # Estimate base prompt length (plan, bibliography, instructions)
        # Use compact JSON for estimates to be closer to actual prompt size
        try:
            plan_json_compact = json.dumps(research_plan, separators=(',', ':'))
            base_prompt_elements_len = (
                len(topic) + len(plan_json_compact) + len(bibliography_prompt_list) + 3000 # Estimate for instructions text
            )
        except Exception as json_err:
             logger.error(f"Error estimating report prompt base length: {json_err}")
             base_prompt_elements_len = 5000 # Fallback estimate

        available_chars_for_synthesis_in_report = config.MAX_CONTEXT_CHARS - base_prompt_elements_len

        # Truncate synthesis *if necessary* to fit context window for the report generation call
        if len(accumulated_synthesis_md) > available_chars_for_synthesis_in_report:
            # Keep slightly less than available to be safe
            chars_to_keep = max(0, available_chars_for_synthesis_in_report - 500) # Ensure non-negative
            truncated_synthesis_md = accumulated_synthesis_md[:chars_to_keep] + "\n\n... [Synthesis truncated due to context limits for final report generation]"
            logger.warning(f"Truncating synthesis markdown (from {len(accumulated_synthesis_md)} to {len(truncated_synthesis_md)} chars) for report prompt.")
            yield {'type': 'progress', 'message': "  -> Warning: Synthesized text truncated for final report generation due to context limits. Report might be incomplete.", 'is_error': True, 'is_fatal': False}
        else:
            truncated_synthesis_md = accumulated_synthesis_md

        # --- Report Prompt ---
        # Use clear system and user prompts again
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
        # Free memory from intermediate variables
        del accumulated_synthesis_md
        del truncated_synthesis_md
        del research_plan # No longer needed
        del bibliography_prompt_list

        # --- Execute Report Stream ---
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
                elif event_type == 'stream_error':
                    report_stream_error = result.get('message', 'Unknown stream error')
                    report_stream_fatal = result.get('is_fatal', False)
                    logger.error(f"LLM stream error during report generation: {report_stream_error}")
                    yield {'type': 'progress', 'message': f"LLM Error (Report): {escape(report_stream_error)}", 'is_error': True, 'is_fatal': report_stream_fatal}
                    if report_stream_fatal:
                        fatal_error_occurred = True
                        return # Stop generation
                    break
                elif event_type == 'stream_warning':
                     msg = result.get('message', 'Unknown stream warning')
                     logger.warning(f"LLM Stream Warning (Report): {msg}")
                     yield {'type': 'progress', 'message': f"LLM Warning (Report): {escape(msg)}", 'is_error': True, 'is_fatal': False}
                elif event_type == 'stream_end':
                     reason = result.get('finish_reason', 'N/A')
                     logger.info(f"LLM report stream finished. Finish Reason: {reason}")
                     if reason not in ["STOP", "MAX_TOKENS", "1", "2", None]:
                         logger.warning(f"Report stream finished with non-standard reason: {reason}")
                         yield {'type': 'progress', 'message': f"Warning: Report stream finished with reason: {reason}. Output might be incomplete.", 'is_error': True, 'is_fatal': False}
                     break # Normal end or handled warning
        except Exception as e:
             logger.error(f"Fatal error processing LLM report stream: {e}", exc_info=True)
             yield {'type': 'progress', 'message': f"Fatal Error: Unexpected issue during final report generation ({type(e).__name__}).", 'is_error': True, 'is_fatal': True}
             fatal_error_occurred = True
             return

        yield {'type': 'progress', 'message': "Report generation finished." + (" (Errors occurred)" if report_stream_error else ""), 'is_error': bool(report_stream_error), 'is_fatal': False}

        if not final_report_markdown.strip():
            if report_stream_error:
                 logger.error(f"Final report generation resulted in empty content due to stream error: {report_stream_error}")
                 # Generate a fallback report indicating failure
                 final_report_markdown = f"# Research Report: {escape(topic)}\n\n## Error\n\nThe final report could not be generated due to an error during the process:\n\n```\n{escape(report_stream_error)}\n```"
                 yield {'type': 'progress', 'message': "Error: Final report generation failed due to an LLM error.", 'is_error': True, 'is_fatal': False} # Not fatal to the *stream*, but the *report* failed
            else:
                 logger.error("Final report generation resulted in empty content, but no stream error reported.")
                 final_report_markdown = f"# Research Report: {escape(topic)}\n\n## Error\n\nReport generation failed unexpectedly and produced no content. Please check the logs."
                 yield {'type': 'progress', 'message': "Error: Final report generation resulted in empty content.", 'is_error': True, 'is_fatal': False}


        # === Step 6: Final Processing and Completion ===
        yield {'type': 'progress', 'message': "Processing final report for display...", 'is_error': False, 'is_fatal': False}

        # --- Strip potential Markdown code fences (LLMs sometimes add them) ---
        cleaned_report_markdown = final_report_markdown.strip()
        # More robust fence removal
        if cleaned_report_markdown.startswith("```markdown") and cleaned_report_markdown.endswith("```"):
             cleaned_report_markdown = cleaned_report_markdown[len("```markdown"): -len("```")].strip()
             logger.info("Stripped surrounding ```markdown fences from the final report.")
        elif cleaned_report_markdown.startswith("```") and cleaned_report_markdown.endswith("```"):
             # Handle case with just ``` fence
             possible_content = cleaned_report_markdown[3:-3].strip()
             # Avoid stripping if the content *itself* is meant to be a code block
             if not possible_content.startswith('#'): # Simple heuristic: If content doesn't start like the report title, it might be a code block itself. Keep fences.
                  logger.info("Stripped surrounding ``` fences from the final report.")
                  cleaned_report_markdown = possible_content
             else:
                  logger.info("Detected ``` fences, but content looks like the report itself. Keeping fences.")


        final_report_markdown = cleaned_report_markdown # Use the potentially cleaned version

        # --- Convert Markdown to HTML (Server-Side) ---
        # This ensures footnotes and other markdown features are rendered correctly
        # and reduces client-side processing load/complexity.
        report_html = convert_markdown_to_html(final_report_markdown)

        if report_html.strip().lower().startswith(('<p><em>report content is empty', '<pre><strong>error', '<p><em>markdown conversion resulted', 'report generation failed')):
            logger.error("Final Markdown report content seems invalid or conversion failed.")
            yield {'type': 'progress', 'message': "Error preparing final report for display. Check logs.", 'is_error': True, 'is_fatal': False}
            # Use the potentially error-containing HTML generated by convert_markdown_to_html or the raw markdown
            if not report_html.strip(): # If conversion resulted in truly empty string
                 report_html = f"<h2>Report Display Error</h2><p>Could not convert report Markdown to HTML. Raw Markdown content:</p><pre><code>{escape(final_report_markdown)}</code></pre>"
        elif not report_html.strip(): # Should be caught above, but double-check
             logger.error("Markdown conversion resulted in empty HTML unexpectedly.")
             yield {'type': 'progress', 'message': "Error: Markdown conversion resulted in empty HTML.", 'is_error': True, 'is_fatal': False}
             report_html = f"<h2>Report Display Error</h2><p>Markdown conversion resulted in empty content. Raw Markdown content:</p><pre><code>{escape(final_report_markdown)}</code></pre>"


        # --- Yield Final Data Package ---
        yield {'type': 'progress', 'message': "Sending final results to client...", 'is_error': False, 'is_fatal': False}
        final_data = {
            'type': 'complete',
            'report_html': report_html
            # Optionally send raw markdown if client needs it:
            # 'report_markdown': final_report_markdown
        }
        yield {'type': 'event', 'data': final_data}

        end_time_total = time.time()
        total_duration = end_time_total - start_time_total
        logger.info(f"Research process for '{topic}' completed in {total_duration:.2f} seconds.")
        yield {'type': 'progress', 'message': f"Research process completed successfully in {total_duration:.2f} seconds.", 'is_error': False, 'is_fatal': False}

    except Exception as e:
        # Catch-all for unexpected errors within the orchestrator's main try block
        logger.error(f"FATAL: Unhandled exception in research orchestrator for topic '{topic}': {e}", exc_info=True)
        # Generic error message for the client
        error_msg = "An unexpected server error occurred during research orchestration. Please check server logs for details."
        if not fatal_error_occurred: # Avoid sending duplicate fatal errors
            try:
                 # Yield fatal error progress event
                 yield {'type': 'progress', 'message': error_msg, 'is_error': True, 'is_fatal': True}
            except Exception as callback_err:
                 # Log if we can't even yield the error message
                 logger.critical(f"Failed to yield final fatal error message in orchestrator: {callback_err}")
        fatal_error_occurred = True # Ensure flag is set
    finally:
        # This block executes whether the try block succeeded or failed
        if fatal_error_occurred:
            logger.error(f"Orchestrator generator stopped prematurely due to fatal error for topic: '{topic}'.")
        else:
            logger.info(f"Orchestrator generator finished executing for topic: '{topic}'.")
        # Actual cleanup of temp files is handled by the caller (app.py) using the
        # 'scrape_success' events yielded during the process.