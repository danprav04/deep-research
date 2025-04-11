# research_orchestrator.py
import time
import json
import os
import traceback
import logging
from typing import Dict, Any, List, Callable, Optional, Tuple
from html import escape
import concurrent.futures

import config as config
from llm_interface import call_gemini, stream_gemini
from web_research import perform_web_search, scrape_url
from utils import (
    parse_research_plan, generate_bibliography_map, convert_markdown_to_html
)

# Configure logger for this module
logger = logging.getLogger(__name__)

# Define callback function types for clarity
SendEventCallback = Callable[[Dict[str, Any]], None]
SendProgressCallback = Callable[[str, bool, bool], None]


def run_research_process(
    topic: str,
    send_event_callback: SendEventCallback,
    send_progress_callback: SendProgressCallback
) -> List[Optional[str]]:
    """
    Executes the entire research process from planning to report generation.

    Args:
        topic: The research topic.
        send_event_callback: Function to send SSE data events to the client.
        send_progress_callback: Function to send progress/error messages to the client.

    Returns:
        A list of temporary file paths created during scraping that need cleanup.
        Returns an empty list if no files were created or if a fatal error occurred early.
    """
    # --- Research state variables ---
    scraped_source_metadata_list: List[Dict[str, Any]] = []
    temp_files_to_clean: List[Optional[str]] = []
    research_plan: List[Dict[str, Any]] = []
    accumulated_synthesis_md: str = ""
    final_report_markdown: str = ""
    url_to_index_map: Dict[str, int] = {}
    start_time_total = time.time()

    try: # <<< START OF MAIN TRY BLOCK >>>
        logger.info(f"Starting research process for topic: '{topic}'")

        # === Step 1: Generate Research Plan ===
        send_progress_callback(f"Generating research plan for: '{topic}'...", False, False)
        plan_prompt = f"""
        Create a detailed, step-by-step research plan with 5-7 distinct steps for the topic: "{topic}"
        Each step should represent a specific question or area of inquiry relevant to a concise research report.
        Format the output STRICTLY as a JSON list of objects within a ```json ... ``` block.
        Each object MUST have keys "step" (string description) and "keywords" (list of 2-3 relevant search query strings).

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
        Output ONLY the valid JSON list inside the markdown block. Generate 5 to 7 steps for the topic: "{topic}".
        """
        try:
            plan_response = call_gemini(plan_prompt)
            research_plan = parse_research_plan(plan_response) # parse_research_plan handles basic error reporting
            if not research_plan or (len(research_plan) == 1 and research_plan[0]["step"].startswith("Failed")):
                 fail_reason = research_plan[0]["step"] if research_plan else "Could not parse plan structure from LLM."
                 raw_snippet = f" Raw LLM Response Snippet: '{plan_response[:150]}...'" if plan_response else " (LLM Response was empty)"
                 logger.error(f"Failed to create/parse research plan. Reason: {fail_reason}.{raw_snippet}")
                 send_progress_callback(f"Fatal Error: Failed to create or parse research plan. Reason: {fail_reason}.", True, True)
                 return [] # Fatal error, return empty list
        except Exception as e:
             logger.error(f"LLM Error generating research plan: {e}", exc_info=True)
             send_progress_callback(f"Fatal Error: Failed during research plan generation: {e}", True, True)
             return [] # Fatal error

        send_progress_callback(f"Generated {len(research_plan)} step plan.", False, False)
        for i, step in enumerate(research_plan):
             step_desc = step.get('step', 'Unnamed Step')
             step_keywords = step.get('keywords', [])
             send_progress_callback(f"  Step {i+1}: {step_desc} (Keywords: {step_keywords})", False, False)

        # === Step 2a: Search and Collect URLs ===
        send_progress_callback("Starting web search...", False, False)
        start_search_time = time.time()
        all_urls_from_search_step = set()
        total_search_errors = 0
        total_search_queries = 0

        for i, step in enumerate(research_plan):
            step_desc = step.get('step', f'Unnamed Step {i+1}')
            keywords = step.get('keywords', [])
            progress_msg = f"Searching - Step {i+1}/{len(research_plan)}: '{step_desc[:70]}{'...' if len(step_desc)>70 else ''}'"
            send_progress_callback(progress_msg, False, False)

            if not keywords:
                send_progress_callback("  -> No keywords provided for this step, skipping search.", False, False)
                continue

            total_search_queries += 1
            # perform_web_search aggregates results from providers defined within it
            step_urls, step_errors = perform_web_search(keywords)

            if step_errors:
                total_search_errors += len(step_errors)
                for err in step_errors:
                    send_progress_callback(f"    -> Search Warning: {err}", True, False) # Report non-fatal search errors

            new_urls_count = len(set(step_urls) - all_urls_from_search_step)
            all_urls_from_search_step.update(step_urls)
            send_progress_callback(f"  -> Found {len(step_urls)} URLs for step keywords, {new_urls_count} new. Total unique URLs so far: {len(all_urls_from_search_step)}.", False, False)

            # Add a small delay between keyword searches within a step if needed
            if i < len(research_plan) - 1:
                 time.sleep(config.INTER_SEARCH_DELAY_SECONDS)

        search_duration = time.time() - start_search_time
        send_progress_callback(f"Search phase completed in {search_duration:.2f}s.", False, False)
        send_progress_callback(f"Collected {len(all_urls_from_search_step)} total unique URLs ({total_search_errors} search errors).", False, False)

        # --- Filter URLs ---
        urls_to_scrape_list = []
        send_progress_callback("Filtering URLs for scraping...", False, False)
        for url in sorted(list(all_urls_from_search_step)):
             if len(urls_to_scrape_list) >= config.MAX_TOTAL_URLS_TO_SCRAPE:
                  send_progress_callback(f"  -> Reached URL scraping limit ({config.MAX_TOTAL_URLS_TO_SCRAPE}). Skipping remaining URLs.", False, False)
                  break

             lower_url = url.lower()
             # Basic checks for non-HTML/text content or unwanted schemes
             path_part = lower_url.split('?')[0].split('#')[0]
             is_file_extension = path_part.endswith(('.pdf', '.jpg', '.png', '.gif', '.zip', '.mp4', '.mp3', '.docx', '.xlsx', '.pptx', '.webp', '.svg', '.xml', '.css', '.js', '.jpeg', '.doc', '.xls', '.ppt', '.txt', '.exe', '.dmg', '.iso', '.rar', '.gz', '.tar', '.bz2', '.7z'))
             is_unwanted_scheme = lower_url.startswith(('mailto:', 'javascript:', 'ftp:', 'tel:', 'file:', 'data:'))
             is_local = lower_url.startswith(('localhost', '127.0.0.1')) # Often points to dev environments
             is_valid_http = url.startswith(('http://', 'https://'))

             if is_valid_http and not any([is_file_extension, is_unwanted_scheme, is_local]):
                  urls_to_scrape_list.append(url)
             # else:
             #     logger.debug(f"Filtered out URL: {url}")


        send_progress_callback(f"Selected {len(urls_to_scrape_list)} URLs for scraping after filtering (limit was {config.MAX_TOTAL_URLS_TO_SCRAPE}).", False, False)

        if not urls_to_scrape_list:
             logger.error("No suitable URLs found to scrape after search and filtering.")
             send_progress_callback("Fatal Error: No suitable URLs found to scrape. Cannot proceed.", True, True)
             return [] # Fatal error

        # === Step 2b: Scrape URLs Concurrently ===
        send_progress_callback(f"Starting concurrent scraping ({config.MAX_WORKERS} workers)...", False, False)
        start_scrape_time = time.time()
        scraped_source_metadata_list = [] # Reset before scraping
        # temp_files_to_clean is managed outside this function now
        processed_scrape_count = 0
        successful_scrape_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            # Submit all scrape tasks
            future_to_url = {executor.submit(scrape_url, url): url for url in urls_to_scrape_list}

            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                processed_scrape_count += 1
                try:
                    result_dict = future.result() # result() blocks until the future completes
                    if result_dict and 'temp_filepath' in result_dict and result_dict['temp_filepath']:
                        scraped_source_metadata_list.append(result_dict)
                        temp_files_to_clean.append(result_dict['temp_filepath']) # Add path to list for cleanup later
                        successful_scrape_count += 1
                        # logger.debug(f"Successfully scraped and saved temp file for: {url}")
                    # else: result was None (scrape failed or filtered out)
                        # logger.debug(f"Scrape returned None for: {url}")

                except Exception as exc:
                    # Catch errors from the scrape_url function itself (e.g., network issues not caught internally)
                    logger.error(f"Error during scraping task for {url[:70]}...: {exc}", exc_info=False) # Log traceback optionally
                    send_progress_callback(f"    -> Scrape Error for {url[:60]}...: {escape(str(exc))}", True, False)

                # Send progress update periodically
                if processed_scrape_count % 5 == 0 or processed_scrape_count == len(urls_to_scrape_list):
                      progress_perc = (processed_scrape_count * 100) // len(urls_to_scrape_list)
                      send_progress_callback(f"  -> Scraping Progress: {processed_scrape_count}/{len(urls_to_scrape_list)} URLs processed ({progress_perc}%). Successful scrapes: {successful_scrape_count}", False, False)

        scrape_duration = time.time() - start_scrape_time
        send_progress_callback(f"Scraping finished in {scrape_duration:.2f}s. Successfully scraped content from {successful_scrape_count} URLs.", False, False)

        if not scraped_source_metadata_list:
            logger.error("Failed to scrape any content successfully.")
            send_progress_callback("Fatal Error: Failed to scrape any content successfully. Cannot proceed with synthesis.", True, True)
            return temp_files_to_clean # Return files created so far for cleanup

        # Ensure scraped list is ordered consistently (optional, but helpful for reproducibility)
        scraped_url_map = {item['url']: item for item in scraped_source_metadata_list}
        ordered_scraped_metadata_list = [scraped_url_map[url] for url in urls_to_scrape_list if url in scraped_url_map]
        scraped_source_metadata_list = ordered_scraped_metadata_list


        # === Step 3: Generate Bibliography Map ===
        url_to_index_map, bibliography_prompt_list = generate_bibliography_map(scraped_source_metadata_list)
        send_progress_callback(f"Generated bibliography map for {len(url_to_index_map)} successfully scraped sources.", False, False)

        # === Step 4: Synthesize Information (Streaming, RAM Optimized) ===
        send_progress_callback(f"Synthesizing information from {len(scraped_source_metadata_list)} scraped sources using {config.GOOGLE_MODEL_NAME}...", False, False)
        send_event_callback({'type': 'stream_start', 'target': 'synthesis'}) # Signal start of synthesis stream

        context_for_llm_parts = [] # Build context incrementally
        current_total_chars = 0
        sources_included_count = 0
        estimated_total_scraped_chars = 0

        send_progress_callback(f"  -> Preparing context for synthesis (limit ~{config.MAX_CONTEXT_CHARS // 1000}k chars)...", False, False)

        # --- RAM Optimized Context Building ---
        for source_metadata in scraped_source_metadata_list:
            filepath = source_metadata.get('temp_filepath')
            url = source_metadata.get('url')
            if not filepath or not url or not os.path.exists(filepath):
                 logger.warning(f"Skipping source for synthesis, missing temp file or metadata. URL: {url or 'Unknown'}, Path: {filepath}")
                 send_progress_callback(f"  -> Warning: Skipping source, missing temp file or metadata for URL {url or 'Unknown'}", True, False)
                 continue

            try:
                 file_size = os.path.getsize(filepath)
                 estimated_total_scraped_chars += file_size

                 # Check if adding this file *exceeds* the limit
                 if (current_total_chars + file_size) <= config.MAX_CONTEXT_CHARS:
                     try:
                         with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                             content = f.read()
                         if content.strip(): # Only add if content is not just whitespace
                              # Append structured part for the prompt
                              context_for_llm_parts.append({'url': url, 'content': content})
                              current_total_chars += len(content) # Use actual length read
                              sources_included_count += 1
                         else:
                              logger.warning(f"Read empty or whitespace-only content from temp file {os.path.basename(filepath)} for {url[:60]}...")
                              send_progress_callback(f"  -> Warning: Read empty content from temp file for {url[:60]}...", True, False)
                     except Exception as read_err:
                         logger.error(f"Error reading temp file {os.path.basename(filepath)} for {url[:60]}: {read_err}", exc_info=False)
                         send_progress_callback(f"  -> Error reading temp file for {url[:60]}...: {read_err}", True, False)
                 else:
                      logger.warning(f"Context limit ({config.MAX_CONTEXT_CHARS // 1000}k chars) reached. Stopping context build. Included {sources_included_count} sources.")
                      send_progress_callback(f"  -> Context limit reached. Synthesizing based on first {sources_included_count}/{len(scraped_source_metadata_list)} sources (~{current_total_chars // 1000}k chars).", True, False) # Non-fatal warning
                      break # Stop adding more sources

            except OSError as e:
                 logger.error(f"Error accessing temp file {os.path.basename(filepath)} for {url[:60]}...: {e}", exc_info=False)
                 send_progress_callback(f"  -> Error accessing temp file metadata for {url[:60]}...: {e}", True, False)
            except Exception as e:
                 logger.error(f"Unexpected error processing temp file {os.path.basename(filepath)} for {url[:60]}...: {e}", exc_info=False)
                 send_progress_callback(f"  -> Unexpected error processing temp file for {url[:60]}...: {e}", True, False)

        if sources_included_count == 0:
             logger.error("No source content could be prepared for synthesis (possibly due to read errors or all files being too large).")
             send_progress_callback("Fatal Error: No source content available for synthesis. Cannot proceed.", True, True)
             return temp_files_to_clean # Return files for cleanup

        estimated_tokens = current_total_chars / config.CHARS_PER_TOKEN_ESTIMATE
        send_progress_callback(f"  -> Prepared synthesis context using {sources_included_count} sources (~{current_total_chars // 1000}k chars / ~{estimated_tokens / 1000:.1f}k est. tokens).", False, False)

        # Construct the final context string (JSON representation)
        try:
            context_json_str = json.dumps(context_for_llm_parts, indent=2, ensure_ascii=False)
        except Exception as json_err:
            logger.error(f"Failed to serialize context parts to JSON: {json_err}", exc_info=True)
            send_progress_callback(f"Fatal Error: Could not prepare context data for LLM: {json_err}", True, True)
            return temp_files_to_clean

        synthesis_prompt = f"""
        Analyze the provided web content about "{topic}" based on the research plan.
        Synthesize key information relevant to each plan step, citing sources accurately using ONLY the provided URLs.

        Research Topic: {topic}
        Research Plan:
        ```json
        {json.dumps(research_plan, indent=2)}
        ```
        Source Content ({len(context_for_llm_parts)} sources):
        ```json
        {context_json_str}
        ```
        Instructions:
        1. Address each plan step sequentially based on the "Research Plan".
        2. Find relevant information for each step across ALL provided sources in "Source Content".
        3. Synthesize findings concisely in **Markdown format**. Use headings (`### Step X: <Step Description from Plan>`), lists, bold, etc. for clarity. Structure the output strictly by plan steps. Start each step's synthesis with `### Step X: <Step Description from Plan>`.
        4. **MANDATORY CITATION**: Immediately after ANY piece of information derived from a specific source, cite it using the exact format: `[Source URL: <full_url_here>]`. Cite every distinct piece of information. Use the full URL provided in the "Source Content" section. If multiple sources support the same point, list citations consecutively: `[Source URL: <url1>][Source URL: <url2>]`.
        5. If no relevant information is found for a specific plan step in the provided sources, state clearly under the step heading: "No specific information found for this step in the provided sources."
        6. Use `---` as a separator ONLY between the synthesis for different plan steps.
        7. Output ONLY the synthesized Markdown content structured by plan steps. Do NOT include an introduction, conclusion, summary, or bibliography in this output. Focus solely on presenting the synthesized findings per step with citations.
        """

        accumulated_synthesis_md = ""
        synthesis_stream_error = None
        try:
            stream_generator = stream_gemini(synthesis_prompt)
            for result in stream_generator:
                if result['type'] == 'chunk':
                    send_event_callback({'type': 'llm_chunk', 'content': result['content'], 'target': 'synthesis'})
                    accumulated_synthesis_md += result['content']
                elif result['type'] == 'stream_error':
                    synthesis_stream_error = result['message']
                    logger.error(f"LLM stream error during synthesis: {synthesis_stream_error}")
                    # Determine if fatal based on common error messages
                    is_fatal_err = any(indicator in synthesis_stream_error.lower()
                                       for indicator in ["api key", "quota", "resource has been exhausted", "permission_denied", "billing", "invalid model"])
                    send_progress_callback(f"LLM stream error during synthesis: {synthesis_stream_error}", True, is_fatal_err)
                    if is_fatal_err: return temp_files_to_clean # Stop process on fatal API errors
                    break # Break on non-fatal stream error (like safety block during generation)
                elif result['type'] == 'stream_warning':
                     logger.warning(f"LLM Stream Warning (Synthesis): {result['message']}")
                     send_progress_callback(f"LLM Stream Warning (Synthesis): {result['message']}", True, False) # Report as non-fatal error
                elif result['type'] == 'stream_end':
                     logger.info(f"LLM synthesis stream ended. Finish Reason: {result.get('finish_reason', 'N/A')}")
                     break # Normal end
        except Exception as e:
             logger.error(f"Fatal error processing LLM synthesis stream: {e}", exc_info=True)
             send_progress_callback(f"Fatal error processing LLM synthesis stream: {escape(str(e))}", True, True)
             return temp_files_to_clean

        send_progress_callback("Synthesis generation finished.", False, False)
        if not accumulated_synthesis_md.strip() and not synthesis_stream_error:
             # It's possible the LLM genuinely found nothing relevant
             logger.warning("Synthesis resulted in empty content, but no stream error reported.")
             send_progress_callback("Warning: Synthesis resulted in empty content. The final report might lack detailed findings.", True, False) # Non-fatal warning
        elif not accumulated_synthesis_md.strip() and synthesis_stream_error:
             logger.error(f"Synthesis resulted in empty content due to stream error: {synthesis_stream_error}")
             # Progress message already sent by stream_error handling

        # === Step 5: Generate Final Report (Streaming) ===
        send_progress_callback(f"Generating final report using {config.GOOGLE_MODEL_NAME}...", False, False)
        send_event_callback({'type': 'stream_start', 'target': 'report'}) # Signal start of report stream

        # Estimate size of static parts of the prompt to see how much space is left for synthesis
        # These are rough estimates, actual token counts vary.
        base_prompt_elements_len = (
            len(topic) +
            len(json.dumps(research_plan)) +
            len(bibliography_prompt_list) +
            2000 # Estimate for fixed instructions text
        )
        available_chars_for_synthesis_in_report = config.MAX_CONTEXT_CHARS - base_prompt_elements_len

        if len(accumulated_synthesis_md) > available_chars_for_synthesis_in_report:
            chars_to_keep = available_chars_for_synthesis_in_report - 100 # Keep buffer for truncation message
            truncated_synthesis_md = accumulated_synthesis_md[:chars_to_keep] + "\n\n... [Synthesis truncated due to context limits for report generation]"
            logger.warning(f"Truncating synthesis markdown (from {len(accumulated_synthesis_md)} to {len(truncated_synthesis_md)} chars) for report prompt.")
            send_progress_callback(f"  -> Warning: Synthesis text truncated for final report generation due to context limits.", True, False)
        else:
            truncated_synthesis_md = accumulated_synthesis_md

        report_prompt = f"""
        Create a comprehensive research report in Markdown format on the topic: "{topic}".

        You are provided with the following inputs:
        1. The Original Research Plan:
           ```json
           {json.dumps(research_plan, indent=2)}
           ```
        2. Synthesized Information (in Markdown, potentially incomplete or truncated, with raw URL citations like `[Source URL: <full_url_here>]`):
           ```markdown
           {truncated_synthesis_md if truncated_synthesis_md.strip() else "No synthesized information was generated or provided."}
           ```
        3. A Bibliography Map (mapping URLs to citation numbers):
           ```
           {bibliography_prompt_list if bibliography_prompt_list else "No sources available for bibliography."}
           ```

        **Instructions for Generating the Final Markdown Report:**

        1.  **Structure:** Create the report with the following sections using Markdown headings:
            *   `# Research Report: {topic}` (Main Title)
            *   `## Introduction`
            *   `## Findings` (This section will contain subsections for each plan step)
            *   `## Conclusion`
            *   `## Bibliography`
        2.  **Introduction:** Briefly introduce the research topic "{topic}". State the report's purpose (to synthesize information based on the plan and sources). List or briefly describe the key steps from the Research Plan. Keep it concise (2-3 sentences).
        3.  **Findings:**
            *   Under `## Findings`, create a subsection for each step in the original Research Plan using `### Step X: <Step Description>`.
            *   For each step's subsection, carefully integrate the relevant information from the "Synthesized Information" provided above. Rephrase and structure the information clearly for readability. Ensure smooth transitions.
            *   **Crucially, perform citation replacement:** Find every instance of a raw URL citation `[Source URL: <full_url_here>]` within the Synthesized Information you are using. Replace **each complete instance** with the corresponding Markdown footnote reference `[^N]`, where `N` is the number associated with `<full_url_here>` in the provided Bibliography Map.
            *   **Handling Missing Citations:** If a URL inside a `[Source URL: ...]` tag from the synthesis is NOT found as a key in the Bibliography Map (this shouldn't happen if the map is correct, but as a fallback), **OMIT** the footnote marker `[^N]` for that specific instance. Do not invent numbers or leave partial tags. Just remove the `[Source URL: ...]` tag entirely in that case.
            *   If the "Synthesized Information" section was empty, or if no relevant synthesis was provided for a specific plan step, clearly state this under the relevant `### Step X:` heading (e.g., "No specific findings were synthesized for this step based on the available sources.").
        4.  **Conclusion:** Summarize the key findings presented in the report (or note the lack thereof if applicable). Briefly mention any limitations encountered (e.g., limited sources scraped, potential synthesis truncation, information gaps mentioned in Findings). Keep it concise (2-4 sentences). Do not introduce new information.
        5.  **Bibliography:**
            *   Under `## Bibliography`, list all the sources from the Bibliography Map.
            *   Use the standard Markdown footnote definition format: `[^N]: <full_url_here>`
            *   Each footnote definition **must** be on its own line.
            *   Ensure the numbers `N` match those used in the Findings section.
            *   If the Bibliography Map was empty, simply state "No sources were cited in this report." under the heading.
        6.  **Formatting:** Use standard Markdown (headings, lists, bold, italics, paragraphs). Ensure proper spacing. Use bullet points (`*` or `-`) where appropriate.
        7.  **Output:** Generate ONLY the complete Markdown report according to these instructions. Start directly with the `# Research Report:` title. Do not include any preliminary remarks, explanations, or text outside the defined report structure.

        Generate the Markdown report now for topic: "{topic}".
        """
        final_report_markdown = ""
        report_stream_error = None
        try:
            stream_generator = stream_gemini(report_prompt)
            for result in stream_generator:
                if result['type'] == 'chunk':
                    send_event_callback({'type': 'llm_chunk', 'content': result['content'], 'target': 'report'})
                    final_report_markdown += result['content'] # Accumulate raw markdown
                elif result['type'] == 'stream_error':
                    report_stream_error = result['message']
                    logger.error(f"LLM stream error during report generation: {report_stream_error}")
                    is_fatal_err = any(indicator in report_stream_error.lower()
                                       for indicator in ["api key", "quota", "resource has been exhausted", "permission_denied", "billing", "invalid model"])
                    send_progress_callback(f"LLM stream error during report generation: {report_stream_error}", True, is_fatal_err)
                    if is_fatal_err: return temp_files_to_clean
                    break # Break on non-fatal stream error
                elif result['type'] == 'stream_warning':
                     logger.warning(f"LLM Stream Warning (Report): {result['message']}")
                     send_progress_callback(f"LLM Stream Warning (Report): {result['message']}", True, False)
                elif result['type'] == 'stream_end':
                     logger.info(f"LLM report stream ended. Finish Reason: {result.get('finish_reason', 'N/A')}")
                     break # Normal end
        except Exception as e:
             logger.error(f"Fatal error processing LLM report stream: {e}", exc_info=True)
             send_progress_callback(f"Fatal error processing LLM report stream: {escape(str(e))}", True, True)
             return temp_files_to_clean

        send_progress_callback("Report generation finished.", False, False)
        if not final_report_markdown.strip() and not report_stream_error:
             logger.error("Final report generation resulted in empty content, but no stream error reported.")
             send_progress_callback("Error: Final report generation resulted in empty content.", True, False) # Non-fatal, show error but don't stop
             final_report_markdown = f"# Research Report: {topic}\n\n*Report generation failed or produced no content. Synthesis might have been empty or an error occurred during report formatting.*"
        elif not final_report_markdown.strip() and report_stream_error:
             logger.error(f"Final report generation resulted in empty content due to stream error: {report_stream_error}")
             # Error message already sent

        # === Step 6: Final Processing and Completion ===
        send_progress_callback("Processing final report for display...", False, False)

        # Convert final Markdown to HTML for display
        # The utility function now includes logging for conversion errors
        report_html = convert_markdown_to_html(final_report_markdown)

        # Check if conversion indicated an error or was empty
        if report_html.strip().lower().startswith(('<pre><strong>error', '<p><em>markdown conversion resulted', '<p><em>report content is empty')):
            logger.error("Failed to convert final Markdown report to HTML for display.")
            send_progress_callback("Error: Failed to convert final Markdown report to HTML. Displaying raw Markdown instead.", True, False)
            # Fallback: wrap raw markdown in pre/code tags for basic display
            report_html = f"<h2>Report Display Error</h2><p>Could not convert report Markdown to HTML. Raw Markdown content:</p><pre><code>{escape(final_report_markdown)}</code></pre>"
        elif not report_html.strip():
             logger.error("Markdown conversion resulted in empty HTML without specific error message.")
             send_progress_callback("Error: Markdown conversion resulted in empty HTML. Displaying raw Markdown instead.", True, False)
             report_html = f"<h2>Report Display Error</h2><p>Markdown conversion resulted in empty content. Raw Markdown content:</p><pre><code>{escape(final_report_markdown)}</code></pre>"


        # Send final data package to the client
        send_progress_callback("Sending final results to client...", False, False)
        final_data = {
            'type': 'complete',
            'report_html': report_html
        }
        send_event_callback(final_data)

        end_time_total = time.time()
        total_duration = end_time_total - start_time_total
        logger.info(f"Research process for '{topic}' completed successfully in {total_duration:.2f} seconds.")
        send_progress_callback(f"Research process completed successfully in {total_duration:.2f} seconds.", False, False)

        return temp_files_to_clean # Return list of files for cleanup

    except Exception as e:
        # Catch any unexpected errors in the main orchestration workflow
        logger.error(f"FATAL: Unhandled exception in research orchestrator for topic '{topic}': {e}", exc_info=True)
        # traceback.print_exc() # Log full traceback to server logs
        error_msg = f"Unexpected server error during research orchestration: {type(e).__name__} - {escape(str(e))}"
        # Try to send final fatal error message
        try:
             send_progress_callback(error_msg, True, True)
        except Exception as callback_err:
             logger.error(f"Failed to send final fatal error message to client: {callback_err}")

        # Return any temp files created *before* the fatal error occurred
        return temp_files_to_clean