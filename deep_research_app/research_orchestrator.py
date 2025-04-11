# research_orchestrator.py
import time
import json
import os
import traceback
import logging
from typing import Dict, Any, List, Callable, Optional, Tuple, Generator
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

# Define yielded event structures for clarity (optional but good practice)
ProgressEvent = Dict[str, Any] # keys: type='progress', message, is_error, is_fatal
DataEvent = Dict[str, Any]     # keys: type='event', data={...}
ScrapeSuccessEvent = Dict[str, Any] # keys: type='scrape_success', metadata={...}


def run_research_process(topic: str) -> Generator[Dict[str, Any], None, None]:
    """
    Executes the entire research process as a generator, yielding events.

    Yields dictionaries representing different stages and data:
    - {'type': 'progress', 'message': str, 'is_error': bool, 'is_fatal': bool}
    - {'type': 'event', 'data': Dict[str, Any]} (e.g., for llm_chunk, stream_start)
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

    try: # <<< START OF MAIN TRY BLOCK >>>
        logger.info(f"Starting research process for topic: '{topic}'")

        # === Step 1: Generate Research Plan ===
        yield {'type': 'progress', 'message': f"Generating research plan for: '{topic}'...", 'is_error': False, 'is_fatal': False}
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
            research_plan = parse_research_plan(plan_response)
            if not research_plan or (len(research_plan) == 1 and research_plan[0]["step"].startswith("Failed")):
                 fail_reason = research_plan[0]["step"] if research_plan else "Could not parse plan structure from LLM."
                 raw_snippet = f" Raw LLM Response Snippet: '{plan_response[:150]}...'" if plan_response else " (LLM Response was empty)"
                 logger.error(f"Failed to create/parse research plan. Reason: {fail_reason}.{raw_snippet}")
                 yield {'type': 'progress', 'message': f"Fatal Error: Failed to create or parse research plan. Reason: {fail_reason}.", 'is_error': True, 'is_fatal': True}
                 fatal_error_occurred = True
                 return # Stop generation
        except Exception as e:
             logger.error(f"LLM Error generating research plan: {e}", exc_info=True)
             yield {'type': 'progress', 'message': f"Fatal Error: Failed during research plan generation: {e}", 'is_error': True, 'is_fatal': True}
             fatal_error_occurred = True
             return # Stop generation

        yield {'type': 'progress', 'message': f"Generated {len(research_plan)} step plan.", 'is_error': False, 'is_fatal': False}
        for i, step in enumerate(research_plan):
             step_desc = step.get('step', 'Unnamed Step')
             step_keywords = step.get('keywords', [])
             yield {'type': 'progress', 'message': f"  Step {i+1}: {step_desc} (Keywords: {step_keywords})", 'is_error': False, 'is_fatal': False}

        # === Step 2a: Search and Collect URLs ===
        yield {'type': 'progress', 'message': "Starting web search...", 'is_error': False, 'is_fatal': False}
        start_search_time = time.time()
        all_urls_from_search_step = set()
        total_search_errors = 0
        total_search_queries = 0

        for i, step in enumerate(research_plan):
            step_desc = step.get('step', f'Unnamed Step {i+1}')
            keywords = step.get('keywords', [])
            progress_msg = f"Searching - Step {i+1}/{len(research_plan)}: '{step_desc[:70]}{'...' if len(step_desc)>70 else ''}'"
            yield {'type': 'progress', 'message': progress_msg, 'is_error': False, 'is_fatal': False}

            if not keywords:
                yield {'type': 'progress', 'message': "  -> No keywords provided for this step, skipping search.", 'is_error': False, 'is_fatal': False}
                continue

            total_search_queries += 1
            step_urls, step_errors = perform_web_search(keywords)

            if step_errors:
                total_search_errors += len(step_errors)
                for err in step_errors:
                    yield {'type': 'progress', 'message': f"    -> Search Warning: {err}", 'is_error': True, 'is_fatal': False}

            new_urls_count = len(set(step_urls) - all_urls_from_search_step)
            all_urls_from_search_step.update(step_urls)
            yield {'type': 'progress', 'message': f"  -> Found {len(step_urls)} URLs for step keywords, {new_urls_count} new. Total unique URLs so far: {len(all_urls_from_search_step)}.", 'is_error': False, 'is_fatal': False}

            if i < len(research_plan) - 1:
                 time.sleep(config.INTER_SEARCH_DELAY_SECONDS)

        search_duration = time.time() - start_search_time
        yield {'type': 'progress', 'message': f"Search phase completed in {search_duration:.2f}s.", 'is_error': False, 'is_fatal': False}
        yield {'type': 'progress', 'message': f"Collected {len(all_urls_from_search_step)} total unique URLs ({total_search_errors} search errors).", 'is_error': False, 'is_fatal': False}

        # --- Filter URLs ---
        urls_to_scrape_list = []
        yield {'type': 'progress', 'message': "Filtering URLs for scraping...", 'is_error': False, 'is_fatal': False}
        for url in sorted(list(all_urls_from_search_step)):
             if len(urls_to_scrape_list) >= config.MAX_TOTAL_URLS_TO_SCRAPE:
                  yield {'type': 'progress', 'message': f"  -> Reached URL scraping limit ({config.MAX_TOTAL_URLS_TO_SCRAPE}). Skipping remaining URLs.", 'is_error': False, 'is_fatal': False}
                  break

             lower_url = url.lower()
             path_part = lower_url.split('?')[0].split('#')[0]
             is_file_extension = path_part.endswith(('.pdf', '.jpg', '.png', '.gif', '.zip', '.mp4', '.mp3', '.docx', '.xlsx', '.pptx', '.webp', '.svg', '.xml', '.css', '.js', '.jpeg', '.doc', '.xls', '.ppt', '.txt', '.exe', '.dmg', '.iso', '.rar', '.gz', '.tar', '.bz2', '.7z'))
             is_unwanted_scheme = lower_url.startswith(('mailto:', 'javascript:', 'ftp:', 'tel:', 'file:', 'data:'))
             is_local = lower_url.startswith(('localhost', '127.0.0.1'))
             is_valid_http = url.startswith(('http://', 'https://'))

             if is_valid_http and not any([is_file_extension, is_unwanted_scheme, is_local]):
                  urls_to_scrape_list.append(url)

        yield {'type': 'progress', 'message': f"Selected {len(urls_to_scrape_list)} URLs for scraping after filtering (limit was {config.MAX_TOTAL_URLS_TO_SCRAPE}).", 'is_error': False, 'is_fatal': False}

        if not urls_to_scrape_list:
             logger.error("No suitable URLs found to scrape after search and filtering.")
             yield {'type': 'progress', 'message': "Fatal Error: No suitable URLs found to scrape. Cannot proceed.", 'is_error': True, 'is_fatal': True}
             fatal_error_occurred = True
             return # Stop generation

        # === Step 2b: Scrape URLs Concurrently ===
        yield {'type': 'progress', 'message': f"Starting concurrent scraping ({config.MAX_WORKERS} workers)...", 'is_error': False, 'is_fatal': False}
        start_scrape_time = time.time()
        scraped_source_metadata_list = [] # Reset before scraping
        processed_scrape_count = 0
        successful_scrape_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            future_to_url = {executor.submit(scrape_url, url): url for url in urls_to_scrape_list}

            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                processed_scrape_count += 1
                try:
                    result_dict = future.result()
                    if result_dict and 'temp_filepath' in result_dict and result_dict['temp_filepath']:
                        scraped_source_metadata_list.append(result_dict)
                        successful_scrape_count += 1
                        yield {'type': 'scrape_success', 'metadata': result_dict}
                except Exception as exc:
                    logger.error(f"Error during scraping task for {url[:70]}...: {exc}", exc_info=False)
                    yield {'type': 'progress', 'message': f"    -> Scrape Error for {url[:60]}...: {escape(str(exc))}", 'is_error': True, 'is_fatal': False}

                if processed_scrape_count % 5 == 0 or processed_scrape_count == len(urls_to_scrape_list):
                      progress_perc = (processed_scrape_count * 100) // len(urls_to_scrape_list)
                      yield {'type': 'progress', 'message': f"  -> Scraping Progress: {processed_scrape_count}/{len(urls_to_scrape_list)} URLs processed ({progress_perc}%). Successful scrapes: {successful_scrape_count}", 'is_error': False, 'is_fatal': False}

        scrape_duration = time.time() - start_scrape_time
        yield {'type': 'progress', 'message': f"Scraping finished in {scrape_duration:.2f}s. Successfully scraped content from {successful_scrape_count} URLs.", 'is_error': False, 'is_fatal': False}

        if not scraped_source_metadata_list:
            logger.error("Failed to scrape any content successfully.")
            yield {'type': 'progress', 'message': "Fatal Error: Failed to scrape any content successfully. Cannot proceed with synthesis.", 'is_error': True, 'is_fatal': True}
            fatal_error_occurred = True
            return # Stop generation

        scraped_url_map = {item['url']: item for item in scraped_source_metadata_list}
        ordered_scraped_metadata_list = [scraped_url_map[url] for url in urls_to_scrape_list if url in scraped_url_map]
        scraped_source_metadata_list = ordered_scraped_metadata_list

        # === Step 3: Generate Bibliography Map ===
        url_to_index_map, bibliography_prompt_list = generate_bibliography_map(scraped_source_metadata_list)
        yield {'type': 'progress', 'message': f"Generated bibliography map for {len(url_to_index_map)} successfully scraped sources.", 'is_error': False, 'is_fatal': False}

        # === Step 4: Synthesize Information (Streaming, RAM Optimized) ===
        yield {'type': 'progress', 'message': f"Synthesizing information from {len(scraped_source_metadata_list)} scraped sources using {config.GOOGLE_MODEL_NAME}...", 'is_error': False, 'is_fatal': False}
        yield {'type': 'event', 'data': {'type': 'stream_start', 'target': 'synthesis'}}

        context_for_llm_parts = []
        current_total_chars = 0
        sources_included_count = 0

        yield {'type': 'progress', 'message': f"  -> Preparing context for synthesis (limit ~{config.MAX_CONTEXT_CHARS // 1000}k chars)...", 'is_error': False, 'is_fatal': False}

        for source_metadata in scraped_source_metadata_list:
            filepath = source_metadata.get('temp_filepath')
            url = source_metadata.get('url')
            if not filepath or not url or not os.path.exists(filepath):
                 logger.warning(f"Skipping source for synthesis, missing temp file or metadata. URL: {url or 'Unknown'}, Path: {filepath}")
                 yield {'type': 'progress', 'message': f"  -> Warning: Skipping source, missing temp file or metadata for URL {url or 'Unknown'}", 'is_error': True, 'is_fatal': False}
                 continue

            try:
                 file_size = os.path.getsize(filepath)
                 if (current_total_chars + file_size) <= config.MAX_CONTEXT_CHARS:
                     try:
                         with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                             content = f.read()
                         if content.strip():
                              context_for_llm_parts.append({'url': url, 'content': content})
                              current_total_chars += len(content)
                              sources_included_count += 1
                         else:
                              logger.warning(f"Read empty or whitespace-only content from temp file {os.path.basename(filepath)} for {url[:60]}...")
                              yield {'type': 'progress', 'message': f"  -> Warning: Read empty content from temp file for {url[:60]}...", 'is_error': True, 'is_fatal': False}
                     except Exception as read_err:
                         logger.error(f"Error reading temp file {os.path.basename(filepath)} for {url[:60]}: {read_err}", exc_info=False)
                         yield {'type': 'progress', 'message': f"  -> Error reading temp file for {url[:60]}...: {read_err}", 'is_error': True, 'is_fatal': False}
                 else:
                      logger.warning(f"Context limit ({config.MAX_CONTEXT_CHARS // 1000}k chars) reached. Stopping context build. Included {sources_included_count} sources.")
                      yield {'type': 'progress', 'message': f"  -> Context limit reached. Synthesizing based on first {sources_included_count}/{len(scraped_source_metadata_list)} sources (~{current_total_chars // 1000}k chars).", 'is_error': True, 'is_fatal': False}
                      break

            except OSError as e:
                 logger.error(f"Error accessing temp file {os.path.basename(filepath)} for {url[:60]}...: {e}", exc_info=False)
                 yield {'type': 'progress', 'message': f"  -> Error accessing temp file metadata for {url[:60]}...: {e}", 'is_error': True, 'is_fatal': False}
            except Exception as e:
                 logger.error(f"Unexpected error processing temp file {os.path.basename(filepath)} for {url[:60]}...: {e}", exc_info=False)
                 yield {'type': 'progress', 'message': f"  -> Unexpected error processing temp file for {url[:60]}...: {e}", 'is_error': True, 'is_fatal': False}

        if sources_included_count == 0:
             logger.error("No source content could be prepared for synthesis.")
             yield {'type': 'progress', 'message': "Fatal Error: No source content available for synthesis. Cannot proceed.", 'is_error': True, 'is_fatal': True}
             fatal_error_occurred = True
             return # Stop generation

        estimated_tokens = current_total_chars / config.CHARS_PER_TOKEN_ESTIMATE
        yield {'type': 'progress', 'message': f"  -> Prepared synthesis context using {sources_included_count} sources (~{current_total_chars // 1000}k chars / ~{estimated_tokens / 1000:.1f}k est. tokens).", 'is_error': False, 'is_fatal': False}

        try:
            context_json_str = json.dumps(context_for_llm_parts, indent=2, ensure_ascii=False)
            del context_for_llm_parts
        except Exception as json_err:
            logger.error(f"Failed to serialize context parts to JSON: {json_err}", exc_info=True)
            yield {'type': 'progress', 'message': f"Fatal Error: Could not prepare context data for LLM: {json_err}", 'is_error': True, 'is_fatal': True}
            fatal_error_occurred = True
            return

        synthesis_prompt = f"""
        Analyze the provided web content about "{topic}" based on the research plan.
        Synthesize key information relevant to each plan step, citing sources accurately using ONLY the provided URLs.

        Research Topic: {topic}
        Research Plan:
        ```json
        {json.dumps(research_plan, indent=2)}
        ```
        Source Content ({sources_included_count} sources):
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
        del context_json_str

        accumulated_synthesis_md = ""
        synthesis_stream_error = None
        try:
            stream_generator = stream_gemini(synthesis_prompt)
            for result in stream_generator:
                if result['type'] == 'chunk':
                    yield {'type': 'event', 'data': {'type': 'llm_chunk', 'content': result['content'], 'target': 'synthesis'}}
                    accumulated_synthesis_md += result['content']
                elif result['type'] == 'stream_error':
                    synthesis_stream_error = result['message']
                    logger.error(f"LLM stream error during synthesis: {synthesis_stream_error}")
                    is_fatal_err = any(indicator in synthesis_stream_error.lower()
                                       for indicator in ["api key", "quota", "resource has been exhausted", "permission_denied", "billing", "invalid model"])
                    yield {'type': 'progress', 'message': f"LLM stream error during synthesis: {synthesis_stream_error}", 'is_error': True, 'is_fatal': is_fatal_err}
                    if is_fatal_err:
                        fatal_error_occurred = True
                        return # Stop generation
                    break # Break on non-fatal stream error
                elif result['type'] == 'stream_warning':
                     logger.warning(f"LLM Stream Warning (Synthesis): {result['message']}")
                     yield {'type': 'progress', 'message': f"LLM Stream Warning (Synthesis): {result['message']}", 'is_error': True, 'is_fatal': False}
                elif result['type'] == 'stream_end':
                     logger.info(f"LLM synthesis stream ended. Finish Reason: {result.get('finish_reason', 'N/A')}")
                     break # Normal end
        except Exception as e:
             logger.error(f"Fatal error processing LLM synthesis stream: {e}", exc_info=True)
             yield {'type': 'progress', 'message': f"Fatal error processing LLM synthesis stream: {escape(str(e))}", 'is_error': True, 'is_fatal': True}
             fatal_error_occurred = True
             return

        yield {'type': 'progress', 'message': "Synthesis generation finished.", 'is_error': False, 'is_fatal': False}
        if not accumulated_synthesis_md.strip() and not synthesis_stream_error:
             logger.warning("Synthesis resulted in empty content, but no stream error reported.")
             yield {'type': 'progress', 'message': "Warning: Synthesis resulted in empty content. Final report may lack detail.", 'is_error': True, 'is_fatal': False}
        elif not accumulated_synthesis_md.strip() and synthesis_stream_error:
             logger.error(f"Synthesis resulted in empty content due to stream error: {synthesis_stream_error}")

        # === Step 5: Generate Final Report (Streaming) ===
        yield {'type': 'progress', 'message': f"Generating final report using {config.GOOGLE_MODEL_NAME}...", 'is_error': False, 'is_fatal': False}
        yield {'type': 'event', 'data': {'type': 'stream_start', 'target': 'report'}}

        base_prompt_elements_len = (
            len(topic) + len(json.dumps(research_plan)) + len(bibliography_prompt_list) + 2000
        )
        available_chars_for_synthesis_in_report = config.MAX_CONTEXT_CHARS - base_prompt_elements_len

        if len(accumulated_synthesis_md) > available_chars_for_synthesis_in_report:
            chars_to_keep = available_chars_for_synthesis_in_report - 100
            truncated_synthesis_md = accumulated_synthesis_md[:chars_to_keep] + "\n\n... [Synthesis truncated due to context limits for report generation]"
            logger.warning(f"Truncating synthesis markdown (from {len(accumulated_synthesis_md)} to {len(truncated_synthesis_md)} chars) for report prompt.")
            yield {'type': 'progress', 'message': "  -> Warning: Synthesis text truncated for final report generation due to context limits.", 'is_error': True, 'is_fatal': False}
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
           ```           {bibliography_prompt_list if bibliography_prompt_list else "No sources available for bibliography."}
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
        del accumulated_synthesis_md
        del truncated_synthesis_md

        final_report_markdown = ""
        report_stream_error = None
        try:
            stream_generator = stream_gemini(report_prompt)
            for result in stream_generator:
                if result['type'] == 'chunk':
                    yield {'type': 'event', 'data': {'type': 'llm_chunk', 'content': result['content'], 'target': 'report'}}
                    final_report_markdown += result['content']
                elif result['type'] == 'stream_error':
                    report_stream_error = result['message']
                    logger.error(f"LLM stream error during report generation: {report_stream_error}")
                    is_fatal_err = any(indicator in report_stream_error.lower()
                                       for indicator in ["api key", "quota", "resource has been exhausted", "permission_denied", "billing", "invalid model"])
                    yield {'type': 'progress', 'message': f"LLM stream error during report generation: {report_stream_error}", 'is_error': True, 'is_fatal': is_fatal_err}
                    if is_fatal_err:
                        fatal_error_occurred = True
                        return # Stop generation
                    break # Break on non-fatal stream error
                elif result['type'] == 'stream_warning':
                     logger.warning(f"LLM Stream Warning (Report): {result['message']}")
                     yield {'type': 'progress', 'message': f"LLM Stream Warning (Report): {result['message']}", 'is_error': True, 'is_fatal': False}
                elif result['type'] == 'stream_end':
                     logger.info(f"LLM report stream ended. Finish Reason: {result.get('finish_reason', 'N/A')}")
                     break # Normal end
        except Exception as e:
             logger.error(f"Fatal error processing LLM report stream: {e}", exc_info=True)
             yield {'type': 'progress', 'message': f"Fatal error processing LLM report stream: {escape(str(e))}", 'is_error': True, 'is_fatal': True}
             fatal_error_occurred = True
             return

        yield {'type': 'progress', 'message': "Report generation finished.", 'is_error': False, 'is_fatal': False}
        if not final_report_markdown.strip() and not report_stream_error:
             logger.error("Final report generation resulted in empty content, but no stream error reported.")
             yield {'type': 'progress', 'message': "Error: Final report generation resulted in empty content.", 'is_error': True, 'is_fatal': False}
             final_report_markdown = f"# Research Report: {topic}\n\n*Report generation failed or produced no content.*"
        elif not final_report_markdown.strip() and report_stream_error:
             logger.error(f"Final report generation resulted in empty content due to stream error: {report_stream_error}")

        # === Step 6: Final Processing and Completion ===
        yield {'type': 'progress', 'message': "Processing final report for display...", 'is_error': False, 'is_fatal': False}

        # --- <<< ADD SERVER-SIDE FIX: Strip potential Markdown code fences >>> ---
        cleaned_report_markdown = final_report_markdown.strip() # Initial strip for easier checking
        fence_start_md = "```markdown\n"
        fence_start_simple = "```\n"
        fence_end = "\n```"
        fence_end_simple = "```"

        was_stripped = False
        if cleaned_report_markdown.startswith(fence_start_md) and cleaned_report_markdown.endswith(fence_end):
            cleaned_report_markdown = cleaned_report_markdown[len(fence_start_md):-len(fence_end)].strip()
            was_stripped = True
        elif cleaned_report_markdown.startswith(fence_start_simple) and cleaned_report_markdown.endswith(fence_end):
            cleaned_report_markdown = cleaned_report_markdown[len(fence_start_simple):-len(fence_end)].strip()
            was_stripped = True
        elif cleaned_report_markdown.startswith(fence_end_simple) and cleaned_report_markdown.endswith(fence_end_simple): # Handles ``` only case
             cleaned_report_markdown = cleaned_report_markdown[len(fence_end_simple):-len(fence_end_simple)].strip()
             was_stripped = True

        if was_stripped:
             logger.info("Stripped surrounding Markdown code fences from the final report content before sending.")
             final_report_markdown = cleaned_report_markdown # Use the cleaned version
        # --- <<< END SERVER-SIDE FIX >>> ---

        # Convert the (potentially cleaned) Markdown to HTML
        report_html = convert_markdown_to_html(final_report_markdown)

        if report_html.strip().lower().startswith(('<pre><strong>error', '<p><em>markdown conversion resulted', '<p><em>report content is empty')):
            logger.error("Failed to convert final Markdown report to HTML for display.")
            yield {'type': 'progress', 'message': "Error converting final report Markdown to HTML. Displaying raw Markdown.", 'is_error': True, 'is_fatal': False}
            # Send the raw (but cleaned) markdown if HTML conversion fails
            report_html = f"<h2>Report Display Error</h2><p>Could not convert report Markdown to HTML. Raw Markdown content:</p><pre><code>{escape(final_report_markdown)}</code></pre>"
        elif not report_html.strip():
             logger.error("Markdown conversion resulted in empty HTML without specific error message.")
             yield {'type': 'progress', 'message': "Error: Markdown conversion resulted in empty HTML.", 'is_error': True, 'is_fatal': False}
             # Send the raw (but cleaned) markdown if HTML conversion results in nothing
             report_html = f"<h2>Report Display Error</h2><p>Markdown conversion resulted in empty content. Raw Markdown content:</p><pre><code>{escape(final_report_markdown)}</code></pre>"

        # Yield final data package to the client
        yield {'type': 'progress', 'message': "Sending final results to client...", 'is_error': False, 'is_fatal': False}
        final_data = {
            'type': 'complete',
            'report_html': report_html
            # Note: Consider sending raw 'report_markdown': final_report_markdown as well
            # if you want the client to *always* have the raw source, regardless of HTML conversion.
        }
        yield {'type': 'event', 'data': final_data}

        end_time_total = time.time()
        total_duration = end_time_total - start_time_total
        logger.info(f"Research process for '{topic}' completed successfully in {total_duration:.2f} seconds.")
        yield {'type': 'progress', 'message': f"Research process completed successfully in {total_duration:.2f} seconds.", 'is_error': False, 'is_fatal': False}

    except Exception as e:
        logger.error(f"FATAL: Unhandled exception in research orchestrator for topic '{topic}': {e}", exc_info=True)
        error_msg = f"Unexpected server error during research orchestration: {type(e).__name__} - {escape(str(e))}"
        if not fatal_error_occurred:
            try:
                 yield {'type': 'progress', 'message': error_msg, 'is_error': True, 'is_fatal': True}
            except Exception as callback_err:
                 logger.error(f"Failed to yield final fatal error message: {callback_err}")
    finally:
        if not fatal_error_occurred:
            logger.info("Orchestrator generator finished.")
        else:
            logger.error("Orchestrator generator stopped due to fatal error.")
        # Cleanup is handled by the caller (app.py)