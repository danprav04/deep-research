# llm_interface.py
import time
import logging
from typing import Dict, Any, Optional, Generator

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig

import config as config

# Configure logger for this module
logger = logging.getLogger(__name__)

# Note: genai.configure() is called once in app.py after loading config.

def _prepare_llm_call_args(system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Prepares common arguments for LLM calls."""
    generation_config = GenerationConfig(
        temperature=config.LLM_TEMPERATURE,
        # max_output_tokens=8192 # Usually set by default, adjust if hitting limits
        # stop_sequences=["..."] # Add stop sequences if needed
    )

    # Map string constants from config to HarmCategory/HarmBlockThreshold enums
    # This ensures compatibility even if future API versions require enums.
    try:
        safety_settings_mapped = {
            HarmCategory[k]: HarmBlockThreshold[v]
            for k, v in config.SAFETY_SETTINGS.items()
            if hasattr(HarmCategory, k) and hasattr(HarmBlockThreshold, v) # Check if keys/values are valid enums
        }
        if len(safety_settings_mapped) != len(config.SAFETY_SETTINGS):
            logger.warning("Some safety setting keys/values in config might be invalid.")
    except KeyError as e:
        logger.error(f"Invalid key/value in SAFETY_SETTINGS: {e}. Using default safety settings.", exc_info=True)
        safety_settings_mapped = {} # Fallback to default or handle as needed


    model_kwargs = {
        "generation_config": generation_config,
        "safety_settings": safety_settings_mapped,
    }
    if system_prompt and system_prompt.strip():
         # Use the dedicated system_instruction parameter
         model_kwargs["system_instruction"] = system_prompt.strip()
         logger.debug("Using system instruction for LLM call.")

    return model_kwargs

def call_gemini(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Calls the specified Google Gemini model with retry logic for non-streaming requests.

    Args:
        prompt: The main user prompt.
        system_prompt: An optional system-level instruction.

    Returns:
        The generated text content from the LLM.

    Raises:
        ValueError: If the API response is blocked or invalid after retries.
        RuntimeError: For critical API errors like invalid key or quota exhaustion after retries.
        Exception: For other unexpected errors during the API call.
    """
    model_kwargs = _prepare_llm_call_args(system_prompt)
    logger.info(f"Calling Gemini model {config.GOOGLE_MODEL_NAME} (non-streaming). Prompt length: {len(prompt)} chars.")

    last_exception = None
    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            model = genai.GenerativeModel(config.GOOGLE_MODEL_NAME, **model_kwargs)
            response = model.generate_content(prompt) # Pass only user prompt if system_instruction is used

            # --- Detailed Response Validation ---
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = str(response.prompt_feedback.block_reason)
                 safety_ratings_str = str(getattr(response.prompt_feedback, 'safety_ratings', 'N/A'))
                 raise ValueError(f"API response blocked due to prompt feedback. Reason: {block_reason}. Safety Ratings: {safety_ratings_str}")

            if not response.candidates:
                 finish_reason = getattr(response, 'finish_reason', 'UNKNOWN') # More robust access
                 usage_metadata = getattr(response, 'usage_metadata', None)
                 raise ValueError(f"API response contained no candidates. Finish Reason: {finish_reason}. Usage: {usage_metadata}")

            # Check the first candidate (usually the only one for basic calls)
            candidate = response.candidates[0]
            candidate_finish_reason = getattr(candidate, 'finish_reason', None) # Use getattr for safety
            # Define acceptable finish reasons (adjust based on API version/behavior)
            # 1: STOP, 2: MAX_TOKENS, 0: UNSPECIFIED (can sometimes be ok), None: OK
            acceptable_reasons = {1, 2, 0, None, "STOP", "MAX_TOKENS", "FINISH_REASON_UNSPECIFIED"}
            if candidate_finish_reason not in acceptable_reasons:
                 safety_ratings_str = str(getattr(candidate, 'safety_ratings', 'N/A'))
                 raise ValueError(f"API candidate finished unexpectedly. Reason: {candidate_finish_reason}. Safety Ratings: {safety_ratings_str}")

            if not candidate.content or not candidate.content.parts:
                # Check if finish reason suggests it's intentionally empty (e.g., safety block DURING generation)
                 if candidate_finish_reason == 2: # FinishReason.SAFETY == 2
                    raise ValueError(f"API candidate content empty, blocked by safety settings during generation. Finish Reason: {candidate_finish_reason}")
                 else:
                    logger.warning(f"API response candidate has no content parts (Finish Reason: {candidate_finish_reason}). Returning empty string.")
                    return "" # Return empty if no content but not explicitly blocked

            # Safely extract text
            response_content = getattr(response, 'text', '').strip()
            if not response_content and candidate_finish_reason not in [2]: # If empty but not due to safety block
                 logger.warning(f"LLM generated empty text content (Finish Reason: {candidate_finish_reason}).")
                 # Decide whether to return "" or raise an error based on expected behavior

            logger.info(f"Gemini call successful. Response length: {len(response_content)} chars.")
            return response_content

        except Exception as e:
            last_exception = e
            logger.warning(f"Error calling Google Gemini API (Attempt {attempt + 1}/{config.LLM_MAX_RETRIES}): {e}")

            error_str = str(e).lower()
            if "api key not valid" in error_str or "permission_denied" in error_str:
                 logger.error("Critical Error: Invalid Google API Key or insufficient permissions.")
                 raise RuntimeError(f"Invalid Google API Key or permissions: {e}") from e # Don't retry
            if "quota" in error_str or "resource has been exhausted" in error_str:
                 logger.error("Quota possibly exceeded.")
                 # Depending on policy, might raise immediately or retry with longer backoff
                 # For now, we continue with the retry loop using standard delay.
            if "prompt" in error_str and ("too long" in error_str or "size" in error_str):
                 logger.error(f"Prompt likely too long for model {config.GOOGLE_MODEL_NAME}.")
                 raise ValueError(f"Prompt too long for model: {e}") from e # Don't retry size issues

            if attempt < config.LLM_MAX_RETRIES - 1:
                logger.info(f"Retrying in {config.LLM_RETRY_DELAY} seconds...")
                time.sleep(config.LLM_RETRY_DELAY)
            else:
                logger.error("Max retries reached for Gemini API call.")
                # Raise the specific error encountered on the last attempt
                if isinstance(last_exception, ValueError):
                    raise last_exception
                else:
                    raise RuntimeError(f"LLM call failed after {config.LLM_MAX_RETRIES} retries: {last_exception}") from last_exception

    # Should not be reached if logic is correct, but acts as a fallback.
    logger.error("LLM call function exited loop unexpectedly.")
    raise RuntimeError(f"LLM call failed after retries. Last known error: {last_exception}")


def stream_gemini(prompt: str, system_prompt: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
    """
    Calls the Google Gemini model with streaming enabled.

    Yields dictionaries representing stream events:
    - {'type': 'chunk', 'content': str} for text chunks.
    - {'type': 'stream_error', 'message': str} for errors during streaming.
    - {'type': 'stream_warning', 'message': str} for non-fatal issues (e.g., potential blocking).
    - {'type': 'stream_end', 'finish_reason': str} when the stream finishes normally.

    Args:
        prompt: The main user prompt.
        system_prompt: An optional system-level instruction.

    Raises:
        Catches exceptions during stream setup/iteration and yields 'stream_error'.
    """
    model_kwargs = _prepare_llm_call_args(system_prompt)
    logger.info(f"Calling Gemini model {config.GOOGLE_MODEL_NAME} (streaming). Prompt length: {len(prompt)} chars.")

    try:
        model = genai.GenerativeModel(config.GOOGLE_MODEL_NAME, **model_kwargs)
        stream = model.generate_content(prompt, stream=True) # Pass only user prompt if system_instruction is used

        stream_interrupted_by_error = False
        final_finish_reason = "Unknown"

        for chunk in stream:
             # 1. Check for Prompt Feedback (can block before generation starts)
             if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                  block_reason_str = str(chunk.prompt_feedback.block_reason)
                  safety_ratings_str = str(getattr(chunk.prompt_feedback, 'safety_ratings', 'N/A'))
                  error_msg = f'LLM prompt blocked by safety filters (Reason: {block_reason_str}). Cannot generate response. Ratings: {safety_ratings_str}'
                  logger.error(error_msg)
                  yield {'type': 'stream_error', 'message': error_msg}
                  stream_interrupted_by_error = True
                  break # Stop processing if prompt is blocked

             # 2. Check Candidate Information (blocking/finish reasons during generation)
             if not stream_interrupted_by_error and chunk.candidates:
                 candidate = chunk.candidates[0]
                 candidate_finish_reason = getattr(candidate, 'finish_reason', None)
                 final_finish_reason = candidate_finish_reason # Track the latest finish reason

                 # Define acceptable finish reasons during streaming (includes ongoing generation)
                 # 0: Unspecified, 1: Stop, 2: Max Tokens, None: Still generating
                 acceptable_stream_reasons = {0, 1, 2, None, "STOP", "MAX_TOKENS", "FINISH_REASON_UNSPECIFIED"}

                 if candidate_finish_reason not in acceptable_stream_reasons:
                     # Handle unexpected stops (like SAFETY=3)
                     finish_reason_str = str(candidate_finish_reason)
                     safety_ratings_str = str(getattr(candidate, 'safety_ratings', 'N/A'))
                     warning_msg = f'LLM stream may have been interrupted or blocked during generation (Reason: {finish_reason_str}). Output may be incomplete. Safety: {safety_ratings_str}'
                     logger.warning(warning_msg)
                     yield {'type': 'stream_warning', 'message': warning_msg}
                     # If blocked due to safety (e.g., reason code 3), treat as error and stop.
                     # Adjust the check based on the specific value for 'SAFETY' if needed.
                     if finish_reason_str == "SAFETY" or candidate_finish_reason == 3: # FinishReason.SAFETY == 3 in some versions?
                         error_msg = f'LLM stream stopped due to safety filter during generation (Reason: {finish_reason_str}).'
                         logger.error(error_msg)
                         yield {'type': 'stream_error', 'message': error_msg}
                         stream_interrupted_by_error = True
                         # Don't 'break' here immediately, let the loop finish to potentially get final metadata if API provides it.

             # 3. Yield Content Chunk if available and stream not errored
             # Check hasattr for robustness, some chunks might be metadata-only
             if not stream_interrupted_by_error and hasattr(chunk, 'text') and chunk.text:
                  yield {'type': 'chunk', 'content': chunk.text}

        # 4. Signal Normal Stream End (if no fatal error occurred)
        if not stream_interrupted_by_error:
             logger.info(f"Gemini stream finished. Final finish reason: {final_finish_reason}")
             yield {'type': 'stream_end', 'finish_reason': str(final_finish_reason)}

    except Exception as e:
        logger.error(f"Error during Google Gemini stream setup or iteration: {e}", exc_info=True)
        error_message = f"LLM stream error: {e}"
        error_str = str(e).lower()

        # Provide more specific error messages based on common issues
        if "api key not valid" in error_str or "permission_denied" in error_str:
            error_message = f"LLM stream error: Invalid Google API Key or insufficient permissions. ({e})"
        elif "quota" in error_str or "resource has been exhausted" in error_str:
            error_message = f"LLM stream error: Quota likely exceeded or resource exhausted. ({e})"
        elif "prompt" in error_str and ("too long" in error_str or "size" in error_str):
             error_message = f"LLM stream error: Prompt likely too long for the model's context window. ({e})"
        elif "model" in error_str and ("not found" in error_str or "invalid" in error_str):
             error_message = f"LLM stream error: Invalid model name '{config.GOOGLE_MODEL_NAME}' specified. ({e})"
        # Add more specific checks as needed

        yield {'type': 'stream_error', 'message': error_message}