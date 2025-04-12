# llm_interface.py
import time
import logging
from typing import Dict, Any, Optional, Generator

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
from google.api_core import exceptions as google_api_exceptions # For specific exceptions

import config as config

# Configure logger for this module
logger = logging.getLogger(__name__)

# Note: genai.configure() is called once in app.py after loading config.

def _prepare_llm_call_args(system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Prepares common arguments for LLM calls, including safety settings."""
    generation_config = GenerationConfig(
        temperature=config.LLM_TEMPERATURE,
        # Consider adding max_output_tokens if needed, but often default is fine
        # max_output_tokens=8192
    )

    # Map string constants from config to HarmCategory/HarmBlockThreshold enums
    safety_settings_mapped = {}
    try:
        safety_settings_mapped = {
            HarmCategory[k]: HarmBlockThreshold[v]
            for k, v in config.SAFETY_SETTINGS.items()
            if hasattr(HarmCategory, k) and hasattr(HarmBlockThreshold, v)
        }
        if len(safety_settings_mapped) != len(config.SAFETY_SETTINGS):
            invalid_settings = {k:v for k,v in config.SAFETY_SETTINGS.items() if not (hasattr(HarmCategory, k) and hasattr(HarmBlockThreshold, v))}
            logger.warning(f"Some safety setting keys/values in config are invalid or unsupported: {invalid_settings}. Using defaults where possible.")
    except KeyError as e:
        logger.error(f"Invalid key/value in SAFETY_SETTINGS: {e}. Using default safety settings.", exc_info=True)
        safety_settings_mapped = {} # Fallback to default

    model_kwargs = {
        "generation_config": generation_config,
        "safety_settings": safety_settings_mapped,
    }
    if system_prompt and system_prompt.strip():
         model_kwargs["system_instruction"] = system_prompt.strip()
         logger.debug("Using system instruction for LLM call.")
    else:
        # Explicitly set to None if empty or not provided, Gemini API might require this.
        model_kwargs["system_instruction"] = None


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
        ValueError: If the API response is blocked, invalid, or the prompt is too long after retries.
        RuntimeError: For critical API errors (invalid key, quota) or if retries are exhausted.
        Exception: For other unexpected errors during the API call.
    """
    model_kwargs = _prepare_llm_call_args(system_prompt)
    prompt_len = len(prompt) # Calculate length once
    logger.info(f"Calling Gemini model {config.GOOGLE_MODEL_NAME} (non-streaming). Prompt length: ~{prompt_len // 1000}k chars.")
    logger.debug(f"Prompt starts with: {prompt[:200]}...") # Log beginning of prompt for debugging

    last_exception = None
    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            start_time = time.time()
            model = genai.GenerativeModel(config.GOOGLE_MODEL_NAME, **model_kwargs)
            # Pass prompt correctly based on whether system_instruction is used
            if model_kwargs.get("system_instruction"):
                 response = model.generate_content(prompt)
            else:
                 # Older way or if system_instruction isn't preferred/supported by model version
                 response = model.generate_content([system_prompt or "", prompt]) # Pass as list if no system_instruction

            duration = time.time() - start_time
            logger.info(f"Gemini non-stream call attempt {attempt + 1} completed in {duration:.2f}s.")

            # --- Detailed Response Validation ---
            # 1. Check Prompt Feedback for blocking *before* generation
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = str(response.prompt_feedback.block_reason)
                 safety_ratings_str = str(getattr(response.prompt_feedback, 'safety_ratings', 'N/A'))
                 err_msg = f"API response blocked due to prompt feedback. Reason: {block_reason}. Safety Ratings: {safety_ratings_str}"
                 logger.error(err_msg)
                 # Treat prompt blocking as a non-retryable ValueError for this specific prompt
                 raise ValueError(err_msg)

            # 2. Check Candidates exist
            if not response.candidates:
                 # Try to get more info if available
                 finish_reason = getattr(response, 'finish_reason', 'UNKNOWN')
                 usage_metadata = getattr(response, 'usage_metadata', None)
                 err_msg = f"API response contained no candidates. Finish Reason: {finish_reason}. Usage: {usage_metadata}"
                 logger.error(err_msg)
                 # This might be retryable depending on the reason, but let's treat as ValueError for now
                 raise ValueError(err_msg)

            # 3. Check the first candidate (usually the only one)
            candidate = response.candidates[0]
            candidate_finish_reason = getattr(candidate, 'finish_reason', None)
            acceptable_reasons = {1, 2, 0, None, "STOP", "MAX_TOKENS", "FINISH_REASON_UNSPECIFIED"} # 1:STOP, 2:MAX_TOKENS, 0:UNSPECIFIED

            # 4. Check for blocking *during* generation (finish reason)
            if candidate_finish_reason not in acceptable_reasons:
                 safety_ratings_str = str(getattr(candidate, 'safety_ratings', 'N/A'))
                 err_msg = f"API candidate finished unexpectedly or was blocked during generation. Reason: {candidate_finish_reason}. Safety Ratings: {safety_ratings_str}"
                 logger.error(err_msg)
                 # Treat safety blocks during generation as non-retryable for this prompt
                 # Check common safety codes/strings (adjust if API changes)
                 if candidate_finish_reason == 3 or str(candidate_finish_reason) == "SAFETY":
                     raise ValueError(err_msg)
                 else:
                     # Other unexpected reasons might be retryable
                     raise RuntimeError(err_msg) # Use RuntimeError for potentially retryable unexpected finishes

            # 5. Check for content presence
            if not candidate.content or not candidate.content.parts:
                 logger.warning(f"API response candidate has no content parts (Finish Reason: {candidate_finish_reason}). Returning empty string.")
                 return "" # Return empty if no content but not explicitly blocked

            # 6. Safely extract text content
            response_content = getattr(response, 'text', None) # Use getattr for safety
            if response_content is None:
                 # Fallback: Try iterating parts if .text is missing (unlikely for non-stream)
                 try:
                     response_content = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                 except Exception:
                     logger.error("Failed to extract text from response parts.")
                     response_content = "" # Fallback to empty

            response_content = response_content.strip()
            if not response_content:
                 logger.warning(f"LLM generated empty text content (Finish Reason: {candidate_finish_reason}).")
                 # Return empty string, as it's valid output (though potentially undesirable)

            # 7. Success
            logger.info(f"Gemini call successful. Response length: {len(response_content)} chars.")
            return response_content

        # --- Exception Handling for the Attempt ---
        except google_api_exceptions.ResourceExhausted as e:
             last_exception = e
             logger.warning(f"Quota possibly exceeded or resource exhausted (Attempt {attempt + 1}/{config.LLM_MAX_RETRIES}): {e}")
             # Potentially retry with longer delay or raise immediately depending on policy
             # For now, continue with standard retry loop/delay.
        except google_api_exceptions.InvalidArgument as e:
             last_exception = e
             error_str = str(e).lower()
             if "prompt" in error_str and ("too long" in error_str or "size" in error_str):
                 logger.error(f"Prompt too long for model {config.GOOGLE_MODEL_NAME}. Prompt length: {prompt_len} chars.")
                 raise ValueError(f"Prompt too long for model ({prompt_len} chars): {e}") from e # Don't retry size issues
             elif "api key not valid" in error_str or "permission_denied" in error_str:
                 logger.critical(f"Invalid Google API Key or permissions: {e}")
                 raise RuntimeError(f"Invalid Google API Key or permissions: {e}") from e # Don't retry auth issues
             else:
                 logger.error(f"Invalid argument calling Google API (Attempt {attempt + 1}/{config.LLM_MAX_RETRIES}): {e}")
                 # Could be a malformed request, might not be retryable
                 # For now, let's retry, but might need specific handling.
        except google_api_exceptions.GoogleAPIError as e:
             # Catch other Google API specific errors (e.g., DeadlineExceeded, ServiceUnavailable)
             last_exception = e
             logger.warning(f"Google API Error (Attempt {attempt + 1}/{config.LLM_MAX_RETRIES}): {type(e).__name__} - {e}")
             # These are generally retryable
        except ValueError as e: # Catch our explicit ValueErrors (like prompt block)
            last_exception = e
            logger.error(f"ValueError during Gemini call (Attempt {attempt + 1}): {e}. Not retrying this error.")
            raise e # Re-raise immediately, don't retry ValueErrors raised within the try block
        except Exception as e:
            last_exception = e
            logger.warning(f"Unexpected Error calling Google API (Attempt {attempt + 1}/{config.LLM_MAX_RETRIES}): {type(e).__name__} - {e}", exc_info=False)
            # Retry generic exceptions

        # --- Retry Delay ---
        if attempt < config.LLM_MAX_RETRIES - 1:
            delay = config.LLM_RETRY_DELAY * (2 ** attempt) # Exponential backoff
            logger.info(f"Retrying Gemini call in {delay:.1f} seconds...")
            time.sleep(delay)
        else:
            logger.error(f"Max retries ({config.LLM_MAX_RETRIES}) reached for Gemini API call.")
            # Raise the specific error encountered on the last attempt
            if isinstance(last_exception, (ValueError, RuntimeError)):
                raise last_exception
            else:
                raise RuntimeError(f"LLM call failed after {config.LLM_MAX_RETRIES} retries: {last_exception}") from last_exception

    # Fallback if loop finishes unexpectedly (should not happen with current logic)
    logger.error("LLM call function exited loop unexpectedly.")
    raise RuntimeError(f"LLM call failed after retries. Last known error: {last_exception}")


def stream_gemini(prompt: str, system_prompt: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
    """
    Calls the Google Gemini model with streaming enabled and yields structured events.

    Yields dictionaries representing stream events:
    - {'type': 'chunk', 'content': str}
    - {'type': 'stream_error', 'message': str, 'is_fatal': bool}
    - {'type': 'stream_warning', 'message': str}
    - {'type': 'stream_end', 'finish_reason': str, 'usage_metadata': Optional[dict]}

    Args:
        prompt: The main user prompt.
        system_prompt: An optional system-level instruction.

    Raises:
        Catches exceptions during stream setup/iteration and yields 'stream_error'.
    """
    model_kwargs = _prepare_llm_call_args(system_prompt)
    prompt_len = len(prompt)
    logger.info(f"Calling Gemini model {config.GOOGLE_MODEL_NAME} (streaming). Prompt length: ~{prompt_len // 1000}k chars.")
    logger.debug(f"Stream prompt starts with: {prompt[:200]}...")

    stream = None
    stream_interrupted_by_error = False
    final_finish_reason = "Unknown"
    final_usage_metadata = None
    is_fatal_error = False

    try:
        model = genai.GenerativeModel(config.GOOGLE_MODEL_NAME, **model_kwargs)
        # Pass prompt correctly based on whether system_instruction is used
        if model_kwargs.get("system_instruction"):
             stream = model.generate_content(prompt, stream=True)
        else:
             stream = model.generate_content([system_prompt or "", prompt], stream=True)

        start_time = time.time() # Time the streaming part
        logger.info("Gemini stream initiated.")

        for chunk in stream:
             # 1. Check for Prompt Feedback (can block before generation starts)
             if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                  block_reason_str = str(chunk.prompt_feedback.block_reason)
                  safety_ratings_str = str(getattr(chunk.prompt_feedback, 'safety_ratings', 'N/A'))
                  error_msg = f'LLM prompt blocked by safety filters (Reason: {block_reason_str}). Cannot generate response. Ratings: {safety_ratings_str}'
                  logger.error(error_msg)
                  yield {'type': 'stream_error', 'message': error_msg, 'is_fatal': True}
                  stream_interrupted_by_error = True
                  is_fatal_error = True
                  break # Stop processing if prompt is blocked

             # 2. Check Candidate Information (blocking/finish reasons during generation)
             # Note: Safety ratings might appear on intermediate chunks or only the final one.
             if not stream_interrupted_by_error and chunk.candidates:
                 candidate = chunk.candidates[0]
                 candidate_finish_reason = getattr(candidate, 'finish_reason', None)
                 # Update final reason only if it's a terminal state (not None or UNSPECIFIED/0)
                 if candidate_finish_reason not in [None, 0, "FINISH_REASON_UNSPECIFIED"]:
                      final_finish_reason = candidate_finish_reason

                 # Define non-error finish reasons during streaming
                 acceptable_stream_reasons = {0, 1, 2, None, "STOP", "MAX_TOKENS", "FINISH_REASON_UNSPECIFIED"}

                 # Handle unexpected stops (like SAFETY=3)
                 if candidate_finish_reason not in acceptable_stream_reasons:
                     finish_reason_str = str(candidate_finish_reason)
                     safety_ratings_str = str(getattr(candidate, 'safety_ratings', 'N/A'))
                     is_safety_block = candidate_finish_reason == 3 or finish_reason_str == "SAFETY"

                     log_msg_func = logger.error if is_safety_block else logger.warning
                     yield_type = 'stream_error' if is_safety_block else 'stream_warning'
                     error_msg = f'LLM stream {"stopped" if is_safety_block else "interrupted"} due to safety/policy filter during generation (Reason: {finish_reason_str}). Output may be incomplete. Safety: {safety_ratings_str}'

                     log_msg_func(error_msg)
                     yield {'type': yield_type, 'message': error_msg, 'is_fatal': is_safety_block}

                     if is_safety_block:
                         stream_interrupted_by_error = True
                         is_fatal_error = True
                         # Allow loop to potentially get final metadata chunk if API sends one after block

             # 3. Yield Content Chunk if available and stream not fatally errored
             # Check hasattr for robustness, some chunks might be metadata-only
             if not is_fatal_error and hasattr(chunk, 'text') and chunk.text:
                  yield {'type': 'chunk', 'content': chunk.text}

        # --- Stream Finished Iterating ---
        duration = time.time() - start_time
        logger.info(f"Gemini stream iteration finished in {duration:.2f}s.")

        # 4. Get final metadata if available (might be in the last chunk or response object)
        if stream and hasattr(stream, 'usage_metadata'): # Check if the stream object itself holds it
             final_usage_metadata = getattr(stream, 'usage_metadata', None)
        elif 'chunk' in locals() and hasattr(chunk, 'usage_metadata'): # Check last chunk
             final_usage_metadata = getattr(chunk, 'usage_metadata', None)

        # 5. Signal Normal Stream End (if no fatal error occurred during iteration)
        if not is_fatal_error:
             logger.info(f"Gemini stream ended. Final finish reason: {final_finish_reason}. Usage: {final_usage_metadata}")
             yield {'type': 'stream_end', 'finish_reason': str(final_finish_reason), 'usage_metadata': final_usage_metadata}

    except (google_api_exceptions.ResourceExhausted,
            google_api_exceptions.InvalidArgument,
            google_api_exceptions.GoogleAPIError,
            Exception) as e:
        logger.error(f"Error during Google Gemini stream setup or iteration: {type(e).__name__} - {e}", exc_info=True)
        is_fatal = False
        error_message = f"LLM stream error: {type(e).__name__} - {str(e)}" # Default message
        error_str = str(e).lower()

        # Provide more specific error messages based on common issues and determine fatality
        if isinstance(e, google_api_exceptions.ResourceExhausted) or "quota" in error_str or "resource has been exhausted" in error_str:
             error_message = f"LLM stream error: Quota likely exceeded or resource exhausted. ({e})"
             is_fatal = True # Quota errors are usually fatal for the current request context
        elif isinstance(e, google_api_exceptions.InvalidArgument):
            if "api key not valid" in error_str or "permission_denied" in error_str:
                 error_message = f"LLM stream error: Invalid Google API Key or insufficient permissions. ({e})"
                 is_fatal = True
            elif "prompt" in error_str and ("too long" in error_str or "size" in error_str):
                 error_message = f"LLM stream error: Prompt likely too long ({prompt_len} chars) for the model's context window. ({e})"
                 is_fatal = True # Prompt size errors are fatal for this specific prompt
            elif "model" in error_str and ("not found" in error_str or "invalid" in error_str):
                 error_message = f"LLM stream error: Invalid model name '{config.GOOGLE_MODEL_NAME}' specified. ({e})"
                 is_fatal = True
            else:
                 # Other invalid args might indicate code issues, treat as fatal
                 error_message = f"LLM stream error: Invalid argument provided to API. ({e})"
                 is_fatal = True
        elif isinstance(e, google_api_exceptions.GoogleAPIError):
             # General API errors (like ServiceUnavailable, DeadlineExceeded) might be transient
             error_message = f"LLM stream error: Google API communication issue. ({type(e).__name__} - {e})"
             is_fatal = False # Assume potentially transient, let orchestrator decide based on context
        else: # Catchall for other exceptions
             error_message = f"LLM stream error: An unexpected error occurred. ({type(e).__name__} - {e})"
             is_fatal = True # Assume unexpected errors are fatal

        yield {'type': 'stream_error', 'message': error_message, 'is_fatal': is_fatal}
