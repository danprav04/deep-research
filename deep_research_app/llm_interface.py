# llm_interface.py
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import config

# Note: genai.configure() should be called once in app.py after loading config.

def _prepare_llm_call_args(system_prompt=None):
    """Prepares common arguments for LLM calls."""
    generation_config = genai.types.GenerationConfig(
        temperature=config.LLM_TEMPERATURE,
        # max_output_tokens=8192 # Adjust if needed per model limits
    )
    # Map string constants from config to HarmBlockThreshold enum if needed by API version
    # For recent versions, string names often work directly.
    # If using older versions, you might need:
    # from google.generativeai.types import HarmCategory, HarmBlockThreshold
    # safety_settings_mapped = {
    #     getattr(HarmCategory, k): getattr(HarmBlockThreshold, v)
    #     for k, v in config.SAFETY_SETTINGS.items()
    # }
    # Using string names as they are likely compatible:
    safety_settings_mapped = {
        HarmCategory[k]: HarmBlockThreshold[v]
        for k, v in config.SAFETY_SETTINGS.items()
    }


    model_kwargs = {
        "generation_config": generation_config,
        "safety_settings": safety_settings_mapped,
    }
    if system_prompt:
         # Use the dedicated system_instruction parameter if available and non-empty
         model_kwargs["system_instruction"] = system_prompt

    return model_kwargs

def call_gemini(prompt, system_prompt=None):
    """Calls the specified Google Gemini model with retry logic."""
    model_kwargs = _prepare_llm_call_args(system_prompt)

    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            model = genai.GenerativeModel(config.GOOGLE_MODEL_NAME, **model_kwargs)
            # Pass only the user prompt if system_instruction is used
            effective_prompt = prompt
            response = model.generate_content(effective_prompt)

            # Handle blocked prompts or empty responses
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = str(response.prompt_feedback.block_reason)
                 safety_ratings_str = str(getattr(response.prompt_feedback, 'safety_ratings', 'N/A'))
                 raise ValueError(f"API response blocked by safety settings. Block Reason: {block_reason}. Safety Ratings: {safety_ratings_str}")

            if not response.candidates:
                 finish_details = getattr(response, 'finish_details', 'N/A')
                 raise ValueError(f"API response contained no candidates. Finish Details: {finish_details}")

            candidate = response.candidates[0]
            if candidate.finish_reason not in [None, 1, "STOP", "MAX_TOKENS"]: # 1 == StopReason.STOP
                 raise ValueError(f"API response candidate finished unexpectedly. Finish Reason: {candidate.finish_reason}. Safety Ratings: {getattr(candidate, 'safety_ratings', 'N/A')}")

            if not candidate.content or not candidate.content.parts:
                 raise ValueError(f"API response candidate has no content parts. Finish Reason: {candidate.finish_reason}")

            response_content = response.text.strip()
            if not response_content:
                 # Don't retry on genuinely empty content, might be intentional
                 print("Warning: LLM generated empty text content after successful call.")
                 return "" # Return empty string

            return response_content

        except Exception as e:
            print(f"Error calling Google Gemini API (Attempt {attempt + 1}/{config.LLM_MAX_RETRIES}): {e}")
            if "API key not valid" in str(e):
                 print("Critical Error: Invalid Google API Key.")
                 raise # Don't retry on invalid key
            if "quota" in str(e).lower() or "resource has been exhausted" in str(e).lower():
                 print("Warning: Quota possibly exceeded.")
                 # Potentially raise or implement longer backoff depending on policy
            if attempt < config.LLM_MAX_RETRIES - 1:
                print(f"Retrying in {config.LLM_RETRY_DELAY} seconds...")
                time.sleep(config.LLM_RETRY_DELAY)
            else:
                print("Max retries reached. Failing LLM call.")
                raise # Raise the last exception after retries are exhausted

def stream_gemini(prompt, system_prompt=None):
    """Calls the Google Gemini model with streaming enabled and yields content chunks."""
    model_kwargs = _prepare_llm_call_args(system_prompt)

    try:
        model = genai.GenerativeModel(config.GOOGLE_MODEL_NAME, **model_kwargs)
        # Pass only the user prompt if system_instruction is used
        effective_prompt = prompt
        stream = model.generate_content(effective_prompt, stream=True)

        stream_blocked = False

        for chunk in stream:
             # Check for prompt blocking first (can happen before generation)
             if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                  block_reason_str = str(chunk.prompt_feedback.block_reason)
                  safety_ratings_str = str(getattr(chunk.prompt_feedback, 'safety_ratings', 'N/A'))
                  print(f"ERROR: LLM stream blocked by prompt safety filters. Reason: {block_reason_str}. Ratings: {safety_ratings_str}")
                  yield {'type': 'stream_error', 'message': f'LLM prompt blocked by safety filters (Reason: {block_reason_str}). Cannot generate response.'}
                  stream_blocked = True
                  break # Stop processing if prompt is blocked

             # Check for candidate blocking/finish reasons during generation
             if not stream_blocked and chunk.candidates:
                 candidate = chunk.candidates[0]
                 # Check finish_reason if present
                 if candidate.finish_reason and candidate.finish_reason not in [None, 1, "STOP", "MAX_TOKENS"]: # 1 == StopReason.STOP
                     finish_reason_str = str(candidate.finish_reason)
                     safety_ratings_str = str(getattr(candidate, 'safety_ratings', 'N/A'))
                     print(f"WARNING: LLM stream stopped during generation. Reason: {finish_reason_str}. Safety: {safety_ratings_str}")
                     yield {'type': 'stream_warning', 'message': f'LLM stream may have been interrupted or blocked during generation (Reason: {finish_reason_str}). Output may be incomplete.'}
                     # If blocked due to safety during generation, stop yielding content
                     if finish_reason_str == "SAFETY": # FinishReason.SAFETY == 2
                         stream_blocked = True
                         # Potentially break here if no more useful info is expected

             # Yield content chunk if not blocked
             if not stream_blocked and hasattr(chunk, 'text') and chunk.text:
                  yield {'type': 'chunk', 'content': chunk.text}

        # Indicate normal completion if stream wasn't blocked
        if not stream_blocked:
             yield {'type': 'stream_end', 'finish_reason': 'IterationComplete'}

    except Exception as e:
        print(f"Error during Google Gemini stream: {e}")
        error_message = f"LLM stream error: {e}"
        if "API key not valid" in str(e):
            error_message = f"LLM stream error: Invalid Google API Key. ({e})"
        elif "quota" in str(e).lower() or "resource has been exhausted" in str(e).lower():
            error_message = f"LLM stream error: Quota likely exceeded. ({e})"
        elif "prompt" in str(e).lower() and ("too long" in str(e).lower() or "size" in str(e).lower()):
             error_message = f"LLM stream error: Prompt likely too long for the model's context window. ({e})"

        yield {'type': 'stream_error', 'message': error_message}