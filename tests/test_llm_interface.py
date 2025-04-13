
# tests/test_llm_interface.py (Placeholder)
import pytest
from unittest.mock import patch, MagicMock

from deep_research_app import llm_interface
from deep_research_app import config
import google.generativeai as genai
from google.api_core import exceptions as google_api_exceptions

# --- Fixtures ---

# Mock the genai.GenerativeModel instance
@pytest.fixture
def mock_genai_model():
    with patch('google.generativeai.GenerativeModel', autospec=True) as mock_model_class:
        mock_instance = mock_model_class.return_value
        yield mock_instance

# --- Test Cases for call_gemini ---

def test_call_gemini_success(mock_genai_model):
    """Test successful non-streaming call."""
    # Configure mock response
    mock_response = MagicMock()
    mock_response.text = "Successful response text."
    mock_response.candidates = [MagicMock(finish_reason="STOP", content=MagicMock(parts=[MagicMock()]))] # Simulate valid candidate structure
    mock_response.prompt_feedback = None # No blocking
    mock_genai_model.generate_content.return_value = mock_response

    result = llm_interface.call_gemini("Test prompt")

    assert result == "Successful response text."
    mock_genai_model.generate_content.assert_called_once()
    # Add assertions to check arguments passed to generate_content if needed

def test_call_gemini_prompt_blocked(mock_genai_model):
    """Test call blocked by prompt safety filters."""
    mock_response = MagicMock()
    mock_response.prompt_feedback = MagicMock(block_reason="SAFETY")
    mock_response.candidates = [] # No candidates when prompt blocked
    mock_genai_model.generate_content.return_value = mock_response

    with pytest.raises(ValueError, match="API response blocked due to prompt feedback"):
        llm_interface.call_gemini("Blocked prompt")

def test_call_gemini_response_blocked(mock_genai_model):
    """Test call blocked during response generation."""
    mock_response = MagicMock()
    mock_response.prompt_feedback = None
    mock_response.candidates = [MagicMock(finish_reason="SAFETY", content=None)] # Finish reason SAFETY
    mock_genai_model.generate_content.return_value = mock_response

    with pytest.raises(ValueError, match="API candidate finished unexpectedly or was blocked"):
        llm_interface.call_gemini("Prompt causing safety block")

def test_call_gemini_no_candidates(mock_genai_model):
    """Test response with no candidates."""
    mock_response = MagicMock()
    mock_response.prompt_feedback = None
    mock_response.candidates = []
    mock_genai_model.generate_content.return_value = mock_response

    with pytest.raises(ValueError, match="API response contained no candidates"):
        llm_interface.call_gemini("Prompt leading to no candidates")

def test_call_gemini_empty_content(mock_genai_model):
    """Test response with empty content but successful finish."""
    mock_response = MagicMock()
    mock_response.text = "" # Empty text
    mock_response.candidates = [MagicMock(finish_reason="STOP", content=MagicMock(parts=[]))] # Valid finish, no parts
    mock_response.prompt_feedback = None
    mock_genai_model.generate_content.return_value = mock_response

    result = llm_interface.call_gemini("Test prompt")
    assert result == "" # Should return empty string

def test_call_gemini_retry_logic(mock_genai_model):
    """Test retry logic on transient API errors."""
    # Simulate a transient error then success
    mock_response_success = MagicMock(text="Success after retry", candidates=[MagicMock(finish_reason="STOP", content=MagicMock(parts=[MagicMock()]))], prompt_feedback=None)
    mock_genai_model.generate_content.side_effect = [
        google_api_exceptions.DeadlineExceeded("Timeout"), # First call fails
        mock_response_success # Second call succeeds
    ]

    # Temporarily reduce retry delay for faster testing
    with patch('time.sleep', return_value=None), \
         patch.object(config, 'LLM_RETRY_DELAY', 0.1):
        result = llm_interface.call_gemini("Test prompt")

    assert result == "Success after retry"
    assert mock_genai_model.generate_content.call_count == 2

def test_call_gemini_max_retries_fail(mock_genai_model):
    """Test failure after exhausting retries."""
    mock_genai_model.generate_content.side_effect = google_api_exceptions.ServiceUnavailable("Server down")

    # Temporarily reduce retry delay
    with patch('time.sleep', return_value=None), \
         patch.object(config, 'LLM_RETRY_DELAY', 0.1), \
         patch.object(config, 'LLM_MAX_RETRIES', 2): # Set max retries for test
            with pytest.raises(RuntimeError, match="LLM call failed after 2 retries"):
                llm_interface.call_gemini("Test prompt")

    assert mock_genai_model.generate_content.call_count == 2

def test_call_gemini_invalid_key(mock_genai_model):
    """Test non-retryable InvalidArgument for API key."""
    mock_genai_model.generate_content.side_effect = google_api_exceptions.InvalidArgument("API key not valid.")

    with pytest.raises(RuntimeError, match="Invalid Google API Key"):
        llm_interface.call_gemini("Test prompt")
    assert mock_genai_model.generate_content.call_count == 1 # Should not retry

def test_call_gemini_prompt_too_long(mock_genai_model):
    """Test non-retryable InvalidArgument for prompt size."""
    mock_genai_model.generate_content.side_effect = google_api_exceptions.InvalidArgument("Prompt is too long.")

    with pytest.raises(ValueError, match="Prompt too long"):
        llm_interface.call_gemini("Very long prompt...")
    assert mock_genai_model.generate_content.call_count == 1 # Should not retry


# --- Test Cases for stream_gemini ---
# Testing generators requires iterating over the results

def test_stream_gemini_success(mock_genai_model):
    """Test successful streaming call."""
    mock_stream_chunks = [
        MagicMock(text="Hello ", candidates=[MagicMock(finish_reason=None)], prompt_feedback=None),
        MagicMock(text="World!", candidates=[MagicMock(finish_reason=None)], prompt_feedback=None),
        MagicMock(text=None, candidates=[MagicMock(finish_reason="STOP")], prompt_feedback=None, usage_metadata="mock_metadata") # Final chunk with reason
    ]
    mock_genai_model.generate_content.return_value = iter(mock_stream_chunks)

    results = list(llm_interface.stream_gemini("Test stream prompt"))

    assert len(results) == 3
    assert results[0] == {'type': 'chunk', 'content': 'Hello '}
    assert results[1] == {'type': 'chunk', 'content': 'World!'}
    assert results[2] == {'type': 'stream_end', 'finish_reason': 'STOP', 'usage_metadata': 'mock_metadata'}
    mock_genai_model.generate_content.assert_called_once()

def test_stream_gemini_prompt_blocked(mock_genai_model):
    """Test stream blocked by prompt safety."""
    mock_stream_chunks = [
        MagicMock(text=None, candidates=[], prompt_feedback=MagicMock(block_reason="SAFETY"))
    ]
    mock_genai_model.generate_content.return_value = iter(mock_stream_chunks)

    results = list(llm_interface.stream_gemini("Blocked stream prompt"))

    assert len(results) == 1
    assert results[0]['type'] == 'stream_error'
    assert results[0]['is_fatal'] is True
    assert "LLM prompt blocked by safety filters" in results[0]['message']

def test_stream_gemini_response_blocked(mock_genai_model):
    """Test stream blocked during response generation."""
    mock_stream_chunks = [
        MagicMock(text="Some safe content.", candidates=[MagicMock(finish_reason=None)], prompt_feedback=None),
        MagicMock(text=None, candidates=[MagicMock(finish_reason="SAFETY")], prompt_feedback=None) # Blocked chunk
    ]
    mock_genai_model.generate_content.return_value = iter(mock_stream_chunks)

    results = list(llm_interface.stream_gemini("Stream prompt causing block"))

    assert len(results) == 2 # Chunk + Error
    assert results[0] == {'type': 'chunk', 'content': 'Some safe content.'}
    assert results[1]['type'] == 'stream_error' # Should yield fatal error on safety block
    assert results[1]['is_fatal'] is True
    assert "stopped due to safety/policy filter during generation" in results[1]['message']

def test_stream_gemini_api_error(mock_genai_model):
    """Test handling of API errors during streaming."""
    mock_genai_model.generate_content.side_effect = google_api_exceptions.DeadlineExceeded("Timeout during stream")

    results = list(llm_interface.stream_gemini("Stream prompt causing timeout"))

    assert len(results) == 1
    assert results[0]['type'] == 'stream_error'
    assert results[0]['is_fatal'] is False # DeadlineExceeded might be transient
    assert "LLM stream error: Google API communication issue" in results[0]['message']

def test_stream_gemini_invalid_key_error(mock_genai_model):
    """Test handling fatal API key errors during streaming setup."""
    mock_genai_model.generate_content.side_effect = google_api_exceptions.InvalidArgument("API key not valid.")

    results = list(llm_interface.stream_gemini("Stream prompt"))

    assert len(results) == 1
    assert results[0]['type'] == 'stream_error'
    assert results[0]['is_fatal'] is True
    assert "Invalid Google API Key" in results[0]['message']
