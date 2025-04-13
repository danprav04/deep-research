
# tests/test_web_research.py (Placeholder)
import pytest
from unittest.mock import patch, MagicMock, mock_open
import requests
import os

from deep_research_app import web_research
from deep_research_app import config
from duckduckgo_search import DDGS, exceptions as ddgs_exceptions

# --- Fixtures ---

@pytest.fixture
def mock_requests_get(mocker):
    """Fixture to mock requests.get"""
    mock = mocker.patch('requests.get', autospec=True)
    # Configure a default successful response
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.headers = {'content-type': 'text/html; charset=utf-8'}
    mock_response.encoding = 'utf-8'
    mock_response.apparent_encoding = 'utf-8'
    mock_response.content = b"<html><body><p>Mock HTML content</p></body></html>"
    # Make response usable as a context manager and iterable
    mock_response.iter_content.return_value = iter([mock_response.content])
    mock_response.raise_for_status.return_value = None
    mock_response.close.return_value = None
    mock.return_value = mock_response
    return mock

@pytest.fixture
def mock_ddgs(mocker):
    """Fixture to mock duckduckgo_search.DDGS"""
    # Mock the __enter__ and __exit__ methods for the context manager
    mock_instance = MagicMock()
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_instance
    mock_context_manager.__exit__.return_value = None
    mock_ddgs_class = mocker.patch('deep_research_app.web_research.DDGS', return_value=mock_context_manager)
    # Make the instance returned by __enter__ callable/usable
    mock_instance.text.return_value = iter([
        {'href': 'http://example.com/result1'},
        {'href': 'https://example.org/result2'},
        {'href': 'http://example.com/%20result%203%20'}, # Needs unquoting
        {'href': 'ftp://invalid.com'}, # Invalid scheme
    ])
    return mock_instance

@pytest.fixture
def mock_tempfile(mocker):
    """Fixture to mock tempfile.mkstemp"""
    # Return a mock file descriptor and a predictable path
    mock_fd = 10
    mock_path = '/tmp/sanitized_test_file.txt'
    mock = mocker.patch('tempfile.mkstemp', return_value=(mock_fd, mock_path))
    # Mock os.fdopen to work with the mock fd
    mocker.patch('os.fdopen', mock_open())
    # Mock os.path.exists and os.path.getsize for verification after write
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('os.path.getsize', return_value=100) # Simulate non-empty file
    return mock, mock_path

@pytest.fixture
def mock_sanitize(mocker):
    """Fixture to mock the sanitize_scraped_content utility."""
    # By default, return input slightly modified to indicate it was called
    mock = mocker.patch('deep_research_app.web_research.sanitize_scraped_content', side_effect=lambda x, url: f"SANITIZED:{x}")
    return mock

# --- Tests for search_duckduckgo_provider ---

def test_search_ddg_success(mock_ddgs):
    """Test successful DDG search."""
    keywords = ["test", "search"]
    result = web_research.search_duckduckgo_provider(keywords, max_results=5)

    assert result['success'] is True
    assert result['engine'] == "DuckDuckGo"
    assert result['error'] is None
    # Check that URLs are returned and cleaned/validated
    assert 'http://example.com/result1' in result['urls']
    assert 'https://example.org/result2' in result['urls']
    assert 'http://example.com/ result 3 ' in result['urls'] # Check unquoting
    assert 'ftp://invalid.com' not in result['urls'] # Check scheme validation
    assert len(result['urls']) == 3
    mock_ddgs.text.assert_called_once_with("test search", max_results=5)

def test_search_ddg_rate_limit_retry_success(mock_ddgs, mocker):
    """Test retry logic after rate limit, eventually succeeding."""
    mock_sleep = mocker.patch('time.sleep', return_value=None)
    # Simulate rate limit error then success
    mock_ddgs.text.side_effect = [
        ddgs_exceptions.RateLimitException("Too many requests"),
        iter([{'href': 'http://success.com'}]) # Successful results on retry
    ]

    keywords = ["retry", "test"]
    # Temporarily reduce retry delay for testing
    with patch.object(config, 'DDGS_RETRY_DELAY_SECONDS', 0.1):
        result = web_research.search_duckduckgo_provider(keywords, max_results=3)

    assert result['success'] is True
    assert 'http://success.com' in result['urls']
    assert mock_ddgs.text.call_count == 2
    assert mock_sleep.call_count == 1 # Should sleep once before retry

def test_search_ddg_max_retries_fail(mock_ddgs, mocker):
    """Test failure after exceeding max retries."""
    mock_sleep = mocker.patch('time.sleep', return_value=None)
    mock_ddgs.text.side_effect = ddgs_exceptions.RateLimitException("Persistent rate limit")

    keywords = ["fail", "test"]
    # Temporarily reduce retries and delay
    with patch.object(config, 'DDGS_RETRY_DELAY_SECONDS', 0.1), \
         patch.object(web_research, 'max_outer_retries', 1): # Override local variable for test
        result = web_research.search_duckduckgo_provider(keywords, max_results=3)

    assert result['success'] is False
    assert "Max retries reached" in result['error']
    assert mock_ddgs.text.call_count == 2 # Initial call + 1 retry
    assert mock_sleep.call_count == 1

def test_search_ddg_empty_keywords():
    """Test search with empty keywords list."""
    result = web_research.search_duckduckgo_provider([], max_results=5)
    assert result['success'] is True # No error, just no results
    assert result['urls'] == []
    assert result['error'] is None

# --- Tests for perform_web_search ---

def test_perform_web_search_success(mocker):
    """Test aggregating results from (mocked) providers."""
    mock_provider1 = mocker.patch('deep_research_app.web_research.search_duckduckgo_provider')
    mock_provider1.return_value = {
        'engine': 'MockProvider1',
        'urls': ['http://url1.com', 'http://url2.com'],
        'success': True, 'error': None
    }
    # Assume only one provider for simplicity, can add more mocks if needed

    urls, errors = web_research.perform_web_search(["keywords"])

    assert errors == []
    assert sorted(urls) == ['http://url1.com', 'http://url2.com']
    mock_provider1.assert_called_once_with(["keywords"], max_results=config.MAX_SEARCH_RESULTS_PER_ENGINE_STEP)

def test_perform_web_search_with_errors(mocker):
    """Test handling errors from providers."""
    mock_provider1 = mocker.patch('deep_research_app.web_research.search_duckduckgo_provider')
    mock_provider1.return_value = {
        'engine': 'MockProvider1',
        'urls': [],
        'success': False, 'error': "Provider error message"
    }

    urls, errors = web_research.perform_web_search(["keywords"])

    assert urls == []
    assert len(errors) == 1
    assert "MockProvider1: Provider error message" in errors[0]

def test_perform_web_search_duplicates(mocker):
     """Test that duplicate URLs from providers are handled."""
     mock_provider1 = mocker.patch('deep_research_app.web_research.search_duckduckgo_provider')
     mock_provider1.return_value = {
         'engine': 'MockProvider1',
         'urls': ['http://url1.com', 'http://url2.com'],
         'success': True, 'error': None
     }
     # Add a mock for a second provider if testing aggregation from multiple
     # mock_provider2 = mocker.patch('deep_research_app.web_research.search_some_other_provider')
     # mock_provider2.return_value = {
     #     'engine': 'MockProvider2',
     #     'urls': ['http://url2.com', 'http://url3.com'], # url2 is duplicate
     #     'success': True, 'error': None
     # }
     # # Ensure the second provider is included in the list in web_research.py for this test

     urls, errors = web_research.perform_web_search(["keywords"])
     assert errors == []
     assert sorted(urls) == ['http://url1.com', 'http://url2.com'] # Should only contain unique URLs

# --- Tests for scrape_url ---

def test_scrape_url_success(mock_requests_get, mock_sanitize, mock_tempfile):
    """Test successful scraping, sanitization, and saving."""
    mock_temp_creator, mock_path = mock_tempfile
    # Ensure sanitizer returns content long enough
    mock_sanitize.return_value = "SANITIZED:" + "A" * config.MIN_SANITIZED_CONTENT_LENGTH

    url = "http://example.com/goodpage"
    result = web_research.scrape_url(url)

    assert result is not None
    assert result['url'] == url
    assert result['temp_filepath'] == mock_path
    assert result['original_size'] > 0
    assert result['sanitized_size'] > 0
    mock_requests_get.assert_called_once_with(url, headers=pytest.ANY, timeout=config.REQUEST_TIMEOUT, allow_redirects=True, stream=True)
    mock_sanitize.assert_called_once()
    mock_temp_creator.assert_called_once()
    # Check if content was written (via mock_open used by os.fdopen mock)
    handle = mock_open().return_value
    handle.write.assert_called_once_with("SANITIZED:" + "A" * config.MIN_SANITIZED_CONTENT_LENGTH)

def test_scrape_url_non_html(mock_requests_get):
    """Test skipping non-HTML content types."""
    mock_response = mock_requests_get.return_value
    mock_response.headers = {'content-type': 'application/pdf'}

    result = web_research.scrape_url("http://example.com/document.pdf")
    assert result is None

def test_scrape_url_too_large(mock_requests_get):
    """Test skipping content exceeding size limit."""
    # Simulate large content by making bytes_read exceed limit
    large_content = b'a' * (config.MAX_SCRAPE_CONTENT_BYTES + 100)
    mock_response = mock_requests_get.return_value
    mock_response.iter_content.return_value = iter([large_content]) # Single large chunk

    result = web_research.scrape_url("http://example.com/largepage")
    assert result is None

def test_scrape_url_http_error(mock_requests_get):
    """Test handling of HTTP errors."""
    mock_response = mock_requests_get.return_value
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)

    result = web_research.scrape_url("http://example.com/notfound")
    assert result is None

def test_scrape_url_connection_error(mock_requests_get):
    """Test handling of connection errors."""
    mock_requests_get.side_effect = requests.exceptions.ConnectionError("DNS lookup failed")

    result = web_research.scrape_url("http://nonexistent.domain.xyz")
    assert result is None

def test_scrape_url_timeout_error(mock_requests_get):
    """Test handling of timeout errors."""
    mock_requests_get.side_effect = requests.exceptions.Timeout("Request timed out")

    result = web_research.scrape_url("http://example.com/slowpage")
    assert result is None

def test_scrape_url_sanitization_fails_or_too_short(mock_requests_get, mock_sanitize):
    """Test skipping if content is too short after sanitization."""
    # 1. Sanitizer returns very short content
    mock_sanitize.return_value = "Too short"
    result1 = web_research.scrape_url("http://example.com/shortcontent")
    assert result1 is None

    # 2. Sanitizer returns empty content
    mock_sanitize.return_value = ""
    result2 = web_research.scrape_url("http://example.com/emptycontent")
    assert result2 is None

def test_scrape_url_temp_file_write_error(mock_requests_get, mock_sanitize, mock_tempfile, mocker):
    """Test handling errors during temp file writing."""
    mock_temp_creator, mock_path = mock_tempfile
    mock_sanitize.return_value = "SANITIZED:" + "A" * config.MIN_SANITIZED_CONTENT_LENGTH
    # Mock os.fdopen to raise an error
    mocker.patch('os.fdopen', side_effect=IOError("Disk full"))
    # Mock os.remove for cleanup check
    mock_remove = mocker.patch('os.remove')

    url = "http://example.com/writefail"
    result = web_research.scrape_url(url)

    assert result is None
    # Check if cleanup was attempted
    mock_remove.assert_called_with(mock_path)
