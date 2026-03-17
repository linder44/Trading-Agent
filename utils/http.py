"""HTTP utilities with retry logic for unreliable APIs.

All external API calls should use `request_with_retry` instead of `requests.get`
to handle transient timeouts and network errors gracefully.
"""

import time as _time

import requests
from loguru import logger


class HttpClientError(Exception):
    """Raised on 4xx HTTP errors (client errors that won't fix on retry)."""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        super().__init__(message)


def request_with_retry(
    url: str,
    params: dict | None = None,
    timeout: int = 5,
    retries: int = 3,
    backoff: float = 1.0,
) -> requests.Response | None:
    """Make HTTP GET request with retry on timeout/connection errors.

    Args:
        url: URL to fetch
        params: Query parameters
        timeout: Timeout per attempt in seconds
        retries: Number of attempts (default 3)
        backoff: Wait between retries in seconds (doubles each retry)

    Returns:
        Response object on success, None on all retries exhausted.

    Raises:
        HttpClientError: on 4xx responses so callers can distinguish
            "not supported" (400) from "network issue" (timeout/None).
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < retries:
                wait = backoff * (2 ** (attempt - 1))  # 1s, 2s, 4s
                logger.debug(f"Retry {attempt}/{retries} for {url} after {wait}s: {e}")
                _time.sleep(wait)
            else:
                logger.warning(f"All {retries} attempts failed for {url}: {e}")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if 400 <= status < 500:
                raise HttpClientError(status, str(e)) from e
            logger.warning(f"HTTP error for {url}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error for {url}: {e}")
            return None

    return None
