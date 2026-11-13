"""Classify temporary download failures and bound retry delays."""
import http.client
import socket
import ssl
import urllib.error
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


def transient_download_error(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in {408, 429, 500, 502, 503, 504}
    if isinstance(error, ssl.SSLError):
        return False
    if isinstance(error, urllib.error.URLError):
        return transient_download_error(error.reason)
    if isinstance(error, socket.gaierror):
        return error.errno == socket.EAI_AGAIN
    return isinstance(error, (TimeoutError, ConnectionError, http.client.IncompleteRead))


def retry_delay(error, attempt, initial_delay, maximum_delay):
    delay = min(initial_delay * 2 ** (attempt - 1), maximum_delay)
    if isinstance(error, urllib.error.HTTPError) and error.headers:
        value = error.headers.get("Retry-After")
        if value:
            try:
                requested = float(value)
            except ValueError:
                try:
                    date = parsedate_to_datetime(value)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    requested = (date - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    return delay
            if requested >= 0:
                delay = max(delay, min(requested, maximum_delay))
    return delay
