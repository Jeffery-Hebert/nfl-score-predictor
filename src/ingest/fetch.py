"""
Every nflverse download, retried through a transient network failure.

Why. nflreadpy makes one attempt per file, and a run is all or nothing: a
dropped connection, a timeout or a 5xx from GitHub's release CDN failed the
whole pull, and with it every forecast that run would have made -- the next
scheduled run is up to sixteen hours away, often after kickoff. So a network
failure is retried, with waits long enough for a CDN hiccup to clear.

Not retried: a missing file (HTTP 404 -- a season nflverse has not published
yet), which load_seasons must see at once to fall back a season, and anything
not known to be a network hiccup, which a retry would only repeat.
"""

import time

import requests

# Seconds to wait before each retry: four attempts, about four minutes in all.
WAITS = (15, 60, 180)

NETWORK = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def transient(err: BaseException) -> bool:
    """A failure that may not happen again: no connection, a timeout, a cut-off
    transfer, rate limiting (429) or a server error (5xx). nflreadpy re-raises
    every requests error as the built-in ConnectionError with the original
    chained as its __cause__; without one, nothing says it was the network."""
    if not isinstance(err, ConnectionError):
        return False
    cause = err.__cause__
    if isinstance(cause, requests.exceptions.HTTPError):
        status = getattr(cause.response, "status_code", None)
        return status is not None and (status == 429 or status >= 500)
    return isinstance(cause, NETWORK)


def fetch(loader, *args, sleep=None, **kwargs):
    """loader(*args, **kwargs), retried after each transient failure."""
    sleep = sleep or time.sleep
    for attempt, wait in enumerate((*WAITS, None), start=1):
        try:
            return loader(*args, **kwargs)
        except Exception as err:
            if wait is None or not transient(err):
                raise
            name = getattr(loader, "__name__", "download")
            print(
                f"NOTE: {name} failed on attempt {attempt} ({err}); "
                f"retrying in {wait}s",
                flush=True,
            )
            sleep(wait)
