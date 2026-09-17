import threading
import time

from src.rate_limiter import RateLimiter


def test_allows_requests_within_quota():
    limiter = RateLimiter(max_requests=3, window_seconds=60)

    assert limiter.allow("client-1")
    assert limiter.allow("client-1")
    assert limiter.allow("client-1")


def test_rejects_requests_over_quota():
    limiter = RateLimiter(max_requests=2, window_seconds=60)

    assert limiter.allow("client-1")
    assert limiter.allow("client-1")
    assert not limiter.allow("client-1")


def test_different_clients_have_independent_quotas():
    limiter = RateLimiter(max_requests=1, window_seconds=60)

    assert limiter.allow("client-1")
    assert not limiter.allow("client-1")
    assert limiter.allow("client-2")


def test_window_expires():
    limiter = RateLimiter(max_requests=1, window_seconds=0.05)

    assert limiter.allow("client-1")
    assert not limiter.allow("client-1")

    time.sleep(0.06)

    assert limiter.allow("client-1")


def test_thread_safety_exact_quota():
    quota = 10
    limiter = RateLimiter(max_requests=quota, window_seconds=60)

    results = []
    lock = threading.Lock()

    def worker():
        allowed = limiter.allow("client-1")
        with lock:
            results.append(allowed)

    threads = [threading.Thread(target=worker) for _ in range(30)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert sum(results) == quota