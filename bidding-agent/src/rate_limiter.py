import time
from collections import defaultdict, deque
from threading import Lock


class RateLimiter:
    """基于客户端 IP 的滑动窗口限流器。"""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def is_allowed(self, client_ip: str) -> bool:
        """判断客户端当前请求是否允许通过。"""
        now = time.monotonic()
        window_start = now - self.window_seconds

        with self._lock:
            timestamps = self._requests[client_ip]

            while timestamps and timestamps[0] <= window_start:
                timestamps.popleft()

            if len(timestamps) >= self.max_requests:
                return False

            timestamps.append(now)
            return True