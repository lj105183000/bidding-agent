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

    def allow(self, client_ip: str) -> bool:
        """判断客户端当前请求是否允许通过（方法名与测试契约一致）。"""
        now = time.monotonic()
        window_start = now - self.window_seconds

        with self._lock:
            timestamps = self._requests[client_ip]

            # 移除窗口外的旧时间戳
            while timestamps and timestamps[0] <= window_start:
                timestamps.popleft()

            if len(timestamps) >= self.max_requests:
                return False

            timestamps.append(now)
            return True

    # 兼容别名：部分早期调用方使用 is_allowed
    def is_allowed(self, client_ip: str) -> bool:
        return self.allow(client_ip)

    def reset(self, client_ip: str | None = None) -> None:
        """清空指定客户端（或全部）的限流计数，便于测试。"""
        with self._lock:
            if client_ip is None:
                self._requests.clear()
            else:
                self._requests.pop(client_ip, None)
