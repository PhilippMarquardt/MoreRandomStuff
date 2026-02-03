import time
import threading
from functools import wraps


def ttl_cache(ttl: int = 300):
    def decorator(func):
        cache = {}
        lock = threading.Lock()

        @wraps(func)
        def wrapped(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()

            # Fast path: read under lock
            with lock:
                hit = cache.get(key)
                if hit is not None:
                    value, timestamp = hit
                    if now - timestamp < ttl:
                        return value

            # Slow path: compute without holding lock
            result = func(*args, **kwargs)

            # Store under lock (double-check to avoid overwriting fresher value)
            now2 = time.time()
            with lock:
                hit = cache.get(key)
                if hit is not None:
                    value, timestamp = hit
                    if now2 - timestamp < ttl:
                        return value  # someone refreshed while we computed
                cache[key] = (result, now2)
                return result

        return wrapped
    return decorator
