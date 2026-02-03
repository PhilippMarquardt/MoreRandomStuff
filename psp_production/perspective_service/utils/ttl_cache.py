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
            with lock:
                if key in cache:
                    value, timestamp = cache[key]
                    if now - timestamp < ttl:
                        return value
                result = func(*args, **kwargs)
                cache[key] = (result, now)
                return result
        return wrapped
    return decorator