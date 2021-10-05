import time


def to_string(seconds: float) -> str:
    if seconds >= 60:
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)
        return f"{minutes} minutes and {seconds} seconds"
    return f"{seconds:.2f} seconds"


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"Done in {to_string(elapsed)}")
        return result

    return wrapper
