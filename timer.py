import time


def to_string(seconds: float) -> str:
    if seconds >= 60:
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)
        return (
            f"{minutes} {'minute' if minutes == 1 else 'minutes'} and "
            f"{seconds} {'second' if seconds == 1 else 'seconds'}"
        )
    elif seconds == 1:
        return
    return f"{seconds:.2f} {'second' if seconds == 1 else 'seconds'}"


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"Done in {to_string(elapsed)}")
        return result

    return wrapper
