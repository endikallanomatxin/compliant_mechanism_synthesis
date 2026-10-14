"""Per-operation mechanics counters, including saved-context adjoint work."""

import time
from contextvars import ContextVar
from functools import wraps

CURRENT = ContextVar("mechanics_work", default=None)


def record(name, amount=1, work=None):
    work = CURRENT.get() if work is None else work
    if work is not None:
        work[name] = work.get(name, 0) + amount


def measured(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        current = CURRENT.get()
        work = {} if current is None else current
        before = dict(work)
        token = CURRENT.set(work)
        started = time.perf_counter()
        try:
            result = method(self, *args, **kwargs)
            delta = {key: value - before.get(key, 0) for key, value in work.items()}
            self.last_diagnostics.update(delta)
            if method.__name__ == "_adaptive_gradient":
                self.last_diagnostics["seconds"] = time.perf_counter() - started
            for key, value in delta.items():
                self.writer.add_scalar(
                    "mechanics/" + key,
                    value,
                    self.gradient_updates
                    if method.__name__ == "_adaptive_gradient"
                    else self.iteration,
                )
            return result
        finally:
            CURRENT.reset(token)

    return run
