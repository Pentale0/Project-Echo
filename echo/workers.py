import queue
import threading
from collections.abc import Callable


class AsyncJob:
    """Run fn() on a worker thread and post the result to the given queue.

    Payload posted to the queue:
      ("done", label, result)    on success
      ("error", label, message)  on failure
    """

    def __init__(
        self,
        queue_out: queue.Queue,
        label: str,
        fn: Callable[[], object],
    ) -> None:
        self._queue = queue_out
        self._label = label
        self._fn = fn

    def start(self) -> None:
        thread = threading.Thread(target=self._run, name=f"echo-{self._label}", daemon=True)
        thread.start()

    def _run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 - surface any worker error to the UI
            self._queue.put(("error", self._label, f"{type(exc).__name__}: {exc}"))
        else:
            self._queue.put(("done", self._label, result))