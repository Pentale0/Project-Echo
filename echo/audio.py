from __future__ import annotations

import threading
import time

import numpy as np
import sounddevice as sd

_CHUNK = int(16000 / 10)  # 100 ms frames


class Recorder:
    def __init__(self, sample_rate: int = 16000, device: int | None = None) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._stream: sd.InputStream | None = None
        self._buffers: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._started_at: float = 0.0
        self._level: float = 0.0

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    @property
    def level(self) -> float:
        """Current input RMS level in 0..1, refreshed every callback."""
        return self._level

    @property
    def duration_seconds(self) -> float:
        with self._lock:
            total = sum(int(b.size) for b in self._buffers) / self.sample_rate
        return total

    def start(self) -> None:
        if self._stream is not None:
            return
        self._buffers = []

        def callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
            self._level = float(np.sqrt(np.mean(np.square(indata))))
            with self._lock:
                self._buffers.append(indata.copy())

        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=_CHUNK,
            callback=callback,
        )
        self._stream.start()
        self._started_at = time.monotonic()

    def stop(self) -> np.ndarray:
        """Stop recording and return mono float32 audio in [-1, 1]."""
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        time.sleep(0.05)  # let any in-flight callback finish before swapping
        with self._lock:
            buffers, self._buffers = self._buffers, []
        if not buffers:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(buffers).astype(np.float32)

    @staticmethod
    def list_devices() -> list[tuple[int, str]]:
        return [
            (i, dev["name"])
            for i, dev in enumerate(sd.query_devices())
            if dev["max_input_channels"] > 0
        ]