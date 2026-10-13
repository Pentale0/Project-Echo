from __future__ import annotations

import threading

import numpy as np


class Transcriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel  # slow import

            # First load downloads the model into the HuggingFace cache.
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        if audio.size == 0:
            return ""
        self.load()
        segments, _info = self._model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        return "".join(seg.text for seg in segments).strip()