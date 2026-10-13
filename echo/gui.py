from __future__ import annotations

import queue
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from echo.audio import Recorder
from echo.config import Settings
from echo.grammar import GrammarChecker
from echo.transcriber import Transcriber
from echo.workers import AsyncJob

LANGUAGES: list[tuple[str, str | None]] = [
    ("Auto-detect", None),
    ("English", "en"),
    ("French", "fr"),
    ("Spanish", "es"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Dutch", "nl"),
    ("Arabic", "ar"),
    ("Turkish", "tr"),
    ("Russian", "ru"),
    ("Japanese", "ja"),
    ("Chinese", "zh"),
    ("Hindi", "hi"),
]


class EchoApp:
    def __init__(self, root: tk.Tk, settings: Settings) -> None:
        self.root = root
        self.settings = settings
        self.queue: queue.Queue = queue.Queue()

        self.recorder = Recorder(sample_rate=settings.sample_rate)
        self.transcriber = Transcriber(settings.whisper_model)
        self.checker: GrammarChecker | None = (
            GrammarChecker(settings.gemini_api_key, settings.gemini_model)
            if settings.has_api_key
            else None
        )

        self._busy = False
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.root.title("Project Echo")
        self.root.geometry("900x560")
        self.root.minsize(640, 400)

        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.pack(fill="x")
        self.record_btn = ttk.Button(toolbar, text="\u25cf  Record", command=self._toggle_record)
        self.record_btn.pack(side="left")
        ttk.Button(toolbar, text="Clear", command=self._clear).pack(side="left", padx=(6, 0))
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(toolbar, text="Copy raw", command=lambda: self._copy(self.raw_text)).pack(side="left")
        ttk.Button(toolbar, text="Copy corrected", command=lambda: self._copy(self.corrected_text)).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(toolbar, text="Save .txt", command=self._save).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="Re-check grammar", command=self._recheck).pack(side="left", padx=(6, 0))

        options = ttk.Frame(self.root, padding=(8, 2))
        options.pack(fill="x")
        ttk.Label(options, text="Mic:").pack(side="left")
        self.device_var = tk.StringVar()
        self.device_box = ttk.Combobox(
            options, textvariable=self.device_var, state="readonly", width=34
        )
        devices = self.recorder.list_devices()
        self.device_box["values"] = [f"{n}" for _, n in devices]
        if devices:
            default = self._default_device_index(devices)
            self.device_box.current(default)
            self.recorder.device = devices[default][0]
        self.device_box.pack(side="left", padx=(2, 14))
        self.device_box.bind("<<ComboboxSelected>>", self._on_device_change)

        ttk.Label(options, text="Language:").pack(side="left")
        self.language_var = tk.StringVar(value="Auto-detect")
        self.language_box = ttk.Combobox(
            options,
            textvariable=self.language_var,
            values=[label for label, _ in LANGUAGES],
            state="readonly",
            width=14,
        )
        self.language_box.pack(side="left", padx=(2, 0))

        if not self.settings.has_api_key:
            messagebox.showwarning(
                "No Gemini API key",
                "GEMINI_API_KEY is missing from .env. Grammar correction will be skipped;\n"
                "only transcription will work. Add the key and restart.",
            )

        panes = ttk.Frame(self.root, padding=(8, 6))
        panes.pack(fill="both", expand=True)
        panes.grid_rowconfigure(0, weight=1)
        panes.grid_columnconfigure(0, weight=1)
        panes.grid_columnconfigure(1, weight=1)

        self.raw_text = self._text_pane(panes, "Raw transcript")
        self.raw_text.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.corrected_text = self._text_pane(panes, "Corrected")
        self.corrected_text.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        self.status_var = tk.StringVar(value="Ready.")
        status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=(8, 2))
        status.pack(fill="x", side="bottom")

    def _text_pane(self, parent, title: str) -> tk.Text:
        frame = ttk.LabelFrame(parent, text=title, padding=(4, 4))
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        text = tk.Text(frame, wrap="word", font=("Segoe UI", 11), undo=True, relief="flat", borderwidth=1)
        text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scroll.set)
        return text

    def _default_device_index(self, devices: list[tuple[int, str]]) -> int:
        try:
            import sounddevice as sd

            default = sd.default.device
            if isinstance(default, (tuple, list)):
                default = default[0]
            for i, (idx, _name) in enumerate(devices):
                if idx == default:
                    return i
        except Exception:  # noqa: BLE001
            pass
        return 0

    # ------------------------------------------------------------- record
    def _toggle_record(self) -> None:
        if self.recorder.is_recording:
            self._stop_and_process()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if self._busy:
            return
        self.recorder.start()
        self.record_btn.configure(text="\u25a0  Stop")
        self._poll_duration()

    def _poll_duration(self) -> None:
        if not self.recorder.is_recording:
            return
        self.status_var.set(f"Recording\u2026 {self.recorder.duration_seconds:.1f}s")
        self.root.after(100, self._poll_duration)

    def _stop_and_process(self) -> None:
        self._busy = True
        self.record_btn.configure(text="\u25cf  Record", state="disabled")
        audio = self.recorder.stop()
        self.status_var.set("Transcribing\u2026")
        AsyncJob(self.queue, "transcribe", lambda: self._transcribe(audio)).start()

    def _transcribe(self, audio: np.ndarray) -> str:
        language = None
        for _label, code in LANGUAGES:
            if _label == self.language_var.get():
                language = code
                break
        return self.transcriber.transcribe(audio, language=language)

    def _recheck(self) -> None:
        raw = self.raw_text.get("1.0", "end").strip()
        if not raw:
            return
        if not self.checker:
            messagebox.showwarning("No API key", "Grammar correction needs GEMINI_API_KEY in .env")
            return
        self._busy = True
        self.record_btn.configure(state="disabled")
        self.status_var.set("Correcting\u2026")
        AsyncJob(self.queue, "correct", lambda: self.checker.correct(raw)).start()

    def _clear(self) -> None:
        self.raw_text.delete("1.0", "end")
        self.corrected_text.delete("1.0", "end")

    def _copy(self, text_widget: tk.Text) -> None:
        content = text_widget.get("1.0", "end").strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)

    def _save(self) -> None:
        raw = self.raw_text.get("1.0", "end").strip()
        corrected = self.corrected_text.get("1.0", "end").strip()
        if not raw and not corrected:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"echo_{stamp}.txt",
            filetypes=[("Text file", "*.txt")],
        )
        if not path:
            return
        lines = ["--- RAW TRANSCRIPT ---", raw, "", "--- CORRECTED ---", corrected, ""]
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        self.status_var.set(f"Saved: {path}")

    def _on_device_change(self, _event) -> None:
        devices = self.recorder.list_devices()
        idx = self.device_box.current()
        if 0 <= idx < len(devices):
            self.recorder.device = devices[idx][0]
            self.status_var.set(f"Mic: {devices[idx][1]}")

    # --------------------------------------------------------------- jobs
    def _poll(self) -> None:
        try:
            while True:
                kind, label, payload = self.queue.get_nowait()
                self._handle(kind, label, payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _handle(self, kind: str, label: str, payload: object) -> None:
        if label == "transcribe":
            if kind == "done":
                self._on_transcribed(str(payload))
            else:
                self._fail(str(payload))
        elif label == "correct":
            if kind == "done":
                self._on_corrected(str(payload))
            else:
                self._fail(str(payload))

    def _on_transcribed(self, text: str) -> None:
        self.raw_text.delete("1.0", "end")
        self.raw_text.insert("1.0", text)
        if not text.strip():
            self._finish_busy("No speech detected.")
            return
        if not self.checker:
            self._finish_busy("Transcription complete (correction skipped \u2014 no API key).")
            return
        self.status_var.set("Correcting\u2026")
        AsyncJob(self.queue, "correct", lambda: self.checker.correct(text)).start()

    def _on_corrected(self, text: str) -> None:
        self.corrected_text.delete("1.0", "end")
        self.corrected_text.insert("1.0", text)
        self._finish_busy("Ready.")

    def _fail(self, message: str) -> None:
        self._finish_busy(f"Error: {message}")

    def _finish_busy(self, status: str) -> None:
        self._busy = False
        self.record_btn.configure(state="normal")
        self.status_var.set(status)

    def _on_close(self) -> None:
        if self.recorder.is_recording:
            self.recorder.stop()
        self.root.destroy()