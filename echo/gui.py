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

# ---- palette ----------------------------------------------------------
BG = "#17181d"
PANEL = "#1f2128"
PANEL2 = "#282b34"
BORDER = "#363945"
TEXT = "#ececf1"
MUTED = "#8b8f9c"
ACCENT = "#5b8cff"
REC = "#e5484d"
OK = "#3ecf8e"
BAD = "#ff8b8b"

SPIN = ["\u25d0", "\u25d3", "\u25d1", "\u25d2"]
FONT = ("Segoe UI", 10)
FONT_SM = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 11)


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
        self._spin = 0
        self._status = "Ready"
        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<space>", self._on_space)
        self.root.bind("<Control-s>", lambda _e: self._save())
        self.root.bind("<Control-r>", lambda _e: self._recheck())
        self.root.after(80, self._tick)
        self.root.after(60, self._poll)

    # --------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        self.root.title("Echo")
        self.root.configure(bg=BG)
        self.root.geometry("960x620")
        self.root.minsize(780, 500)

        self._style_ttk()

        # status + actions pinned to the bottom
        self.status_bar = tk.Label(
            self.root, text="space  record/stop  \u00b7  ctrl+S save  \u00b7  ctrl+R re-check",
            bg=BG, fg=MUTED, font=FONT_SM, anchor="w",
        )
        self.status_bar.pack(side="bottom", fill="x", padx=18, pady=(0, 8))

        actions = tk.Frame(self.root, bg=BG)
        actions.pack(side="bottom", fill="x", padx=18, pady=(0, 6))
        self._flat_btn(actions, "Copy raw", lambda: self._copy(self.raw_text)).pack(side="left")
        self._flat_btn(actions, "Copy corrected", lambda: self._copy(self.corrected_text)).pack(
            side="left", padx=(6, 0)
        )
        self._flat_btn(actions, "Save", self._save).pack(side="left", padx=(6, 0))
        self._flat_btn(actions, "Re-check grammar", self._recheck).pack(side="left", padx=(6, 0))
        self._flat_btn(actions, "Clear", self._clear).pack(side="right")

        # header
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=18, pady=(16, 2))
        tk.Label(header, text="Echo", bg=BG, fg=ACCENT, font=("Segoe UI", 22, "bold")).pack(side="left")
        tk.Label(header, text="speech  \u2192  text  \u2192  grammar", bg=BG, fg=MUTED,
                 font=FONT_SM).pack(side="left", padx=(12, 0), pady=(7, 0))
        api = OK if self.settings.has_api_key else BAD
        dot = tk.Frame(header, bg=api, width=8, height=8)
        dot.pack(side="right", padx=(0, 4), pady=(10, 0))
        tk.Label(header, text="Gemini" if self.settings.has_api_key else "no API key",
                 bg=BG, fg=MUTED, font=FONT_SM).pack(side="right", pady=(6, 0))

        # control center
        center = tk.Frame(self.root, bg=BG)
        center.pack(fill="x", pady=(10, 6))
        self._build_record_control(center)

        # settings strip
        strip = tk.Frame(self.root, bg=BG)
        strip.pack(fill="x", padx=18, pady=(0, 8))
        ttk.Label(strip, text="Mic", style="Dark.TLabel").pack(side="left")
        self.device_var = tk.StringVar()
        self.device_box = ttk.Combobox(strip, textvariable=self.device_var, state="readonly",
                                       style="Dark.TCombobox", width=36)
        devices = self.recorder.list_devices()
        self.device_box["values"] = [n for _, n in devices]
        if devices:
            default = self._default_device_index(devices)
            self.device_box.current(default)
            self.recorder.device = devices[default][0]
        self.device_box.pack(side="left", padx=(6, 18))
        self.device_box.bind("<<ComboboxSelected>>", self._on_device_change)

        ttk.Label(strip, text="Language", style="Dark.TLabel").pack(side="left")
        self.language_var = tk.StringVar(value="Auto-detect")
        self.language_box = ttk.Combobox(
            strip, textvariable=self.language_var, values=[lbl for lbl, _ in LANGUAGES],
            state="readonly", style="Dark.TCombobox", width=14,
        )
        self.language_box.pack(side="left", padx=(6, 0))

        # transcript panes
        panes = tk.Frame(self.root, bg=BG)
        panes.pack(fill="both", expand=True, padx=18, pady=(4, 10))
        panes.grid_rowconfigure(0, weight=1)
        panes.grid_columnconfigure(0, weight=1, uniform="pane")
        panes.grid_columnconfigure(1, weight=1, uniform="pane")
        self.raw_frame = self._pane(panes, "Raw transcript")
        self.raw_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.raw_text = self.raw_frame.text
        self.corrected_frame = self._pane(panes, "Corrected")
        self.corrected_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.corrected_text = self.corrected_frame.text

        if not self.settings.has_api_key:
            self._set_status("No Gemini API key \u2014 only transcription will run.")

    def _build_record_control(self, parent: tk.Frame) -> None:
        box = tk.Frame(parent, bg=BG)
        box.pack()
        self.record_canvas = tk.Canvas(box, width=150, height=150, bg=BG, highlightthickness=0)
        self.record_canvas.pack()
        self.record_canvas.bind("<Button-1>", lambda _e: self._toggle_record())
        self.record_canvas.bind("<Enter>", lambda _e: self._draw_record())
        self.record_canvas.bind("<Leave>", lambda _e: self._draw_record())

        # live input meter
        meter_box = tk.Frame(box, bg=BG)
        meter_box.pack(pady=(10, 0))
        self.meter = tk.Canvas(meter_box, width=260, height=6, bg=BG, highlightthickness=0)
        self.meter.pack()
        self.meter.create_rectangle(0, 0, 260, 6, fill=BORDER, outline="")
        self.meter_fill = self.meter.create_rectangle(0, 0, 0, 6, fill=ACCENT, outline="")

        self.control_status = tk.Label(box, text="", bg=BG, fg=MUTED, font=FONT_SM)
        self.control_status.pack(pady=(6, 0))

        self._draw_record()

    def _draw_record(self) -> None:
        c = self.record_canvas
        c.delete("all")
        active = self.recorder.is_recording
        hover = bool(c.winfo_containing(c.winfo_pointerx(), c.winfo_pointery()))
        if active:
            fill, outline, fg, label = REC, REC, "white", "Stop"
        elif hover and not self._busy:
            fill, outline, fg, label = "#2c3142", ACCENT, TEXT, "Record"
        else:
            fill, outline, fg, label = "#242834", ACCENT, TEXT, "Record"
        c.create_oval(10, 10, 140, 140, fill=fill, outline=outline, width=3)
        c.create_text(75, 60, text="\u25cf" if not active else "\u25a0",
                      fill=fg, font=("Segoe UI", 26))
        c.create_text(75, 96, text=label, fill=fg, font=("Segoe UI", 12, "bold"))
        if self._busy and not active:
            alpha = REC if hover else MUTED
            c.create_oval(63, 22, 87, 46, outline=alpha, dash=(2, 2))

    def _pane(self, parent: tk.Frame, title: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        tk.Label(frame, text=title.upper(), bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold"),
                 anchor="w").grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 4))
        text = tk.Text(
            frame, wrap="word", font=FONT_MONO, bg=PANEL2, fg=TEXT, insertbackground=TEXT,
            relief="flat", highlightthickness=0, padx=12, pady=10, undo=True,
            selectbackground=ACCENT, selectforeground="white",
        )
        text.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        text.configure(yscrollcommand=scroll.set)
        frame.text = text
        return frame

    def _flat_btn(self, parent: tk.Frame, label: str, command) -> tk.Button:
        btn = tk.Button(
            parent, text=label, command=command, bg=PANEL2, fg=TEXT, font=("Segoe UI", 9, "bold"),
            activebackground=ACCENT, activeforeground="white", relief="flat", bd=0,
            padx=14, pady=5, cursor="hand2",
        )
        btn.bind("<Enter>", lambda _e: btn.configure(bg="#323644"))
        btn.bind("<Leave>", lambda _e: btn.configure(bg=PANEL2))
        return btn

    def _style_ttk(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Dark.TLabel", background=BG, foreground=MUTED, font=FONT_SM)
        style.configure(
            "Dark.TCombobox", fieldbackground=PANEL2, background=PANEL2, foreground=TEXT,
            arrowcolor=TEXT, bordercolor=BORDER, lightcolor=PANEL2, darkcolor=PANEL2,
            selectbackground=PANEL2, selectforeground=TEXT, padding=4, font=FONT_SM,
        )
        style.map("Dark.TCombobox", fieldbackground=[("readonly", PANEL2)])

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

    # ------------------------------------------------------------ record
    def _toggle_record(self) -> None:
        if self.recorder.is_recording:
            self._stop_and_process()
        elif not self._busy:
            self._start_recording()

    def _start_recording(self) -> None:
        self.recorder.start()
        self._draw_record()
        self._set_status("Listening\u2026")

    def _stop_and_process(self) -> None:
        self._busy = True
        self._draw_record()
        audio = self.recorder.stop()
        self._meter_level(0.0)
        if audio.size == 0:
            self._finish_busy("Nothing recorded.")
            return
        self._set_status("Transcribing\u2026")
        AsyncJob(self.queue, "transcribe", lambda: self._transcribe(audio)).start()

    def _transcribe(self, audio: np.ndarray) -> str:
        language = None
        for _label, code in LANGUAGES:
            if _label == self.language_var.get():
                language = code
                break
        return self.transcriber.transcribe(audio, language=language)

    def _meter_level(self, level: float) -> None:
        width = max(0, min(1.0, level)) * 260
        self.meter.coords(self.meter_fill, 0, 0, width, 6)

    # ----------------------------------------------------------- actions
    def _recheck(self) -> None:
        raw = self.raw_text.get("1.0", "end").strip()
        if not raw:
            return
        if not self.checker:
            messagebox.showwarning("No API key", "Grammar correction needs GEMINI_API_KEY in .env")
            return
        self._busy = True
        self._draw_record()
        self._set_status("Correcting\u2026")
        AsyncJob(self.queue, "correct", lambda: self.checker.correct(raw)).start()

    def _clear(self) -> None:
        self.raw_text.delete("1.0", "end")
        self.corrected_text.delete("1.0", "end")

    def _copy(self, widget: tk.Text) -> None:
        content = widget.get("1.0", "end").strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self._set_status("Copied to clipboard.")

    def _save(self) -> None:
        raw = self.raw_text.get("1.0", "end").strip()
        corrected = self.corrected_text.get("1.0", "end").strip()
        if not raw and not corrected:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", initialfile=f"echo_{stamp}.txt", filetypes=[("Text file", "*.txt")]
        )
        if not path:
            return
        Path(path).write_text("\n".join(["--- RAW TRANSCRIPT ---", raw, "", "--- CORRECTED ---", corrected, ""]),
                              encoding="utf-8")
        self._set_status(f"Saved: {path}")

    def _on_device_change(self, _event) -> None:
        devices = self.recorder.list_devices()
        idx = self.device_box.current()
        if 0 <= idx < len(devices):
            self.recorder.device = devices[idx][0]

    # ------------------------------------------------------- status loop
    def _set_status(self, text: str) -> None:
        self._status = text
        self._render_status()

    def _render_status(self) -> None:
        label = self._status
        if self._busy:
            label = f"{label} {SPIN[self._spin % len(SPIN)]}"
        self.control_status.configure(text=label)

    def _tick(self) -> None:
        if self.recorder.is_recording:
            self._meter_level(self.recorder.level)
            self._set_status(f"Listening\u2026 {self.recorder.duration_seconds:.1f}s")
        elif self._busy:
            self._spin += 1
            self._render_status()
        else:
            self._meter_level(0.0)
        self.root.after(80, self._tick)

    def _on_space(self, _event) -> str | None:
        if isinstance(self.root.focus_get(), tk.Text):
            return None  # typing in a pane
        self._toggle_record()
        return "break"

    # -------------------------------------------------------------- jobs
    def _poll(self) -> None:
        try:
            while True:
                kind, label, payload = self.queue.get_nowait()
                self._handle(kind, label, payload)
        except queue.Empty:
            pass
        self.root.after(60, self._poll)

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
        self._set_status("Correcting\u2026")
        AsyncJob(self.queue, "correct", lambda: self.checker.correct(text)).start()

    def _on_corrected(self, text: str) -> None:
        self.corrected_text.delete("1.0", "end")
        self.corrected_text.insert("1.0", text)
        self._finish_busy("Ready.")

    def _fail(self, message: str) -> None:
        self._finish_busy(f"Error: {message}")

    def _finish_busy(self, status: str) -> None:
        self._busy = False
        self._spin = 0
        self._draw_record()
        self._set_status(status)

    def _on_close(self) -> None:
        if self.recorder.is_recording:
            self.recorder.stop()
        self.root.destroy()