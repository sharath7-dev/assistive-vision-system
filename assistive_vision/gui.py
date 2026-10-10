"""Tkinter dashboard. Only this module touches widgets, on the main thread."""

import tkinter as tk
import sys
from tkinter import ttk
from time import monotonic

from PIL import Image, ImageTk

from .desktop_logic import DesktopController, preview_size
from .diagnostics import DebugReporter
from .pipeline import FrameResult, VisionPipeline

BACKGROUND = "#0d141f"
CARD = "#172231"
SURFACE = "#1e2d40"
TEXT = "#edf3fa"
MUTED = "#a8b8cb"
ACCENT = "#62d4c2"
ERROR = "#ffb4ab"


class AssistiveVisionApp:
    def __init__(self, root: tk.Tk, pipeline: VisionPipeline, *, confidence: float = 0.5,
                 muted: bool = False, debug: bool = False) -> None:
        self.root = root
        self.controller = DesktopController(pipeline, muted=muted)
        self.pipeline = pipeline
        self.reporter = DebugReporter(debug)
        self._last_sequence = 0
        self._render_key: tuple[int, int, int] | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._message = ""
        self._tick_id: str | None = None
        self._progress_active = False
        self._last_warning = ""
        self.confidence = tk.DoubleVar(root, value=confidence)
        self.confidence_text = tk.StringVar(root, value=f"{confidence:.2f}")
        self.banner = tk.StringVar(root, value="Ready when you are. Start the camera to begin.")
        self.speech_text = tk.StringVar(root, value="Speech not started")
        self.metrics = {name: tk.StringVar(root, value=value) for name, value in
                        (("FPS", "—"), ("CAMERA", "Closed"),
                         ("DETECTION", "Idle"), ("MODEL", "Not loaded"))}
        root.title("Assistive Vision | Real-Time Object Awareness")
        root.geometry("1240x900")
        root.minsize(1000, 860)
        root.configure(bg=BACKGROUND)
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._styles()
        self._build()
        root.bind("<Control-Return>", lambda event: self._toggle_camera())
        root.bind("<Key-p>", lambda event: self._pause())
        root.bind("<Key-m>", lambda event: self._mute())
        root.bind("<Escape>", lambda event: self._close())
        root.bind("<Control-q>", lambda event: self._close())
        self.start_button.focus_set()
        self._tick()

    def _styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background=BACKGROUND,
                        foreground=TEXT, bordercolor=SURFACE, focuscolor=ACCENT)
        style.configure("TFrame", background=BACKGROUND)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BACKGROUND, foreground=TEXT)
        style.configure("Card.TLabel", background=CARD)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Small.Card.TLabel", background=CARD, foreground=MUTED,
                        font=("Segoe UI", 9))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 27))
        style.configure("Section.Card.TLabel", background=CARD, font=("Segoe UI Semibold", 12))
        style.configure("Metric.Card.TLabel", background=CARD, font=("Segoe UI Semibold", 15))
        style.configure("TButton", background=SURFACE, foreground=TEXT,
                        padding=(14, 11), borderwidth=1, relief="flat")
        style.map("TButton", background=[("disabled", CARD), ("active", "#30445c")],
                  foreground=[("disabled", "#738399")],
                  bordercolor=[("focus", ACCENT), ("!focus", SURFACE)])
        style.configure("Accent.TButton", background=ACCENT, foreground="#092d29",
                        font=("Segoe UI Semibold", 11))
        style.map("Accent.TButton", background=[("disabled", SURFACE), ("active", "#91e4d7")],
                  foreground=[("disabled", MUTED)])
        style.configure("Horizontal.TScale", background=CARD, troughcolor=SURFACE)
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=SURFACE,
                        borderwidth=0, thickness=3)
        style.configure("Vertical.TScrollbar", background=SURFACE, troughcolor=CARD,
                        arrowcolor=MUTED, bordercolor=CARD)
        style.configure("Treeview", background=CARD, fieldbackground=CARD,
                        foreground=TEXT, rowheight=29, borderwidth=0)
        style.map("Treeview", background=[("selected", "#294d51")])
        style.configure("Treeview.Heading", background=CARD, foreground=MUTED,
                        font=("Segoe UI", 9), relief="flat")

    def _card(self, parent: ttk.Frame) -> ttk.Frame:
        return ttk.Frame(parent, style="Card.TFrame", padding=14)

    def _build(self) -> None:
        shell = ttk.Frame(self.root, padding=(26, 22))
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)
        header = ttk.Frame(shell)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 22))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="ASSISTIVE VISION", foreground=ACCENT,
                  font=("Segoe UI Semibold", 10)).grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="See the scene. Hear what’s there.",
                  style="Title.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 4))
        ttk.Label(header, text="Real-time object awareness through visual labels and spoken cues.",
                  style="Muted.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Label(header, text="IPCV  /  DESKTOP PROTOTYPE", style="Muted.TLabel",
                  font=("Segoe UI", 9)).grid(row=0, column=1, sticky="e")

        body = ttk.Frame(shell)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, minsize=320)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        preview = self._card(left)
        preview.grid(row=0, column=0, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(1, weight=1)
        top = ttk.Frame(preview, style="Card.TFrame")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        ttk.Label(top, text="Live camera", style="Section.Card.TLabel").pack(side="left")
        ttk.Label(top, text="BOXES · LABELS · CONFIDENCE",
                  style="Small.Card.TLabel").pack(side="right")
        self.canvas = tk.Canvas(preview, bg="#080f18", highlightthickness=0,
                                width=560, height=360, takefocus=False)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self._image_id = self.canvas.create_image(0, 0, anchor="center", state="hidden")
        self._placeholder = self.canvas.create_text(0, 0, text="", fill=MUTED,
                                                    font=("Segoe UI", 14), justify="center")
        self.canvas.bind("<Configure>", self._resize_preview)
        self.progress = ttk.Progressbar(preview, mode="indeterminate")
        self.progress.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        self.status_label = ttk.Label(preview, textvariable=self.banner,
                                      style="Small.Card.TLabel", wraplength=600)
        self.status_label.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        strip = ttk.Frame(left)
        strip.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        for index, (name, variable) in enumerate(self.metrics.items()):
            strip.columnconfigure(index, weight=1, uniform="metrics")
            metric = ttk.Frame(strip, style="Card.TFrame", padding=(14, 12))
            metric.grid(row=0, column=index, sticky="ew", padx=(0, 8 if index < 3 else 0))
            ttk.Label(metric, text=name, style="Small.Card.TLabel").pack(anchor="w")
            ttk.Label(metric, textvariable=variable, style="Metric.Card.TLabel").pack(anchor="w", pady=(5, 0))

        sidebar = ttk.Frame(body, width=320)
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)
        controls = self._card(sidebar)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Session controls", style="Section.Card.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))
        self.start_button = ttk.Button(controls, text="Start camera", style="Accent.TButton", command=self._toggle_camera)
        self.start_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.pause_button = ttk.Button(controls, text="Pause detection", command=self._pause)
        self.pause_button.grid(row=2, column=0, sticky="ew", padx=(0, 4))
        self.mute_button = ttk.Button(controls, text="Unmute speech" if self.controller.muted else "Mute speech", command=self._mute)
        self.mute_button.grid(row=2, column=1, sticky="ew", padx=(4, 0))
        settings = self._card(sidebar)
        settings.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        settings.columnconfigure(0, weight=1)
        ttk.Label(settings, text="Confidence threshold", style="Section.Card.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(settings, textvariable=self.confidence_text, foreground=ACCENT,
                  style="Card.TLabel", font=("Segoe UI Semibold", 15)).grid(row=0, column=1, padx=(8, 0))
        self.scale = ttk.Scale(settings, from_=0, to=1, variable=self.confidence,
                               command=self._confidence_changed, takefocus=True)
        self.scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(14, 8))
        self.scale.bind("<Left>", lambda event: self._step_confidence(-0.01))
        self.scale.bind("<Right>", lambda event: self._step_confidence(0.01))
        ttk.Label(settings, text="Applies to detection and speech.\nHigher scores do not guarantee accuracy.",
                  style="Small.Card.TLabel").grid(row=2, column=0, columnspan=2, sticky="w")
        recent = ttk.Frame(sidebar, style="Card.TFrame", padding=12)
        recent.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(recent, text="Recently seen", style="Section.Card.TLabel").pack(anchor="w")
        ttk.Label(recent, text="Last 10 seconds · up to 6 object classes", style="Small.Card.TLabel").pack(anchor="w", pady=(4, 10))
        table = ttk.Frame(recent, style="Card.TFrame")
        table.pack(fill="x")
        self.tree = ttk.Treeview(table, columns=("label", "score"), show="headings", height=4, selectmode="browse")
        self.tree.heading("label", text="Object")
        self.tree.heading("score", text="Score")
        self.tree.column("label", width=150, minwidth=100, stretch=True)
        self.tree.column("score", width=65, minwidth=55, stretch=False, anchor="e")
        self.tree.pack(side="left", fill="x", expand=True)
        scrollbar = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.recent_empty = ttk.Label(recent, text="No recent detections", style="Small.Card.TLabel")
        self.recent_empty.pack(anchor="w", pady=(8, 0))
        self.speech_label = ttk.Label(sidebar, textvariable=self.speech_text,
                                     style="Muted.TLabel", wraplength=285)
        self.speech_label.grid(row=3, column=0, sticky="w", pady=(10, 0))

        footer = ttk.Frame(shell)
        footer.grid(row=2, column=0, sticky="ew", pady=(20, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, text="Ctrl+Enter  Start / stop     P  Pause     M  Mute     Esc  Quit",
                  style="Muted.TLabel", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        ttk.Label(footer, text="Real-Time Assistive Vision System for the Visually Impaired\n"
                  "College prototype · No distance measurement or safe-navigation guarantee.",
                  style="Muted.TLabel", font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", pady=(7, 0))
        self.quit_button = ttk.Button(footer, text="Quit application", command=self._close)
        self.quit_button.grid(row=0, column=1, rowspan=2, sticky="e", padx=(16, 0))

    def _toggle_camera(self) -> None:
        if self.controller.stopping or self.controller.closing:
            return
        if self.pipeline.workers_running:
            self.controller.stop()
        elif self.controller.start():
            self._last_sequence = 0
            self._render_key = None

    def _pause(self) -> None:
        self.controller.pause()

    def _mute(self) -> None:
        if not self.controller.closing:
            self.controller.mute()

    def _confidence_changed(self, value: str) -> None:
        confidence = round(float(value), 2)
        self.confidence_text.set(f"{confidence:.2f}")
        self.pipeline.set_confidence(confidence)

    def _step_confidence(self, step: float) -> str:
        self.confidence.set(min(1.0, max(0.0, round(self.confidence.get() + step, 2))))
        self._confidence_changed(str(self.confidence.get()))
        return "break"

    def _close(self) -> None:
        self.controller.close()

    def _resize_preview(self, event: tk.Event) -> None:
        self._render_key = None
        self.status_label.configure(wraplength=max(200, event.width))
        self.canvas.coords(self._placeholder, event.width / 2, event.height / 2)
        self.canvas.itemconfigure(self._placeholder, width=max(100, event.width - 60))

    def _show_preview(self, result: FrameResult | None, message: str) -> None:
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if result is None:
            self.canvas.itemconfigure(self._image_id, state="hidden")
            self.canvas.itemconfigure(self._placeholder, text=message, state="normal")
            self._photo = None
            self._render_key = None
            return
        self.canvas.itemconfigure(self._placeholder, state="hidden")
        key = (result.sequence, width, height)
        if key == self._render_key or min(width, height) < 2:
            return
        self._render_key = key
        frame = result.frame
        fitted = preview_size(frame.shape[1], frame.shape[0], width, height)
        image = Image.fromarray(frame[:, :, ::-1]).resize(fitted, Image.Resampling.BILINEAR)
        self._photo = ImageTk.PhotoImage(image, master=self.root)
        self.canvas.coords(self._image_id, width / 2, height / 2)
        self.canvas.itemconfigure(self._image_id, image=self._photo, state="normal")

    def _update_recent(self, now: float) -> None:
        rows = self.controller.recent.rows(now)
        wanted = {item.label for item in rows}
        for iid in self.tree.get_children():
            if iid not in wanted:
                self.tree.delete(iid)
        for index, item in enumerate(rows):
            values = (item.label.title(), f"{item.confidence:.0%}")
            if self.tree.exists(item.label):
                if tuple(self.tree.item(item.label, "values")) != values:
                    self.tree.item(item.label, values=values)
                self.tree.move(item.label, "", index)
            else:
                self.tree.insert("", index, iid=item.label, values=values)
        self.recent_empty.configure(text="" if rows else "No recent detections")

    def _tick(self) -> None:
        if self.controller.poll_shutdown():
            self.root.destroy()
            return
        state = self.pipeline.state()
        speech_status = self.pipeline.speech.status
        running = self.pipeline.running
        workers = self.pipeline.workers_running
        result = self.pipeline.latest() if running and not self.controller.stopping else None
        now = monotonic()
        if result is not None and result.sequence != self._last_sequence:
            self._last_sequence = result.sequence
            self.controller.recent.update(result.detections, result.observed_at)
        self._update_recent(now)
        loading = workers and (state.camera == "Opening" or state.model == "Loading")
        waiting = self.controller.stopping or self.controller.closing
        if waiting:
            message = "Stopping safely…\nWaiting for camera, model, and speech workers."
            banner = "Finishing current work and releasing the camera. The window remains responsive."
        elif "unavailable" in self.pipeline.status or "failed" in self.pipeline.status:
            message, banner = "Session unavailable\nReview the message below, then retry.", self.pipeline.status
        elif loading:
            message = "Opening camera…" if state.camera == "Opening" else "Loading detection model…\nFirst use may download YOLO11n weights."
            banner = self.pipeline.status
        elif not workers:
            message, banner = "Camera is off\nStart the camera to explore your scene.", "Ready. Camera and speech are stopped."
        else:
            message, banner = "Waiting for a camera frame…", self.pipeline.status
        if not waiting and "unavailable" in speech_status and running:
            banner += " · " + speech_status
        if banner != self._message:
            self._message = banner
            self.banner.set(banner)
            self.status_label.configure(foreground=ERROR if "unavailable" in banner or "failed" in banner else MUTED)
        active_progress = loading or waiting
        if active_progress != self._progress_active:
            self._progress_active = active_progress
            self.progress.start(15) if active_progress else self.progress.stop()
        self._show_preview(result, message)
        self.metrics["FPS"].set(f"{result.fps:.1f}" if result is not None else "—")
        self.metrics["CAMERA"].set(state.camera)
        self.metrics["DETECTION"].set(state.detection)
        self.metrics["MODEL"].set(state.model)
        self.speech_text.set("Speech unavailable" if "unavailable" in speech_status else speech_status)
        if "unavailable" in speech_status and speech_status != self._last_warning:
            self._last_warning = speech_status
            print(speech_status, file=sys.stderr)
        self.start_button.configure(text="Stopping…" if waiting else "Stop camera" if workers else "Start camera")
        self.start_button.state(["disabled"] if waiting or (workers and not running) else ["!disabled"])
        self.pause_button.configure(text="Resume detection" if self.controller.paused else "Pause detection")
        self.pause_button.state(["!disabled"] if running and not waiting else ["disabled"])
        self.mute_button.configure(text="Unmute speech" if self.controller.muted else "Mute speech")
        self.mute_button.state(["disabled"] if self.controller.closing else ["!disabled"])
        self.scale.state(["disabled"] if waiting else ["!disabled"])
        self.quit_button.state(["disabled"] if self.controller.closing else ["!disabled"])
        self.reporter.report(result, status=self.pipeline.status, speech_status=speech_status,
                             last_request=self.pipeline.speech.last_request)
        self._tick_id = self.root.after(33, self._tick)


def launch_gui(pipeline: VisionPipeline, *, confidence: float = 0.5,
               muted: bool = False, debug: bool = False) -> int:
    root = tk.Tk()
    app = AssistiveVisionApp(root, pipeline, confidence=confidence, muted=muted, debug=debug)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app._close()
        root.mainloop()
    return 0
