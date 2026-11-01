"""SAFEBOX AI: dashboard local de control de acceso biométrico."""

from __future__ import annotations

import argparse
from collections import deque
import queue
import re
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from tkinter import Canvas, TclError, filedialog

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from core import (DEFAULT_SIMILARITY_THRESHOLD, ENROLL_SAMPLES, SafeBackend,
                  check_data_integrity, detect_faces, load_models, storage_sizes,
                  verify_local_backup)
from control import AdminManager, SettingsStore
from icons import IconProvider


BASE = Path(__file__).resolve().parent
SAMPLES_REQUIRED = ENROLL_SAMPLES

BG = "#121212"
SURFACE = "#181818"
SURFACE_2 = "#252525"
BORDER = "#383838"
CARD_BORDER = "#2D352F"
GREEN_DIM = "#1B3224"
TEXT = "#FFFFFF"
MUTED = "#B3B3B3"
CYAN = "#1ED760"
BLUE = "#539DF5"
RED = "#F3727F"
AMBER = "#FFA42B"

THEMES = {
    "Verde tecnológico": {"BG": "#121212", "SURFACE": "#181818", "SURFACE_2": "#252525",
                          "BORDER": "#383838", "CARD_BORDER": "#2D352F", "GREEN_DIM": "#1B3224",
                          "CYAN": "#1ED760", "MUTED": "#B3B3B3", "TEXT": "#FFFFFF"},
    "Azul cibernético": {"BG": "#0C1420", "SURFACE": "#142133", "SURFACE_2": "#203249",
                        "BORDER": "#38516B", "CARD_BORDER": "#28445D", "GREEN_DIM": "#173C54",
                        "CYAN": "#66C7FF", "MUTED": "#B7C7D6", "TEXT": "#FFFFFF"},
    "Minimalista oscuro": {"BG": "#101112", "SURFACE": "#1A1C1E", "SURFACE_2": "#292C2F",
                           "BORDER": "#45494D", "CARD_BORDER": "#36393C", "GREEN_DIM": "#303536",
                           "CYAN": "#E3E8E6", "MUTED": "#BFC4C2", "TEXT": "#FFFFFF"},
}

STATE_COLORS = {
    "INICIANDO": BLUE,
    "ESCANEANDO": MUTED,
    "VERIFICANDO": BLUE,
    "PIN REQUERIDO": AMBER,
    "REGISTRANDO": BLUE,
    "AUTORIZADO": CYAN,
    "DENEGADO": RED,
    "CÁMARA DESCONECTADA": AMBER,
    "ERROR DE MODELO": RED,
}


def mix_color(start: str, end: str, amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    a = [int(start[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(end[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * amount):02X}" for x, y in zip(a, b))


def play_welcome_sound() -> None:
    """Melodía breve en Windows; cualquier fallo de audio es no crítico."""
    try:
        import winsound
        for frequency in (784, 988, 1175):
            winsound.Beep(frequency, 75)
    except Exception:
        pass


def save_graphs_png(path: Path, stats: tuple[int, int, int],
                    daily: list[tuple[str, int, int]],
                    hourly: list[tuple[str, int, int]],
                    accent: str = "#1ED760") -> None:
    """Exporta gráficos reales con Pillow, ya instalado para la cámara."""
    total, allowed, denied = stats
    picture = Image.new("RGB", (1050, 430), "#121212")
    draw = ImageDraw.Draw(picture)
    draw.text((28, 20), "SAFEBOX AI · ACTIVIDAD", fill="white")
    draw.text((28, 56), f"Intentos: {total}   Autorizados: {allowed}   Denegados: {denied}",
              fill="#B3B3B3")
    draw.rectangle((28, 87, 420, 109), fill="#252525")
    if total:
        a_width = round(392 * allowed / total)
        d_width = round(392 * denied / total)
        draw.rectangle((28, 87, 28 + a_width, 109), fill=accent)
        draw.rectangle((28 + a_width, 87, 28 + a_width + d_width, 109), fill=RED)
        draw.text((28, 121), f"{allowed / total:.1%} autorizados · {denied / total:.1%} denegados",
                  fill="#B3B3B3")
    else:
        draw.text((28, 121), "Sin intentos registrados", fill="#B3B3B3")
    draw.text((28, 172), "ACTIVIDAD DIARIA", fill="white")
    recent = daily[-14:]
    maximum = max([a + d for _, a, d in recent], default=1) or 1
    for index, (day, a, d) in enumerate(recent):
        x = 28 + index * 70
        h_a, h_d = round(110 * a / maximum), round(110 * d / maximum)
        draw.rectangle((x, 310 - h_a, x + 43, 310), fill=accent)
        draw.rectangle((x, 310 - h_a - h_d, x + 43, 310 - h_a), fill=RED)
        draw.text((x, 320), day[5:], fill="#B3B3B3")
    draw.text((28, 362), "FRANJAS HORARIAS · últimos 7 días en el historial", fill="#B3B3B3")
    counts = [0] * 24
    for _, hour, count in hourly:
        if 0 <= hour < 24:
            counts[hour] += count
    intensity = max(counts, default=0) or 1
    for hour, count in enumerate(counts):
        color = mix_color("#252525", accent, count / intensity)
        x = 28 + hour * 41
        draw.rectangle((x, 388, x + 34, 409), fill=color)
        if hour % 4 == 0:
            draw.text((x, 413), f"{hour:02d}", fill="#B3B3B3")
    picture.save(path, format="PNG")


class SafeApp(ctk.CTk):
    def __init__(self, data_dir: Path, mode: str, port: str | None,
                 similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                 camera_index: int = 0, authorization_seconds: int = 8,
                 settings: SettingsStore | None = None,
                 led_port: str | None = None):
        super().__init__(fg_color=BG)
        self.title("SAFEBOX AI · Dashboard biométrico")
        self.geometry("1220x720+40+30")
        self.minsize(980, 650)

        self.settings = settings or SettingsStore(data_dir)
        self.icons = IconProvider()
        self.icon_bindings: dict[object, tuple[str, str, int]] = {}
        self.icon_displayed: dict[object, tuple[str, int, str]] = {}
        self.icons_ready = False
        self.icon_job = None
        self.admin = AdminManager(data_dir)
        self.admin_results: queue.Queue = queue.Queue()
        self.admin_until = 0.0
        self.admin_window = None
        self.pending_settings = None
        self.led_port = led_port
        self.led_connected = False
        self.device_connected = False
        self.led_error_seen = False
        self.led_error_message: str | None = None
        self.servo_status = "NO CONECTADO"
        self.servo_pulses: tuple[int, int] | None = None
        self.backend = SafeBackend(data_dir, mode, port, similarity_threshold,
                                   camera_index, authorization_seconds,
                                   led_port=led_port)
        self.backend.send("admin_ready", self.admin.configured)
        self.similarity_threshold = similarity_threshold
        self.camera_index = camera_index
        self.authorization_seconds = authorization_seconds
        self.lock_state = "BLOQUEADA" if mode == "simulacion" else "ERROR"
        self.physical_unknown = mode == "hardware"
        self.mode = mode
        self.running = True
        self.camera_connected = False
        self.camera_paused = False
        self.preview_last_at = 0.0
        self.preview_interval = 1 / 20
        self.backend_ready = False
        self.known_users: list[str] = []
        self.people: list[tuple[str, str, bool]] = []
        self.access_pin_ids: set[str] = set()
        self.access_window = None
        self.access_request = None
        self.access_track = None
        self.access_deadline = 0.0
        self.pin_tick_job = None
        self.history_rows: list[tuple] = []
        self._rendered_recent = None
        self._rendered_users = None
        self.history_results: list[tuple] = []
        self.history_offset = 0
        self.current_page = "Inicio"
        self.boxes: list[tuple[int, int, int, int]] = []
        self.face_labels: list[tuple[str, str]] = []
        self.last_frame_seq = 0
        self.status_state = ""
        self.status_color = CYAN
        self.blue_led_until = 0.0
        self.led_off_job = None
        self.last_lock_notice = None
        self.transition_job = None
        self.backend_start_job = None
        self.poll_job = None
        self.clock_job = None
        self.modal_close_job = None
        self.register_window: ctk.CTkToplevel | None = None
        self.photo = None
        self.photo_size = None
        self.last_canvas_size = None
        self.hud_visible = False
        self.scan_line_visible = False
        self.preview_photo = None
        self.search_job = None
        self.notice_clear_job = None
        self.last_notice = ("", 0.0)
        self.presentation_mode = False
        self.fullscreen_demo = False
        self.presentation_technical = True
        self.theme_name = "Verde tecnológico"
        self.high_contrast = False
        self.current_palette = dict(THEMES[self.theme_name])
        self.current_hover = "#1DB954"
        self.session_started = datetime.now().astimezone()
        self.session_counts = {"AUTORIZADO": 0, "DENEGADO": 0}
        self.session_users: set[str] = set()
        self.incidents = deque(maxlen=200)
        self.incident_fingerprint = None
        self.incident_last_at = 0.0
        self.session_had_camera_error = False
        self.session_had_esp_error = False
        self.utility_results: queue.Queue = queue.Queue()
        self.backup_paths: list[Path] = []
        self.last_backup_path: Path | None = None
        self.hourly_rows: list[tuple[str, int, int]] = []
        self.nav_transition_job = None
        self.intro_job = None
        self.daily_rows: list[tuple[str, int, int]] = []
        self.history_dates = (None, None)
        self.chart_drawn = None
        self.heatmap_drawn = None
        self.placeholder_drawn = None
        self.resize_job = None

        self.build_ui()
        self.update_session_summary()
        self.refresh_history()
        self.set_state("INICIANDO", "Preparando cámara y reconocimiento…")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.backend_start_job = self.after(20, self.backend.start)
        self.poll_job = self.after(40, self.poll_backend)
        self.clock_job = self.after(1000, self.update_clock)
        self.icon_job = self.after(450, self.start_icons)
        self.after(60, self.show_intro)
        self.bind("<Configure>", self.schedule_responsive_layout)
        self.bind("<Escape>", lambda _event: self.exit_fullscreen())
        self.bind("<F11>", lambda _event: self.toggle_fullscreen())
        if not self.admin.configured:
            self.after(100, self.prompt_create_pin)

    @staticmethod
    def card(parent, radius: int = 16) -> ctk.CTkFrame:
        return ctk.CTkFrame(
            parent, fg_color=SURFACE, corner_radius=radius,
            border_width=1, border_color=CARD_BORDER,
        )

    def register_icon(self, widget, name: str, role: str = "muted", size: int = 18) -> None:
        binding = (name, role, size)
        if self.icon_bindings.get(widget) == binding:
            return
        self.icon_bindings[widget] = binding
        if self.icons_ready:
            self.apply_widget_icon(widget)

    def start_icons(self) -> None:
        self.icon_job = None
        self.icons_ready = True
        self.icon_queue = list(self.icon_bindings)
        self.apply_icon_batch()

    def apply_icon_batch(self) -> None:
        self.icon_job = None
        for _ in range(3):
            if not self.icon_queue:
                return
            self.apply_widget_icon(self.icon_queue.pop(0))
        self.icon_job = self.after(25, self.apply_icon_batch)

    def apply_widget_icon(self, widget) -> None:
        if not widget.winfo_exists() or widget not in self.icon_bindings:
            return
        name, role, size = self.icon_bindings[widget]
        color = {"accent": CYAN, "muted": MUTED, "text": TEXT,
                 "red": RED, "blue": BLUE, "amber": AMBER}.get(role, TEXT)
        display = (name, size, color)
        if self.icon_displayed.get(widget) == display:
            return
        picture = self.icons.ctk_image(name, size=size, color=color)
        if picture is not None:
            widget.configure(image=picture, compound="left")
            self.icon_displayed[widget] = display

    def forget_icons(self, root) -> None:
        for widget in self._walk_widgets(root):
            self.icon_bindings.pop(widget, None)
            self.icon_displayed.pop(widget, None)

    def build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        navigation = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0, width=210)
        self.navigation = navigation
        navigation.grid(row=0, column=0, rowspan=3, sticky="nsew")
        navigation.grid_propagate(False)
        ctk.CTkLabel(navigation, text="◈  SAFEBOX AI", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 20, "bold")).pack(
            anchor="w", padx=20, pady=(29, 6))
        ctk.CTkLabel(navigation, text="CONTROL BIOMÉTRICO", text_color=MUTED,
                     font=ctk.CTkFont("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=22, pady=(0, 34))
        self.nav_buttons = {}
        for label, icon_name in (("Inicio", "HOME"), ("Escáner", "SCAN"),
                                 ("Usuarios", "USERS"), ("Historial", "HISTORY"),
                                 ("Incidentes", "ALERT_TRIANGLE"), ("Configuración", "SETTINGS")):
            button = ctk.CTkButton(
                navigation, text=f"  {label}", anchor="w", height=43,
                corner_radius=22, fg_color="transparent", hover_color=SURFACE_2,
                text_color=MUTED, font=ctk.CTkFont("Segoe UI", 13, "bold"),
                command=lambda page=label: self.navigate(page),
            )
            button.pack(fill="x", padx=13, pady=3)
            self.nav_buttons[label] = button
            self.register_icon(button, icon_name, "muted", 20)
        ctk.CTkLabel(navigation, text="DATOS LOCALES · SAFEBOX", text_color=MUTED,
                     font=ctk.CTkFont("Segoe UI", 10)).pack(
            side="bottom", anchor="w", padx=22, pady=22)

        header = ctk.CTkFrame(self, fg_color="transparent")
        self.header = header
        header.grid(row=0, column=1, sticky="ew", padx=24, pady=(18, 16))
        header.grid_columnconfigure(0, weight=1)
        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="w")
        self.page_title = ctk.CTkLabel(brand, text="Panel de control", text_color=TEXT,
                                       font=ctk.CTkFont("Segoe UI", 27, "bold"))
        self.page_title.pack(side="left")
        self.header_subtitle = ctk.CTkLabel(header, text="SAFEBOX AI  /  ACCESO BIOMÉTRICO", text_color=MUTED,
                     font=ctk.CTkFont("Segoe UI", 10, "bold"))
        self.header_subtitle.grid(
            row=1, column=0, sticky="w", pady=(0, 0)
        )
        header_right = ctk.CTkFrame(header, fg_color="transparent")
        header_right.grid(row=0, column=1, rowspan=2, sticky="e")
        self.clock_label = ctk.CTkLabel(header_right, text="", text_color=MUTED,
                                        font=ctk.CTkFont("Consolas", 12))
        self.clock_label.pack(side="left", padx=(0, 18))
        self.clock_label.configure(text=datetime.now().strftime("%H:%M:%S"))
        mode_label = ("LED ESP32 · CONECTANDO" if self.led_port else
                      "SIMULACIÓN" if self.mode == "simulacion" else "HARDWARE · INICIANDO")
        self.mode_badge = ctk.CTkLabel(
            header_right, text=f"●  {mode_label}",
            text_color=MUTED if self.mode == "simulacion" else AMBER,
            fg_color=SURFACE_2, corner_radius=18, width=190, height=34,
            font=ctk.CTkFont("Segoe UI", 10, "bold"))
        self.mode_badge.pack(side="left")
        self.register_icon(self.mode_badge, "CPU", "muted", 16)
        self.presentation_button = ctk.CTkButton(
            header_right, text="Presentación", width=128, height=34,
            corner_radius=18, fg_color=SURFACE_2, hover_color=BORDER,
            text_color=TEXT, command=self.toggle_presentation)
        self.presentation_button.pack(side="left", padx=(12, 0))
        self.register_icon(self.presentation_button, "PRESENTATION", "text", 18)
        self.nav_progress = Canvas(header, height=3, bg=BG, highlightthickness=0)
        self.nav_progress.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.nav_progress_item = self.nav_progress.create_rectangle(0, 0, 0, 3, fill=CYAN, outline="")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=1, sticky="nsew", padx=24, pady=(0, 14))
        self.dashboard_body = body
        body.grid_columnconfigure(0, weight=7)
        body.grid_columnconfigure(1, weight=4, minsize=355)
        body.grid_rowconfigure(0, weight=1)

        camera_card = self.card(body, 18)
        camera_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        camera_card.grid_columnconfigure(0, weight=1)
        camera_card.grid_rowconfigure(1, weight=1)
        camera_head = ctk.CTkFrame(camera_card, fg_color="transparent")
        camera_head.grid(row=0, column=0, sticky="ew", padx=19, pady=(15, 12))
        camera_head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(camera_head, text="VISTA BIOMÉTRICA", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.camera_badge = ctk.CTkLabel(camera_head, text="●  INICIANDO", text_color=AMBER,
                                         font=ctk.CTkFont("Segoe UI", 10, "bold"))
        self.camera_badge.grid(row=0, column=1, sticky="e")
        self.register_icon(self.camera_badge, "VIDEO", "accent", 16)

        self.camera_canvas = Canvas(camera_card, bg="#101010", highlightthickness=0,
                                    relief="flat", bd=0)
        self.camera_canvas.grid(row=1, column=0, sticky="nsew", padx=18)
        self.camera_canvas.bind("<Configure>", self.on_camera_resize)
        self.image_item = self.camera_canvas.create_image(0, 0, anchor="nw")
        self.border_item = self.camera_canvas.create_rectangle(1, 1, 2, 2, outline=BORDER)
        self.cam_text_item = self.camera_canvas.create_text(
            18, 19, text="CAM 01  /  LIVE", anchor="w", fill=TEXT,
            font=("Consolas", 10, "bold"), state="hidden",
        )
        self.scan_text_item = self.camera_canvas.create_text(
            18, 19, text="●  BIOMETRIC SCAN", anchor="e", fill=CYAN,
            font=("Consolas", 10), state="hidden",
        )
        self.scan_line_item = self.camera_canvas.create_line(0, 0, 0, 0,
                                                              fill=BORDER, state="hidden")
        self.corner_items = [
            [self.camera_canvas.create_line(0, 0, 0, 0, fill=CYAN, width=3,
                                             state="hidden") for _ in range(8)]
            for _ in range(4)
        ]
        self.label_items: list[tuple[int, int]] = []
        self.image_layout = (0, 0, 1, 1, 1.0)
        self.render_tick = 0

        camera_foot = ctk.CTkFrame(camera_card, fg_color="transparent")
        camera_foot.grid(row=2, column=0, sticky="ew", padx=19, pady=(12, 15))
        camera_foot.grid_columnconfigure(0, weight=1)
        self.faces_label = ctk.CTkLabel(camera_foot, text="ROSTROS DETECTADOS  00",
                                        text_color=MUTED, font=ctk.CTkFont("Consolas", 10))
        self.faces_label.grid(row=0, column=0, sticky="w")
        self.scan_result = ctk.CTkLabel(camera_foot, text="Esperando cámara", text_color=MUTED,
                                        font=ctk.CTkFont("Segoe UI", 11, "bold"))
        self.scan_result.grid(row=0, column=1, sticky="e", padx=12)
        self.retry_button = ctk.CTkButton(
            camera_foot, text="Reintentar cámara", width=128, height=28,
            fg_color=SURFACE_2, hover_color=BORDER, text_color=TEXT,
            corner_radius=18,
            font=ctk.CTkFont("Segoe UI", 10, "bold"), command=self.open_camera,
        )
        self.retry_button.grid(row=0, column=2, sticky="e")
        self.retry_button.grid_remove()
        self.register_icon(self.retry_button, "REFRESH", "text", 15)

        side = ctk.CTkFrame(body, fg_color="transparent")
        side.grid(row=0, column=1, sticky="nsew")
        self.dashboard_side = side
        self.camera_card = camera_card
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(3, weight=1)

        identity = self.card(side)
        self.identity_card = identity
        identity.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        identity.grid_columnconfigure(0, weight=1)
        self.state_chip = ctk.CTkLabel(
            identity, text="●  ESCANEANDO", text_color=CYAN,
            fg_color=SURFACE_2, corner_radius=9, width=130, height=26,
            font=ctk.CTkFont("Segoe UI", 10, "bold"),
        )
        self.state_chip.grid(row=0, column=0, sticky="w", padx=18, pady=(8, 3))
        self.register_icon(self.state_chip, "SCAN", "accent", 16)
        self.identity_title = ctk.CTkLabel(
            identity, text="Esperando identidad", text_color=TEXT,
            font=ctk.CTkFont("Segoe UI", 20, "bold"), anchor="w",
        )
        self.identity_title.grid(row=1, column=0, sticky="ew", padx=18)
        self.detail_label = ctk.CTkLabel(
            identity, text="Sitúate frente a la cámara", text_color=MUTED,
            font=ctk.CTkFont("Segoe UI", 10), anchor="w",
        )
        self.detail_label.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 5))
        metric = ctk.CTkFrame(identity, fg_color=SURFACE_2, corner_radius=10)
        self.metric_frame = metric
        metric.grid(row=3, column=0, sticky="ew", padx=17, pady=(0, 6))
        metric.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(metric, text="SIMILITUD COSENO · SFACE", text_color=MUTED,
                     font=ctk.CTkFont("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(6, 0)
        )
        self.distance_label = ctk.CTkLabel(metric, text="—", text_color=TEXT,
                                            font=ctk.CTkFont("Consolas", 18, "bold"))
        self.distance_label.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 6))
        self.threshold_label = ctk.CTkLabel(metric, text=f"UMBRAL  ≥ {self.similarity_threshold:.3f}   ·   MAYOR ES MEJOR",
                     text_color=MUTED, font=ctk.CTkFont("Consolas", 9))
        self.threshold_label.grid(
            row=0, column=1, rowspan=2, sticky="e", padx=14
        )
        flow = ctk.CTkFrame(identity, fg_color="transparent")
        flow.grid(row=4, column=0, sticky="ew", padx=17, pady=(0, 6))
        self.flow_labels = []
        for index, title in enumerate(("ROSTRO", "PIN", "ACCESO", "PESTILLO", "CIERRE")):
            flow.grid_columnconfigure(index, weight=1)
            label = ctk.CTkLabel(flow, text=title, height=20, corner_radius=9,
                                 fg_color=SURFACE_2, text_color=MUTED,
                                 font=ctk.CTkFont("Segoe UI", 10, "bold"))
            label.grid(row=0, column=index, sticky="ew", padx=(0, 4) if index < 4 else 0)
            self.flow_labels.append(label)

        hardware_row = ctk.CTkFrame(side, fg_color="transparent")
        self.hardware_row = hardware_row
        hardware_row.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        hardware_row.grid_columnconfigure((0, 1), weight=1)
        hardware_row.grid_rowconfigure(0, weight=1)
        lock_card = self.card(hardware_row, 13)
        self.lock_card = lock_card
        lock_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        led_card = self.card(hardware_row, 13)
        self.led_card = led_card
        led_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.lock_icon = ctk.CTkLabel(lock_card, text="▣  PESTILLO", text_color=AMBER,
                                      font=ctk.CTkFont("Segoe UI", 11, "bold"))
        self.lock_icon.pack(anchor="w", padx=14, pady=(6, 0))
        self.register_icon(self.lock_icon, "LOCK", "amber", 18)
        self.lock_value = ctk.CTkLabel(
            lock_card, text="BLOQUEADA · SOLO SIM" if self.mode == "simulacion" else "SIN CONFIRMAR",
            text_color=TEXT, font=ctk.CTkFont("Segoe UI", 12, "bold"),
            wraplength=150, justify="left", anchor="w")
        self.lock_value.pack(anchor="w", fill="x", padx=14, pady=(0, 0))
        self.servo_value = ctk.CTkLabel(lock_card, text=f"MG90S · {self.servo_status}",
                                        text_color=MUTED, font=ctk.CTkFont("Segoe UI", 10),
                                        anchor="w")
        self.servo_value.pack(anchor="w", fill="x", padx=14, pady=(0, 6))
        self.led_value = self.device_label(led_card, "SEÑAL LED", "APAGADO")
        self.register_icon(self.led_value, "BULB", "blue", 17)
        self.led_lamp = ctk.CTkLabel(led_card, text="●", text_color=MUTED,
                                     font=ctk.CTkFont("Segoe UI", 27))
        self.led_lamp.pack(anchor="w", padx=14, pady=(0, 11))
        self.lock_action = ctk.CTkFrame(side, fg_color=SURFACE_2, corner_radius=13)
        self.lock_action.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        self.lock_action.grid_columnconfigure(0, weight=1)
        self.lock_action_label = ctk.CTkLabel(
            self.lock_action, text="CIERRE PENDIENTE · CONFIRMACIÓN MANUAL",
            text_color=AMBER, font=ctk.CTkFont("Segoe UI", 10, "bold"),
            anchor="w", justify="left", wraplength=320)
        self.lock_action_label.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 4))
        self.close_button = ctk.CTkButton(
            self.lock_action, text="Confirmar cierre y preparar MG90S", height=36,
            corner_radius=18, fg_color=CYAN, hover_color="#1DB954", text_color=BG,
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
            command=self.confirm_door_closed)
        self.close_button.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))
        self.register_icon(self.close_button, "LOCK", "text", 17)
        self.lock_action.grid_remove()

        activity = self.card(side)
        self.activity_card = activity
        activity.grid(row=3, column=0, sticky="nsew")
        activity.grid_columnconfigure(0, weight=1)
        activity.grid_rowconfigure(2, weight=1)
        activity_head = ctk.CTkFrame(activity, fg_color="transparent")
        activity_head.grid(row=0, column=0, sticky="ew", padx=17, pady=(8, 4))
        activity_head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(activity_head, text="ACTIVIDAD RECIENTE", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(activity_head, text="LOCAL", text_color=MUTED,
                     font=ctk.CTkFont("Consolas", 9)).grid(row=0, column=1, sticky="e")
        self.history_list = ctk.CTkScrollableFrame(
            activity, fg_color="transparent", scrollbar_button_color=BORDER,
            height=58,
        )
        self.session_label = ctk.CTkLabel(
            activity, text="SESIÓN · 0 autorizados · 0 denegados · 0 errores",
            text_color=MUTED, font=ctk.CTkFont("Segoe UI", 10), anchor="w")
        self.session_label.grid(row=1, column=0, sticky="ew", padx=17, pady=(0, 5))
        self.history_list.grid(row=2, column=0, sticky="nsew", padx=11, pady=(0, 8))

        self.register_button = ctk.CTkButton(
            side, text="+  Registrar nuevo usuario", height=43,
            fg_color=CYAN, hover_color="#1DB954", text_color=BG,
            corner_radius=22,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
            command=self.open_registration,
        )
        self.register_button.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self.register_icon(self.register_button, "USER_PLUS", "text", 19)

        self.presentation_controls = ctk.CTkFrame(side, fg_color="transparent")
        self.presentation_controls.grid(row=5, column=0, sticky="ew", pady=(14, 0))
        self.presentation_controls.grid_columnconfigure((0, 1), weight=1)
        self.fullscreen_button = ctk.CTkButton(
            self.presentation_controls, text="Pantalla completa", height=40,
            corner_radius=20, fg_color=SURFACE_2, hover_color=BORDER,
            command=self.toggle_fullscreen)
        self.fullscreen_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.register_icon(self.fullscreen_button, "ARROWS_MAXIMIZE", "text", 18)
        self.technical_button = ctk.CTkButton(
            self.presentation_controls, text="Ocultar detalles", height=40,
            corner_radius=20, fg_color=SURFACE_2, hover_color=BORDER,
            command=self.toggle_presentation_details)
        self.technical_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.register_icon(self.technical_button, "INFO_CIRCLE", "text", 18)
        ctk.CTkButton(self.presentation_controls, text="Finalizar demo · ver resumen", height=40,
                      corner_radius=20, fg_color=SURFACE_2, hover_color=BORDER,
                      command=self.show_demo_summary).grid(row=1, column=0, columnspan=2,
                                                           sticky="ew", pady=(8, 0))
        self.presentation_session = ctk.CTkLabel(
            side, text="", text_color=MUTED, font=ctk.CTkFont("Segoe UI", 11),
            justify="left", anchor="w")
        self.presentation_session.grid(row=6, column=0, sticky="ew", pady=(10, 0))
        self.presentation_controls.grid_remove()
        self.presentation_session.grid_remove()

        footer = ctk.CTkFrame(self, fg_color="transparent")
        self.footer = footer
        footer.grid(row=2, column=1, sticky="ew", padx=24, pady=(0, 13))
        footer.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(footer, text="SAFEBOX AI  /  RECONOCIMIENTO LOCAL  /  DATOS EN ESTE EQUIPO",
                     text_color=MUTED, font=ctk.CTkFont("Consolas", 9)).grid(
            row=0, column=0, sticky="w"
        )
        self.users_count = ctk.CTkLabel(footer, text="", text_color=MUTED,
                                        font=ctk.CTkFont("Consolas", 9))
        self.users_count.grid(row=0, column=1, sticky="e")
        self.update_user_count()
        self.notice_label = ctk.CTkLabel(footer, text="", text_color=AMBER,
                                         font=ctk.CTkFont("Segoe UI", 10),
                                         wraplength=680, justify="left")
        self.notice_label.grid(row=1, column=0, columnspan=2, sticky="w")
        self.build_secondary_pages()
        self.navigate("Inicio")

    def build_secondary_pages(self) -> None:
        self.pages = {}
        for title, heading, subtitle in (
            ("Usuarios", "Identidades autorizadas", "Gestiona los rostros registrados en este equipo"),
            ("Historial", "Registro de accesos", "Consulta los intentos guardados localmente"),
            ("Incidentes", "Eventos técnicos", "Diagnóstico de esta sesión · separado de los accesos"),
            ("Configuración", "Estado operativo", "Cámara, reconocimiento y modo de conexión"),
        ):
            page = ctk.CTkFrame(self, fg_color="transparent")
            page.grid(row=1, column=1, sticky="nsew", padx=24, pady=(0, 14))
            page.grid_columnconfigure(0, weight=1)
            page.grid_rowconfigure(2, weight=1)
            ctk.CTkLabel(page, text=heading, text_color=TEXT,
                         font=ctk.CTkFont("Segoe UI", 28, "bold")).grid(
                row=0, column=0, sticky="w", pady=(3, 0))
            ctk.CTkLabel(page, text=subtitle, text_color=MUTED,
                         font=ctk.CTkFont("Segoe UI", 12)).grid(
                row=1, column=0, sticky="w", pady=(0, 20))
            self.pages[title] = page
            page.grid_remove()

        users = self.pages["Usuarios"]
        users_card = self.card(users)
        users_card.grid(row=2, column=0, sticky="nsew")
        users_card.grid_columnconfigure(0, weight=1)
        users_card.grid_rowconfigure(2, weight=1)
        users_head = ctk.CTkFrame(users_card, fg_color="transparent")
        users_head.grid(row=0, column=0, sticky="ew", padx=22, pady=(20, 8))
        users_head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(users_head, text="Personas registradas", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 17, "bold")).grid(row=0, column=0, sticky="w")
        users_add = ctk.CTkButton(users_head, text="Registrar rostro", width=170, height=38,
                                  corner_radius=20, fg_color=CYAN, hover_color="#1DB954",
                                  text_color=BG, command=self.open_registration)
        users_add.grid(row=0, column=1)
        self.register_icon(users_add, "USER_PLUS", "text", 18)
        self.users_search = ctk.CTkEntry(
            users_card, placeholder_text="Buscar usuario por nombre…", height=38,
            corner_radius=19, fg_color=SURFACE_2, border_width=0)
        self.users_search.grid(row=1, column=0, sticky="ew", padx=22, pady=(0, 10))
        self.users_search.bind("<KeyRelease>", lambda _event: self.refresh_users())
        self.users_list = ctk.CTkScrollableFrame(users_card, fg_color="transparent")
        self.users_list.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 16))
        self.refresh_users()

        history = self.pages["Historial"]
        filters = self.card(history)
        filters.grid(row=2, column=0, sticky="nsew")
        filters.grid_columnconfigure(0, weight=1)
        filters.grid_rowconfigure(6, weight=1)
        self.stats_label = ctk.CTkLabel(
            filters, text="Cargando estadísticas…", text_color=TEXT,
            font=ctk.CTkFont("Segoe UI", 12, "bold"))
        self.stats_label.grid(row=0, column=0, sticky="w", padx=24, pady=(15, 0))
        self.register_icon(self.stats_label, "CHART_BAR", "accent", 18)
        search_row = ctk.CTkFrame(filters, fg_color="transparent")
        search_row.grid(row=1, column=0, sticky="ew", padx=20, pady=(10, 12))
        search_row.grid_columnconfigure(0, weight=1)
        self.history_search = ctk.CTkEntry(
            search_row, placeholder_text="Buscar usuario…", height=40,
            corner_radius=20, fg_color=SURFACE_2, border_width=0, text_color=TEXT)
        self.history_search.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        self.history_search.bind("<KeyRelease>", self.schedule_history_search)
        self.history_filter = ctk.CTkOptionMenu(
            search_row, values=["Todos", "AUTORIZADO", "DENEGADO"],
            width=155, height=40, corner_radius=20, fg_color=SURFACE_2,
            button_color=BORDER, button_hover_color="#505050",
            text_color=TEXT, command=lambda _: self.reset_history_search())
        self.history_filter.grid(row=0, column=1)
        csv_all = ctk.CTkButton(search_row, text="Exportar CSV", width=115, height=40,
                                corner_radius=20, fg_color=BORDER, hover_color="#505050",
                                text_color=TEXT, command=self.export_history)
        csv_all.grid(row=0, column=2, padx=(10, 0))
        self.register_icon(csv_all, "FILE_EXPORT", "text", 17)
        csv_filtered = ctk.CTkButton(search_row, text="CSV filtrado", width=115, height=40,
                                     corner_radius=20, fg_color=BORDER, hover_color="#505050",
                                     text_color=TEXT, command=self.export_filtered_history)
        csv_filtered.grid(row=0, column=3, padx=(8, 0))
        self.register_icon(csv_filtered, "FILTER", "text", 17)
        date_row = ctk.CTkFrame(filters, fg_color="transparent")
        date_row.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 8))
        self.date_filter = ctk.CTkOptionMenu(
            date_row, values=["Todas", "Hoy", "Últimos 7 días", "Últimos 30 días", "Personalizado"],
            width=160, height=34, corner_radius=17, fg_color=SURFACE_2,
            button_color=BORDER, command=lambda _: self.reset_history_search())
        self.date_filter.pack(side="left", padx=(0, 8))
        self.date_start = ctk.CTkEntry(date_row, placeholder_text="Desde AAAA-MM-DD", width=145, height=34)
        self.date_start.pack(side="left", padx=(0, 6))
        self.date_end = ctk.CTkEntry(date_row, placeholder_text="Hasta AAAA-MM-DD", width=145, height=34)
        self.date_end.pack(side="left", padx=(0, 6))
        for entry in (self.date_start, self.date_end):
            entry.bind("<Return>", lambda _event: self.reset_history_search())
        self.history_chart = Canvas(filters, bg=SURFACE, height=110, highlightthickness=0)
        self.history_chart.grid(row=3, column=0, sticky="ew", padx=24, pady=(0, 4))
        self.history_chart.bind("<Configure>", lambda _event: self.render_history_chart())
        self.heatmap_chart = Canvas(filters, bg=SURFACE, height=105, highlightthickness=0)
        self.heatmap_chart.grid(row=4, column=0, sticky="ew", padx=24, pady=(0, 4))
        self.heatmap_chart.bind("<Configure>", lambda _event: self.render_heatmap())
        graph_export = ctk.CTkButton(filters, text="Guardar gráficos PNG", width=165, height=28,
                                     corner_radius=14, fg_color=SURFACE_2, hover_color=BORDER,
                                     command=self.export_graphs)
        graph_export.grid(row=0, column=0, sticky="e", padx=24, pady=(15, 0))
        self.register_icon(graph_export, "PHOTO", "text", 17)
        self.history_note = ctk.CTkLabel(
            filters, text="Cargando historial…", text_color=MUTED,
            font=ctk.CTkFont("Segoe UI", 11))
        self.history_note.grid(row=5, column=0, sticky="w", padx=24, pady=(0, 8))
        self.history_full_list = ctk.CTkScrollableFrame(filters, fg_color="transparent")
        self.history_full_list.grid(row=6, column=0, sticky="nsew", padx=17, pady=(0, 16))
        pager = ctk.CTkFrame(filters, fg_color="transparent")
        pager.grid(row=7, column=0, sticky="e", padx=20, pady=(0, 16))
        self.history_prev = ctk.CTkButton(pager, text="Anterior", width=95, height=30,
                                           corner_radius=16, fg_color=SURFACE_2,
                                           hover_color=BORDER, command=lambda: self.change_history_page(-1))
        self.history_prev.pack(side="left", padx=(0, 8))
        self.history_next = ctk.CTkButton(pager, text="Siguiente", width=95, height=30,
                                           corner_radius=16, fg_color=SURFACE_2,
                                           hover_color=BORDER, command=lambda: self.change_history_page(1))
        self.history_next.pack(side="left")

        incidents_page = self.pages["Incidentes"]
        incident_card = self.card(incidents_page)
        incident_card.grid(row=2, column=0, sticky="nsew")
        incident_card.grid_columnconfigure(0, weight=1)
        incident_card.grid_rowconfigure(2, weight=1)
        incident_heading = ctk.CTkLabel(incident_card, text="Incidentes técnicos · sesión actual", text_color=TEXT,
                                        font=ctk.CTkFont("Segoe UI", 17, "bold"))
        incident_heading.grid(row=0, column=0, sticky="w", padx=22, pady=(18, 8))
        self.register_icon(incident_heading, "ALERT_TRIANGLE", "amber", 20)
        self.assistant_label = ctk.CTkLabel(
            incident_card, text="Asistente: esperando el estado del sistema…",
            text_color=MUTED, fg_color=SURFACE_2, corner_radius=12,
            font=ctk.CTkFont("Segoe UI", 12), justify="left", anchor="w",
            wraplength=720)
        self.assistant_label.grid(row=1, column=0, sticky="ew", padx=22, pady=(0, 12))
        self.incident_list = ctk.CTkScrollableFrame(incident_card, fg_color="transparent")
        self.incident_list.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 18))
        self.refresh_incidents()

        settings = self.pages["Configuración"]
        settings.grid_rowconfigure(2, weight=0)
        settings.grid_rowconfigure(3, weight=1)
        diagnostic = self.card(settings)
        diagnostic.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        diagnostic.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        self.diagnostic_labels = {}
        self.diagnostic_cache = {}
        for column, title in enumerate(("CÁMARA", "ROSTRO", "ESP32", "MG90S", "ACCESO")):
            tile = ctk.CTkFrame(diagnostic, fg_color=SURFACE_2, corner_radius=11)
            tile.grid(row=0, column=column, sticky="ew", padx=(12 if column == 0 else 4,
                       12 if column == 4 else 4), pady=12)
            heading = ctk.CTkLabel(tile, text=title, text_color=MUTED,
                                   font=ctk.CTkFont("Segoe UI", 9, "bold"))
            heading.pack(anchor="w", padx=10, pady=(8, 0))
            self.register_icon(heading, {"CÁMARA": "VIDEO", "ROSTRO": "FACE_ID",
                                         "ESP32": "CPU", "MG90S": "LOCK",
                                         "ACCESO": "SHIELD_CHECK"}[title], "muted", 15)
            label = ctk.CTkLabel(tile, text="INICIANDO", text_color=AMBER,
                                 font=ctk.CTkFont("Segoe UI", 11, "bold"),
                                 anchor="w", wraplength=125)
            label.pack(fill="x", padx=10, pady=(0, 8))
            self.diagnostic_labels[title] = label
        settings_card = ctk.CTkScrollableFrame(settings, fg_color=SURFACE, corner_radius=16)
        settings_card.grid(row=3, column=0, sticky="nsew")
        settings_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(settings_card, text="Cámara y reconocimiento", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 18, "bold")).grid(
            row=0, column=0, sticky="w", padx=24, pady=(16, 8))
        self.settings_camera = self.setting_line(settings_card, 1, "Cámara", "Iniciando…")
        self.settings_engine = self.setting_line(settings_card, 2, "Reconocimiento", "Iniciando…")
        self.register_icon(self.settings_camera, "VIDEO", "muted", 16)
        self.register_icon(self.settings_engine, "FACE_ID", "muted", 16)
        self.setting_line(settings_card, 3, "Modelo facial", "YuNet + SFace · CPU")
        ctk.CTkLabel(settings_card, text="ESP32 y acceso", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 17, "bold")).grid(
            row=7, column=0, sticky="w", padx=24, pady=(18, 8))
        mode_text = (f"Cerradura simulada · LEDs {self.led_port} conectando" if self.led_port else
                     "Simulación · sin hardware conectado" if self.mode == "simulacion" else
                     f"Hardware · esperando conexión ({self.backend.port})")
        self.settings_device = self.setting_line(settings_card, 8, "Modo", mode_text)
        self.register_icon(self.settings_device, "CPU", "muted", 16)
        ctk.CTkLabel(settings_card, text="Pestillo MG90S", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 17, "bold")).grid(
            row=15, column=0, sticky="w", padx=24, pady=(18, 8))
        self.settings_servo = self.setting_line(settings_card, 16, "MG90S", self.servo_status)
        self.register_icon(self.settings_servo, "LOCK", "muted", 16)
        self.setting_line(settings_card, 12, "Datos", str(self.backend.data_dir))
        self.threshold_entry = self.setting_input(settings_card, 5, "Umbral SFace", f"{self.similarity_threshold:.3f}")
        self.duration_entry = self.setting_input(settings_card, 11, "Autorización (segundos)", str(self.authorization_seconds))
        self.camera_choice = ctk.CTkOptionMenu(settings_card, values=[f"Cámara {i}" for i in range(10)],
                                               fg_color=SURFACE_2, button_color=BORDER)
        self.camera_choice.set(f"Cámara {self.camera_index}")
        self.camera_choice.grid(row=4, column=0, sticky="ew", padx=24, pady=4)
        self.mode_choice = ctk.CTkOptionMenu(settings_card, values=["Simulación", "Hardware"],
                                             fg_color=SURFACE_2, button_color=BORDER)
        self.mode_choice.set("Simulación" if self.mode == "simulacion" else "Hardware")
        self.mode_choice.grid(row=9, column=0, sticky="ew", padx=24, pady=4)
        self.port_entry = self.setting_input(settings_card, 10, "Puerto serial (COMx)",
                                             self.backend.port or self.led_port or "")
        actions = ctk.CTkFrame(settings_card, fg_color="transparent")
        actions.grid(row=13, column=0, sticky="w", padx=24, pady=12)
        ctk.CTkButton(actions, text="Guardar ajustes y comprobar conexión", width=265,
                      height=38, corner_radius=20, fg_color=CYAN, hover_color="#1DB954",
                      text_color=BG, command=self.save_settings).pack(side="left", padx=(0, 8))
        self.settings_retry = ctk.CTkButton(
            settings_card, text="Reintentar cámara", width=170,
            height=38, corner_radius=20, fg_color=SURFACE_2,
            hover_color=BORDER, text_color=TEXT, command=self.open_camera)
        self.settings_retry.grid(row=6, column=0, sticky="w", padx=24, pady=(4, 8))
        self.register_icon(self.settings_retry, "REFRESH", "text", 17)
        ctk.CTkLabel(settings_card,
                     text="El modo hardware requiere ESP32 y ACK serial. Confirma el cierre antes de volver a bloquear.",
                     text_color=MUTED, font=ctk.CTkFont("Segoe UI", 11),
                     wraplength=650, justify="left").grid(
            row=14, column=0, sticky="w", padx=24, pady=(0, 14))
        servo_actions = ctk.CTkFrame(settings_card, fg_color="transparent")
        servo_actions.grid(row=17, column=0, sticky="w", padx=24, pady=(0, 12))
        ctk.CTkButton(servo_actions, text="Probar MG90S", width=150, height=36,
                      corner_radius=18, fg_color=SURFACE_2, hover_color=BORDER,
                      command=lambda: self.require_admin("Calibrar MG90S", self.test_servo)).pack(side="left", padx=(0, 8))
        ctk.CTkButton(servo_actions, text="Detener PWM", width=140, height=36,
                      corner_radius=18, fg_color=SURFACE_2, hover_color=BORDER,
                      command=self.stop_servo_signal).pack(side="left")
        self.servo_open_entry = self.setting_input(settings_card, 18, "Autorización · µs", "—")
        self.servo_lock_entry = self.setting_input(settings_card, 19, "Retorno · µs", "—")
        self.servo_save_button = ctk.CTkButton(
            settings_card, text="Guardar recorrido MG90S", width=210, height=36,
            corner_radius=18, fg_color=SURFACE_2, hover_color=BORDER,
            command=lambda: self.require_admin("Ajustar MG90S", self.save_servo_config))
        self.servo_save_button.grid(row=20, column=0, sticky="w", padx=24, pady=(6, 4))
        self.register_icon(self.servo_save_button, "DEVICE_FLOPPY", "text", 17)
        ctk.CTkLabel(settings_card,
                     text="Solo DEMO · 1000–2000 µs · guardar no mueve el servo."
                          " La posición física no está verificada.",
                     text_color=MUTED, font=ctk.CTkFont("Segoe UI", 11),
                     wraplength=650, justify="left").grid(
            row=21, column=0, sticky="w", padx=24, pady=(0, 14))
        ctk.CTkLabel(settings_card, text="Administración", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 17, "bold")).grid(
            row=22, column=0, sticky="w", padx=24, pady=(18, 8))
        admin_actions = ctk.CTkFrame(settings_card, fg_color="transparent")
        admin_actions.grid(row=23, column=0, sticky="w", padx=24, pady=(0, 16))
        admin_pin_button = ctk.CTkButton(admin_actions, text="Cambiar PIN admin", width=145, height=38,
                                         corner_radius=20, fg_color=SURFACE_2, hover_color=BORDER,
                                         command=self.change_pin)
        admin_pin_button.pack(side="left", padx=(0, 8))
        self.register_icon(admin_pin_button, "KEY", "text", 17)
        ctk.CTkButton(admin_actions, text="Cerrar sesión", width=120, height=38,
                      corner_radius=20, fg_color=SURFACE_2, hover_color=BORDER,
                      command=self.lock_admin_session).pack(side="left")
        backup_button = ctk.CTkButton(admin_actions, text="Crear respaldo local", width=160, height=38,
                                      corner_radius=20, fg_color=SURFACE_2, hover_color=BORDER,
                                      command=lambda: self.require_admin("Crear respaldo", self.choose_backup))
        backup_button.pack(side="left", padx=(8, 0))
        self.register_icon(backup_button, "DATABASE_EXPORT", "text", 17)
        ctk.CTkLabel(settings_card, text="Presentación visual", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 17, "bold")).grid(
                         row=24, column=0, sticky="w", padx=24, pady=(18, 8))
        visual_actions = ctk.CTkFrame(settings_card, fg_color="transparent")
        visual_actions.grid(row=25, column=0, sticky="w", padx=24, pady=(0, 12))
        self.theme_selector = ctk.CTkOptionMenu(
            visual_actions, values=list(THEMES), width=190, height=38,
            fg_color=SURFACE_2, button_color=BORDER, command=self.set_visual_theme)
        self.theme_selector.set(self.theme_name)
        self.theme_selector.pack(side="left", padx=(0, 8))
        self.contrast_button = ctk.CTkButton(
            visual_actions, text="Alto contraste: no", width=170, height=38,
            corner_radius=20, fg_color=SURFACE_2, hover_color=BORDER,
            command=self.toggle_high_contrast)
        self.contrast_button.pack(side="left")
        ctk.CTkLabel(settings_card, text="Datos locales · solo diagnóstico", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 17, "bold")).grid(
                         row=26, column=0, sticky="w", padx=24, pady=(18, 8))
        data_actions = ctk.CTkFrame(settings_card, fg_color="transparent")
        data_actions.grid(row=27, column=0, sticky="w", padx=24, pady=(0, 8))
        integrity_button = ctk.CTkButton(data_actions, text="Comprobar integridad", width=170, height=38,
                                         corner_radius=20, fg_color=SURFACE_2, hover_color=BORDER,
                                         command=lambda: self.require_admin("Comprobar integridad", self.run_integrity_check))
        integrity_button.pack(side="left", padx=(0, 8))
        self.register_icon(integrity_button, "SHIELD_CHECK", "text", 17)
        storage_button = ctk.CTkButton(data_actions, text="Ver almacenamiento", width=160, height=38,
                                       corner_radius=20, fg_color=SURFACE_2, hover_color=BORDER,
                                       command=self.run_storage_check)
        storage_button.pack(side="left", padx=(0, 8))
        self.register_icon(storage_button, "DATABASE", "text", 17)
        self.verify_backup_button = ctk.CTkButton(
            data_actions, text="Verificar respaldo", width=165, height=38,
            corner_radius=20, fg_color=SURFACE_2, hover_color=BORDER,
            command=lambda: self.require_admin(
                "Verificar respaldo", self.run_backup_verification))
        self.verify_backup_button.pack(side="left")
        self.register_icon(self.verify_backup_button, "FILE_CHECK", "text", 17)
        self.data_diagnostics = ctk.CTkLabel(
            settings_card, text="Las comprobaciones no modifican archivos.",
            text_color=MUTED, font=ctk.CTkFont("Segoe UI", 11),
            justify="left", anchor="w", wraplength=700)
        self.data_diagnostics.grid(row=28, column=0, sticky="ew", padx=24, pady=(0, 18))

    @staticmethod
    def setting_input(parent, row: int, title: str, value: str) -> ctk.CTkEntry:
        line = ctk.CTkFrame(parent, fg_color=SURFACE_2, corner_radius=10)
        line.grid(row=row, column=0, sticky="ew", padx=24, pady=4)
        line.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(line, text=title, text_color=MUTED).grid(row=0, column=0, padx=16, pady=8)
        entry = ctk.CTkEntry(line, width=190)
        entry.insert(0, value)
        entry.grid(row=0, column=1, sticky="e", padx=16, pady=8)
        return entry

    @staticmethod
    def setting_line(parent, row: int, title: str, value: str) -> ctk.CTkLabel:
        line = ctk.CTkFrame(parent, fg_color=SURFACE_2, corner_radius=10)
        line.grid(row=row, column=0, sticky="ew", padx=24, pady=4)
        line.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(line, text=title, text_color=MUTED,
                     font=ctk.CTkFont("Segoe UI", 12)).grid(
            row=0, column=0, sticky="w", padx=16, pady=8)
        label = ctk.CTkLabel(line, text=value, text_color=TEXT,
                             font=ctk.CTkFont("Segoe UI", 12, "bold"),
                             wraplength=450, justify="right")
        label.grid(row=0, column=1, sticky="e", padx=16, pady=8)
        return label

    def navigate(self, page: str) -> None:
        if page != "Inicio" and self.presentation_mode:
            self.toggle_presentation()
        previous = self.current_page
        self.current_page = page
        self.page_title.configure(text="Panel de control" if page == "Inicio" else page)
        for name, button in self.nav_buttons.items():
            active = name == page
            button.configure(fg_color=GREEN_DIM if active else "transparent",
                             text_color=CYAN if active else MUTED,
                             hover_color=GREEN_DIM if active else SURFACE_2)
        for frame in self.pages.values():
            frame.grid_remove()
        if page in {"Inicio", "Escáner"}:
            self.dashboard_body.grid()
            if page == "Escáner":
                self.dashboard_side.grid_remove()
                self.dashboard_body.grid_columnconfigure(0, weight=1)
                self.dashboard_body.grid_columnconfigure(1, weight=0, minsize=0)
            else:
                self.dashboard_side.grid()
                self.dashboard_body.grid_columnconfigure(0, weight=7)
                self.dashboard_body.grid_columnconfigure(1, weight=4, minsize=355)
        else:
            self.dashboard_body.grid_remove()
            self.pages[page].grid()
            if page == "Usuarios":
                self.backend.send("list_users")
                self.refresh_users()
            elif page == "Historial":
                self.request_history()
                self.render_history_chart()
                self.render_heatmap()
            elif page == "Incidentes":
                self.refresh_incidents()
                self.refresh_assistant()
            elif page == "Configuración":
                self.refresh_diagnostics()
        if page in {"Inicio", "Escáner"}:
            self.refresh_history()
        self.sync_capture_activity()
        if page != previous:
            self.animate_navigation(0)

    def sync_capture_activity(self) -> None:
        registering = (self.register_window is not None and self.register_window.winfo_exists())
        verifying = (self.access_window is not None and self.access_window.winfo_exists())
        needed = self.current_page in {"Inicio", "Escáner"} or registering or verifying
        self.backend.set_capture_active(needed)

    def animate_navigation(self, step: int) -> None:
        if self.nav_transition_job is not None:
            self.after_cancel(self.nav_transition_job)
            self.nav_transition_job = None
        width = max(1, self.nav_progress.winfo_width())
        self.nav_progress.coords(self.nav_progress_item, 0, 0, width * min(step + 1, 6) / 6, 3)
        if step < 6:
            self.nav_transition_job = self.after(24, lambda: self.animate_navigation(step + 1))
        else:
            self.nav_transition_job = self.after(220, self.clear_navigation_animation)

    def clear_navigation_animation(self) -> None:
        self.nav_transition_job = None
        self.nav_progress.coords(self.nav_progress_item, 0, 0, 0, 3)

    def show_intro(self) -> None:
        if not self.running:
            return
        try:
            overlay = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
            overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.intro_overlay = overlay
            ctk.CTkLabel(overlay, text="◈  SAFEBOX AI", text_color=CYAN,
                         font=ctk.CTkFont("Segoe UI", 35, "bold")).place(relx=0.5, rely=0.43, anchor="center")
            ctk.CTkLabel(overlay, text="SISTEMA BIOMÉTRICO LOCAL", text_color=MUTED,
                         font=ctk.CTkFont("Segoe UI", 12, "bold")).place(relx=0.5, rely=0.52, anchor="center")
            bar = ctk.CTkProgressBar(overlay, width=250, height=5,
                                     progress_color=CYAN, fg_color=SURFACE_2)
            bar.place(relx=0.5, rely=0.6, anchor="center")
            bar.set(0)
            self.intro_progress = bar
            self.animate_intro(0)
        except Exception:
            if getattr(self, "intro_overlay", None) is not None:
                self.intro_overlay.destroy()

    def animate_intro(self, step: int) -> None:
        try:
            self.intro_progress.set(min(1, (step + 1) / 7))
            if step < 6:
                self.intro_job = self.after(55, lambda: self.animate_intro(step + 1))
            else:
                self.intro_job = self.after(120, self.finish_intro)
        except Exception:
            self.finish_intro()

    def finish_intro(self) -> None:
        self.intro_job = None
        if getattr(self, "intro_overlay", None) is not None:
            self.intro_overlay.destroy()
            self.intro_overlay = None

    def toggle_presentation(self) -> None:
        if self.fullscreen_demo:
            self.exit_fullscreen()
        self.presentation_mode = not self.presentation_mode
        if self.presentation_mode:
            self.navigate("Inicio")
            self.navigation.grid_remove()
            self.activity_card.grid_remove()
            self.register_button.grid_remove()
            self.presentation_controls.grid()
            self.presentation_session.grid()
            self.dashboard_side.grid_rowconfigure(3, weight=0)
            self.dashboard_body.grid_columnconfigure(0, weight=9)
            self.dashboard_body.grid_columnconfigure(1, weight=3, minsize=320)
            self.identity_title.configure(font=ctk.CTkFont("Segoe UI", 28, "bold"))
            self.detail_label.configure(font=ctk.CTkFont("Segoe UI", 13))
            self.presentation_button.configure(text="Vista normal")
            self.register_icon(self.presentation_button, "ARROW_LEFT", "text", 18)
            self.page_title.configure(text="Demostración biométrica")
            self.update_session_summary()
        else:
            if not self.presentation_technical:
                self.toggle_presentation_details()
            self.navigation.grid()
            self.activity_card.grid()
            self.register_button.grid()
            self.presentation_controls.grid_remove()
            self.presentation_session.grid_remove()
            self.dashboard_side.grid_rowconfigure(3, weight=1)
            self.identity_title.configure(font=ctk.CTkFont("Segoe UI", 20, "bold"))
            self.detail_label.configure(font=ctk.CTkFont("Segoe UI", 10))
            self.presentation_button.configure(text="Presentación")
            self.register_icon(self.presentation_button, "PRESENTATION", "text", 18)
            self.navigate("Inicio")
            self.apply_responsive_layout()

    def toggle_fullscreen(self) -> None:
        if self.fullscreen_demo:
            self.exit_fullscreen()
            return
        if not self.presentation_mode:
            self.toggle_presentation()
        self.fullscreen_demo = True
        self.attributes("-fullscreen", True)
        self.footer.grid_remove()
        self.header_subtitle.grid_remove()
        self.clock_label.pack_forget()
        self.fullscreen_button.configure(text="Salir de pantalla completa")
        self.presentation_button.configure(text="Salir · Esc")
        self.register_icon(self.fullscreen_button, "ARROWS_MINIMIZE", "text", 18)

    def exit_fullscreen(self) -> None:
        if not self.fullscreen_demo:
            return
        self.fullscreen_demo = False
        self.attributes("-fullscreen", False)
        self.footer.grid()
        self.header_subtitle.grid()
        self.clock_label.pack(side="left", padx=(0, 18), before=self.mode_badge)
        self.fullscreen_button.configure(text="Pantalla completa")
        self.presentation_button.configure(text="Vista normal")
        self.register_icon(self.fullscreen_button, "ARROWS_MAXIMIZE", "text", 18)

    def toggle_presentation_details(self) -> None:
        self.presentation_technical = not self.presentation_technical
        if self.presentation_technical:
            self.metric_frame.grid()
            self.servo_value.pack(anchor="w", fill="x", padx=14, pady=(0, 6))
            self.led_card.grid()
            self.lock_card.grid_configure(columnspan=1)
            self.technical_button.configure(text="Ocultar detalles")
        else:
            self.metric_frame.grid_remove()
            self.servo_value.pack_forget()
            self.led_card.grid_remove()
            self.lock_card.grid_configure(columnspan=2)
            self.technical_button.configure(text="Mostrar detalles")

    def show_demo_summary(self) -> None:
        window = ctk.CTkToplevel(self, fg_color=BG)
        window.title("SAFEBOX AI · Resumen de demostración")
        window.geometry("530x420")
        window.transient(self)
        window.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(window, text="Resumen de la demostración", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 23, "bold")).grid(
                         row=0, column=0, sticky="w", padx=28, pady=(24, 16))
        total = sum(self.session_counts.values())
        details = (f"Intentos de esta sesión: {total}\n"
                   f"Autorizados: {self.session_counts['AUTORIZADO']}  ·  Denegados: {self.session_counts['DENEGADO']}\n"
                   f"Usuarios reconocidos: {len(self.session_users)}\n"
                   f"Incidentes técnicos: {len(self.incidents)}\n"
                   f"Fallos de cámara durante la demo: {'sí' if self.session_had_camera_error else 'no'}\n"
                   f"Fallos de ESP32 durante la demo: {'sí' if self.session_had_esp_error else 'no'}\n"
                   f"Estado lógico: {self.status_state} · {self.lock_state}\n"
                   f"Cámara: {'conectada' if self.camera_connected else 'sin señal'}\n"
                   f"ESP32: {'conectada' if self.device_connected or self.led_connected else 'sin confirmar'}\n"
                   "Posición física de puerta y pestillo no verificada por sensor.")
        ctk.CTkLabel(window, text=details, text_color=MUTED, justify="left", anchor="w",
                     font=ctk.CTkFont("Segoe UI", 13), wraplength=460).grid(
                         row=1, column=0, sticky="ew", padx=28)
        actions = ctk.CTkFrame(window, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=28, pady=24)
        ctk.CTkButton(actions, text="Volver al inicio", height=38, corner_radius=19,
                      fg_color=CYAN, text_color=BG,
                      command=lambda: (window.destroy(), self.exit_fullscreen(),
                                       self.toggle_presentation() if self.presentation_mode else None,
                                       self.navigate("Inicio"))).pack(side="left", padx=(0, 8))
        ctk.CTkButton(actions, text="Cerrar aplicación", height=38, corner_radius=19,
                      fg_color=SURFACE_2, command=lambda: (window.destroy(), self.close())).pack(side="left")

    def schedule_responsive_layout(self, event=None) -> None:
        if event is not None and event.widget is not self:
            return
        if self.resize_job is not None:
            self.after_cancel(self.resize_job)
        self.resize_job = self.after(120, self.apply_responsive_layout)

    def apply_responsive_layout(self) -> None:
        self.resize_job = None
        if self.presentation_mode:
            return
        compact = self.winfo_width() < 1120
        self.navigation.configure(width=175 if compact else 210)
        if self.current_page == "Inicio":
            self.dashboard_body.grid_columnconfigure(0, weight=6 if compact else 7)
            self.dashboard_body.grid_columnconfigure(1, weight=4, minsize=320 if compact else 355)
        if compact:
            self.scan_result.grid_remove()
        else:
            self.scan_result.grid()

    def refresh_users(self) -> None:
        if self.current_page != "Usuarios":
            return
        query = self.users_search.get().strip().casefold()
        fingerprint = (tuple(self.people), frozenset(self.access_pin_ids), query)
        if fingerprint == self._rendered_users:
            return
        self._rendered_users = fingerprint
        for child in self.users_list.winfo_children():
            child.destroy()
        matching = [(uid, name, active) for uid, name, active in self.people
                    if query in name.casefold()]
        if not matching:
            ctk.CTkLabel(self.users_list,
                         text=("◎  SIN COINCIDENCIAS\nPrueba otro nombre." if self.people else
                               "◎  SIN IDENTIDADES\nRegistra el primer rostro para activar el acceso biométrico."),
                         text_color=MUTED, font=ctk.CTkFont("Segoe UI", 14),
                         justify="left").pack(
                anchor="w", padx=12, pady=24)
            return
        for person_id, name, active in matching:
            item = ctk.CTkFrame(self.users_list, fg_color=SURFACE_2, corner_radius=12)
            item.pack(fill="x", pady=5, padx=2)
            item.grid_columnconfigure(1, weight=1)
            initials = "".join(part[0].upper() for part in name.split()[:2]) or "?"
            ctk.CTkLabel(item, text=initials, width=44, height=44, corner_radius=22,
                         fg_color=GREEN_DIM if active else BORDER,
                         text_color=CYAN if active else AMBER,
                         font=ctk.CTkFont("Segoe UI", 14, "bold")).grid(
                row=0, column=0, rowspan=2, padx=(14, 10), pady=10)
            ctk.CTkLabel(item, text=name, text_color=TEXT if active else AMBER,
                         font=ctk.CTkFont("Segoe UI", 13, "bold")).grid(
                row=0, column=1, sticky="w", pady=(9, 0))
            pin_text = "PIN de acceso activo" if person_id in self.access_pin_ids else "PIN de acceso pendiente"
            ctk.CTkLabel(item,
                         text=(f"SFace activo · {pin_text}" if active else
                               f"Registro LBPH anterior · requiere nuevo enrolamiento · {pin_text}"),
                         text_color=MUTED, font=ctk.CTkFont("Segoe UI", 10)).grid(
                row=1, column=1, sticky="w", pady=(0, 9))
            if not active:
                ctk.CTkButton(item, text="Reenrolar", width=95, height=32,
                              corner_radius=17, fg_color=CYAN, hover_color="#1DB954",
                              text_color=BG,
                              command=lambda label=name: self.open_registration(label)).grid(
                    row=0, column=2, rowspan=2, padx=(4, 0))
            missing_pin = person_id not in self.access_pin_ids
            ctk.CTkButton(item, text="Completar PIN" if missing_pin else "PIN acceso", width=105, height=32,
                          corner_radius=17, fg_color=SURFACE, hover_color=BORDER,
                          text_color=TEXT, command=lambda uid=person_id, label=name:
                          self.require_admin("PIN de acceso", lambda: self.set_access_pin(uid, label,
                                                                                    uid not in self.access_pin_ids))).grid(
                row=0, column=3, rowspan=2, padx=(4, 0))
            ctk.CTkButton(item, text="Eliminar", width=95, height=32,
                          corner_radius=17, fg_color="#423035", hover_color="#653B44",
                          text_color=RED, command=lambda uid=person_id, label=name:
                          self.confirm_delete(uid, label)).grid(row=0, column=4, rowspan=2, padx=14)
            ctk.CTkButton(item, text="Detalles", width=82, height=32,
                          corner_radius=17, fg_color=SURFACE, hover_color=BORDER,
                          text_color=TEXT, command=lambda uid=person_id, label=name, enabled=active:
                          self.show_user_details(uid, label, enabled)).grid(
                              row=0, column=5, rowspan=2, padx=(0, 12))

    def show_user_details(self, person_id: str, name: str, active: bool) -> None:
        window = ctk.CTkToplevel(self, fg_color=BG)
        window.title("SAFEBOX AI · Perfil")
        window.geometry("460x315")
        window.transient(self)
        window.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(window, text=name, text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 23, "bold")).grid(
                         row=0, column=0, sticky="w", padx=28, pady=(25, 16))
        pin = person_id in self.access_pin_ids
        details = (f"Registro facial: {'SFace activo' if active else 'requiere reenrolamiento'}\n"
                   f"PIN individual: {'configurado' if pin else 'pendiente'}\n"
                   "Fecha de creación: no registrada en los datos actuales\n"
                   f"Perfil: {'completo' if active and pin else 'incompleto · acceso bloqueado'}")
        ctk.CTkLabel(window, text=details, text_color=MUTED, justify="left", anchor="w",
                     font=ctk.CTkFont("Segoe UI", 13)).grid(
                         row=1, column=0, sticky="ew", padx=28)
        ctk.CTkButton(window, text="Cerrar ficha", height=38, corner_radius=19,
                      fg_color=SURFACE_2, command=window.destroy).grid(
                          row=2, column=0, sticky="ew", padx=28, pady=22)

    def confirm_delete(self, person_id: str, name: str, authorized: bool = False) -> None:
        if not authorized:
            self.require_admin("Eliminar usuario",
                               lambda: self.confirm_delete(person_id, name, True))
            return
        window = ctk.CTkToplevel(self, fg_color=BG)
        window.title("SAFEBOX AI · Confirmar eliminación")
        window.geometry("420x210")
        window.resizable(False, False)
        window.transient(self)
        window.grab_set()
        ctk.CTkLabel(window, text="Eliminar identidad", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 21, "bold")).pack(
            anchor="w", padx=25, pady=(24, 5))
        ctk.CTkLabel(window, text=f"Se eliminarán las muestras de {name}. Esta acción no se puede deshacer.",
                     text_color=MUTED, wraplength=365, justify="left",
                     font=ctk.CTkFont("Segoe UI", 12)).pack(
            anchor="w", padx=25, pady=(0, 22))
        actions = ctk.CTkFrame(window, fg_color="transparent")
        actions.pack(fill="x", padx=25)
        ctk.CTkButton(actions, text="Cancelar", width=125, height=38,
                      corner_radius=20, fg_color=SURFACE_2, hover_color=BORDER,
                      command=window.destroy).pack(side="left")
        def remove() -> None:
            window.destroy()
            self.require_admin("Eliminar usuario", lambda: self.backend.send("delete_user", person_id))
        ctk.CTkButton(actions, text="Eliminar", width=125, height=38,
                      corner_radius=20, fg_color="#883F4D", hover_color="#A54A5B",
                      command=remove).pack(side="right")

    def notify(self, message: str, error: bool = False) -> None:
        now = time.monotonic()
        if (message, error) == self.last_notice[0] and now - self.last_notice[1] < 3.0:
            return
        self.last_notice = ((message, error), now)
        if self.notice_clear_job is not None:
            self.after_cancel(self.notice_clear_job)
        self.notice_label.configure(text=message, text_color=RED if error else CYAN)
        self.register_icon(self.notice_label, "ALERT_TRIANGLE" if error else "INFO_CIRCLE",
                           "red" if error else "accent", 16)
        self.notice_clear_job = self.after(5000, self.clear_notice)

    def clear_notice(self) -> None:
        self.notice_clear_job = None
        if self.running:
            self.notice_label.configure(text="")

    def update_session_summary(self) -> None:
        summary = (f"SESIÓN DESDE {self.session_started:%H:%M}  ·  "
                   f"{self.session_counts['AUTORIZADO']} autorizados  ·  "
                   f"{self.session_counts['DENEGADO']} denegados  ·  "
                   f"{len(self.incidents)} errores  ·  {len(self.session_users)} usuarios")
        self.session_label.configure(text=summary)
        self.presentation_session.configure(text=summary.replace("  ·  ", "\n"))

    def record_incident(self, category: str, detail: str) -> None:
        fingerprint = (category, detail.split(" · reintento en ")[0]
                       if category == "Cámara" else detail)
        now = time.monotonic()
        if fingerprint == self.incident_fingerprint and now - self.incident_last_at < 30:
            return
        self.incident_fingerprint = fingerprint
        self.incident_last_at = now
        self.incidents.appendleft((datetime.now().astimezone().isoformat(timespec="seconds"),
                                   category, detail))
        if category == "Cámara":
            self.session_had_camera_error = True
        if category == "ESP32":
            self.session_had_esp_error = True
        self.update_session_summary()
        if self.current_page == "Incidentes":
            self.refresh_incidents()

    def refresh_incidents(self) -> None:
        for child in self.incident_list.winfo_children():
            child.destroy()
        if not self.incidents:
            ctk.CTkLabel(self.incident_list,
                         text="◇  Sin incidentes técnicos en esta sesión.",
                         text_color=MUTED, font=ctk.CTkFont("Segoe UI", 13)).pack(
                             anchor="w", padx=12, pady=20)
            return
        for when, category, detail in self.incidents:
            item = ctk.CTkFrame(self.incident_list, fg_color=SURFACE_2, corner_radius=11)
            item.pack(fill="x", padx=2, pady=4)
            ctk.CTkLabel(item, text=f"{when[:19].replace('T', ' ')} · {category}",
                         text_color=AMBER, font=ctk.CTkFont("Segoe UI", 11, "bold")).pack(
                             anchor="w", padx=14, pady=(8, 0))
            ctk.CTkLabel(item, text=detail, text_color=TEXT,
                         font=ctk.CTkFont("Segoe UI", 11), wraplength=650,
                         justify="left").pack(anchor="w", padx=14, pady=(0, 8))

    def refresh_assistant(self) -> None:
        if self.current_page != "Incidentes":
            return
        if self.status_state == "ERROR DE MODELO":
            message = "Modelos no cargados: verifica los archivos ONNX y reinicia la aplicación."
        elif not self.camera_connected:
            message = "Cámara sin señal: revisa permisos de Windows y pulsa Reintentar cámara."
        elif (self.mode == "hardware" and not self.device_connected) or (self.led_port and self.led_error_seen):
            message = "ESP32 sin respuesta: comprueba COM, cable USB y que otro programa no use el puerto."
        elif any(active and uid not in self.access_pin_ids for uid, _, active in self.people):
            message = "Usuario incompleto: abre Usuarios y completa su PIN con autorización administrativa."
        elif any(category == "Respaldo" for _, category, _ in self.incidents):
            message = "Respaldo fallido: elige otra ubicación y usa Verificar respaldo en Configuración."
        elif not self.history_rows:
            message = "Historial vacío: aparecerán aquí los intentos de acceso guardados."
        else:
            message = "Sistema operativo. Consulta los incidentes cuando aparezca un error."
        if getattr(self, "assistant_message", None) != message:
            self.assistant_message = message
            self.assistant_label.configure(text=f"ASISTENTE  ·  {message}")

    def submit_admin(self, operation, success, error_label, button) -> None:
        def worker() -> None:
            try:
                operation()
                self.admin_results.put((success, error_label, button, None))
            except (ValueError, PermissionError, OSError) as exc:
                self.admin_results.put((success, error_label, button, str(exc)))
        threading.Thread(target=worker, daemon=True, name="safebox-admin").start()

    def pin_window(self, title: str, height: int = 235):
        if self.admin_window is not None and self.admin_window.winfo_exists():
            self.admin_window.focus()
            return None
        window = ctk.CTkToplevel(self, fg_color=BG)
        self.admin_window = window
        window.title(f"SAFEBOX AI · {title}")
        window.geometry(f"410x{height}")
        window.resizable(False, False)
        window.transient(self)
        window.grab_set()
        window.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(window, text=title, text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 21, "bold")).grid(
            row=0, column=0, sticky="w", padx=26, pady=(22, 8))
        return window

    def close_pin_window(self) -> None:
        if self.admin_window is not None and self.admin_window.winfo_exists():
            self.admin_window.grab_release()
            self.admin_window.destroy()
        self.admin_window = None

    def prompt_create_pin(self) -> None:
        if self.admin.configured or not self.running:
            return
        window = self.pin_window("Crear PIN de administrador", 280)
        if window is None:
            return
        window.protocol("WM_DELETE_WINDOW", self.close)
        first = ctk.CTkEntry(window, placeholder_text="PIN de 6 a 12 dígitos", show="•",
                             height=40, corner_radius=20)
        first.grid(row=1, column=0, sticky="ew", padx=26, pady=6)
        second = ctk.CTkEntry(window, placeholder_text="Repetir PIN", show="•",
                              height=40, corner_radius=20)
        second.grid(row=2, column=0, sticky="ew", padx=26, pady=6)
        error = ctk.CTkLabel(window, text="El PIN protege altas, bajas y ajustes.", text_color=MUTED)
        error.grid(row=3, column=0, sticky="w", padx=26, pady=5)
        button = ctk.CTkButton(window, text="Crear PIN", height=40, corner_radius=20,
                               fg_color=CYAN, hover_color="#1DB954", text_color=BG)
        button.grid(row=4, column=0, sticky="ew", padx=26, pady=8)
        def submit() -> None:
            pin = first.get()
            if pin != second.get():
                error.configure(text="Los PIN no coinciden", text_color=RED)
                return
            button.configure(state="disabled")
            def done() -> None:
                self.admin_until = time.monotonic() + 120
                self.backend.send("admin_ready", True)
                self.close_pin_window()
                self.notify("PIN de administrador creado")
            self.submit_admin(lambda: self.admin.create(pin), done, error, button)
        button.configure(command=submit)
        first.focus_set()

    def require_admin(self, title: str, action) -> None:
        if not self.admin.configured:
            self.prompt_create_pin()
            return
        if time.monotonic() < self.admin_until:
            self.admin_until = time.monotonic() + 120
            action()
            return
        window = self.pin_window(title)
        if window is None:
            return
        window.protocol("WM_DELETE_WINDOW", self.close_pin_window)
        entry = ctk.CTkEntry(window, placeholder_text="PIN de administrador", show="•",
                             height=42, corner_radius=21)
        entry.grid(row=1, column=0, sticky="ew", padx=26, pady=10)
        error = ctk.CTkLabel(window, text="Sesión de 2 minutos tras verificar.", text_color=MUTED)
        error.grid(row=2, column=0, sticky="w", padx=26)
        button = ctk.CTkButton(window, text="Verificar PIN", height=40, corner_radius=20,
                               fg_color=CYAN, hover_color="#1DB954", text_color=BG)
        button.grid(row=3, column=0, sticky="ew", padx=26, pady=14)
        def submit() -> None:
            pin = entry.get()
            button.configure(state="disabled")
            def done() -> None:
                self.admin_until = time.monotonic() + 120
                self.close_pin_window()
                action()
            self.submit_admin(lambda: self.admin.verify(pin), done, error, button)
        button.configure(command=submit)
        entry.focus_set()

    def change_pin(self) -> None:
        window = self.pin_window("Cambiar PIN", 330)
        if window is None:
            return
        window.protocol("WM_DELETE_WINDOW", self.close_pin_window)
        entries = []
        for row, placeholder in enumerate(("PIN actual", "PIN nuevo", "Repetir PIN nuevo"), 1):
            entry = ctk.CTkEntry(window, placeholder_text=placeholder, show="•",
                                 height=40, corner_radius=20)
            entry.grid(row=row, column=0, sticky="ew", padx=26, pady=6)
            entries.append(entry)
        error = ctk.CTkLabel(window, text="", text_color=RED)
        error.grid(row=4, column=0, sticky="w", padx=26)
        button = ctk.CTkButton(window, text="Guardar PIN", height=40, corner_radius=20,
                               fg_color=CYAN, hover_color="#1DB954", text_color=BG)
        button.grid(row=5, column=0, sticky="ew", padx=26, pady=12)
        def submit() -> None:
            old_pin, new_pin, repeated = (entry.get() for entry in entries)
            if new_pin != repeated:
                error.configure(text="Los PIN nuevos no coinciden")
                return
            button.configure(state="disabled")
            def done() -> None:
                self.admin_until = 0.0
                self.close_pin_window()
                self.notify("PIN actualizado; sesión bloqueada")
            self.submit_admin(lambda: self.admin.change(old_pin, new_pin), done, error, button)
        button.configure(command=submit)

    def set_access_pin(self, person_id: str, name: str, only_missing: bool = False) -> None:
        if only_missing and person_id in self.access_pin_ids:
            self.notify("El usuario ya tiene un PIN; no se sobrescribió", error=True)
            return
        window = self.pin_window(f"PIN de acceso · {name}", 280)
        if window is None:
            return
        window.protocol("WM_DELETE_WINDOW", self.close_pin_window)
        first = ctk.CTkEntry(window, placeholder_text="PIN de acceso · 6 a 12 dígitos",
                             show="•", height=40, corner_radius=20)
        first.grid(row=1, column=0, sticky="ew", padx=26, pady=6)
        second = ctk.CTkEntry(window, placeholder_text="Repetir PIN de acceso",
                              show="•", height=40, corner_radius=20)
        second.grid(row=2, column=0, sticky="ew", padx=26, pady=6)
        hint = ctk.CTkLabel(window, text="Este PIN se pedirá después del rostro.", text_color=MUTED)
        hint.grid(row=3, column=0, sticky="w", padx=26, pady=5)
        def save() -> None:
            pin = first.get()
            if pin != second.get() or not re.fullmatch(r"[0-9]{6,12}", pin):
                hint.configure(text="Los PIN deben coincidir y tener 6 a 12 dígitos", text_color=RED)
                return
            self.backend.send("complete_access_pin" if only_missing else "set_access_pin",
                              (person_id, pin))
            self.close_pin_window()
        ctk.CTkButton(window, text="Guardar PIN", height=40, corner_radius=20,
                      fg_color=CYAN, hover_color="#1DB954", text_color=BG,
                      command=save).grid(row=4, column=0, sticky="ew", padx=26, pady=8)
        first.focus_set()

    def show_access_pin(self, person_id: str, name: str, track_id: int,
                        deadline: float | None = None) -> None:
        if (self.admin_window is not None and self.admin_window.winfo_exists()) or (
                self.register_window is not None and self.register_window.winfo_exists()):
            self.backend.send("cancel_access")
            return
        if (self.access_window is not None and self.access_window.winfo_exists() and
                self.access_request == person_id and self.access_track == track_id):
            self.access_window.focus()
            return
        self.close_access_pin(False)
        window = ctk.CTkToplevel(self, fg_color=BG)
        self.access_window = window
        self.sync_capture_activity()
        self.access_request = person_id
        self.access_track = track_id
        self.access_deadline = deadline if deadline is not None else time.monotonic() + 15.0
        window.title("SAFEBOX AI · Segundo factor")
        window.geometry("420x290")
        window.resizable(False, False)
        window.transient(self)
        window.grab_set()
        window.grid_columnconfigure(0, weight=1)
        window.protocol("WM_DELETE_WINDOW", self.close_access_pin)
        ctk.CTkLabel(window, text="Confirmar acceso", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 21, "bold")).grid(
            row=0, column=0, sticky="w", padx=26, pady=(22, 5))
        ctk.CTkLabel(window, text=f"Posible coincidencia: {name}. Mantén el rostro en cámara.",
                     text_color=MUTED, wraplength=370).grid(row=1, column=0, sticky="w", padx=26)
        self.access_entry = ctk.CTkEntry(window, placeholder_text="PIN de acceso", show="•",
                                         height=40, corner_radius=20)
        self.access_entry.grid(row=2, column=0, sticky="ew", padx=26, pady=10)
        self.access_error = ctk.CTkLabel(window, text="Tiempo máximo: 15 segundos", text_color=MUTED)
        self.access_error.grid(row=3, column=0, sticky="w", padx=26)
        self.pin_countdown = ctk.CTkProgressBar(window, height=6,
                                                progress_color=CYAN, fg_color=SURFACE_2)
        self.pin_countdown.grid(row=4, column=0, sticky="ew", padx=26, pady=(6, 4))
        self.pin_countdown_label = ctk.CTkLabel(window, text="", text_color=MUTED,
                                                font=ctk.CTkFont("Segoe UI", 10))
        self.pin_countdown_label.grid(row=5, column=0, sticky="w", padx=26)
        self.access_button = ctk.CTkButton(window, text="Verificar y abrir", height=40,
                                           corner_radius=20, fg_color=CYAN,
                                           hover_color="#1DB954", text_color=BG,
                                           command=self.submit_access_pin)
        self.access_button.grid(row=6, column=0, sticky="ew", padx=26, pady=(6, 10))
        self.register_icon(self.access_button, "KEY", "text", 18)
        self.access_entry.bind("<Return>", lambda _event: self.submit_access_pin())
        self.access_entry.focus_set()
        self.update_pin_countdown()

    def update_pin_countdown(self) -> None:
        self.pin_tick_job = None
        if self.access_window is None or not self.access_window.winfo_exists():
            return
        remaining = max(0.0, self.access_deadline - time.monotonic())
        seconds = int(remaining + 0.999)
        self.pin_countdown.set(min(1.0, remaining / 15.0))
        self.pin_countdown_label.configure(
            text=f"{seconds} s para confirmar el PIN" if seconds else "Tiempo agotado",
            text_color=AMBER if seconds <= 5 else MUTED)
        if remaining > 0:
            self.pin_tick_job = self.after(200, self.update_pin_countdown)
        else:
            self.access_button.configure(state="disabled")

    def submit_access_pin(self) -> None:
        if self.access_request is None:
            return
        pin = self.access_entry.get()
        if not re.fullmatch(r"[0-9]{6,12}", pin):
            self.access_error.configure(text="Introduce 6 a 12 dígitos", text_color=RED)
            return
        self.access_button.configure(state="disabled")
        self.backend.send("access_pin", (self.access_request, pin, self.access_track))

    def close_access_pin(self, cancel: bool = True) -> None:
        if self.pin_tick_job is not None:
            self.after_cancel(self.pin_tick_job)
            self.pin_tick_job = None
        if cancel and self.access_request is not None:
            self.backend.send("cancel_access")
        if self.access_window is not None and self.access_window.winfo_exists():
            self.access_entry.delete(0, "end")
            self.access_window.grab_release()
            self.forget_icons(self.access_window)
            self.access_window.destroy()
        self.access_window = None
        self.access_request = None
        self.access_track = None
        self.access_deadline = 0.0
        if self.running:
            self.sync_capture_activity()

    def lock_admin_session(self) -> None:
        self.admin_until = 0.0
        self.notify("Sesión administrativa cerrada")

    def save_settings(self) -> None:
        try:
            values = {
                "similarity_threshold": float(self.threshold_entry.get().replace(",", ".")),
                "authorization_seconds": int(self.duration_entry.get()),
                "camera_index": int(self.camera_choice.get().split()[-1]),
                "mode": "hardware" if self.mode_choice.get() == "Hardware" else "simulacion",
                "port": self.port_entry.get().strip().upper(),
            }
            self.settings.validate(values)
        except (ValueError, TypeError) as exc:
            self.notify(f"Ajustes inválidos: {exc}", error=True)
            return
        self.require_admin("Guardar configuración", lambda: self.apply_settings(values))

    def apply_settings(self, values: dict) -> None:
        if values["mode"] != self.mode or values["port"] != (self.backend.port or ""):
            self.pending_settings = values
            self.notify("Comprobando conexión serial…")
            self.backend.send("connect_device", (values["mode"], values["port"]))
        else:
            self.commit_settings(values)

    def commit_settings(self, values: dict) -> None:
        try:
            self.settings.save(values)
        except (ValueError, OSError) as exc:
            self.notify(f"No se guardaron los ajustes: {exc}", error=True)
            return
        self.backend.send("update_settings", (values["similarity_threshold"],
                                               values["authorization_seconds"]))
        if values["camera_index"] != self.camera_index:
            self.backend.select_camera(values["camera_index"])
        self.camera_index = values["camera_index"]
        self.authorization_seconds = values["authorization_seconds"]
        self.similarity_threshold = values["similarity_threshold"]
        self.threshold_label.configure(text=f"UMBRAL  ≥ {self.similarity_threshold:.3f}   ·   MAYOR ES MEJOR")
        self.mode = values["mode"]
        self.backend.port = values["port"]
        self.notify("Ajustes guardados")

    def export_history(self) -> None:
        self.require_admin("Exportar historial", self.choose_history_export)

    def choose_history_export(self) -> None:
        path = filedialog.asksaveasfilename(parent=self, title="Exportar historial",
                                            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            self.backend.send("export_history", Path(path))

    def choose_backup(self) -> None:
        name = f"safebox-respaldo-{datetime.now():%Y%m%d-%H%M%S}.zip"
        path = filedialog.asksaveasfilename(
            parent=self, title="Guardar respaldo local sensible", initialfile=name,
            defaultextension=".zip", filetypes=[("ZIP", "*.zip")])
        if path:
            self.backend.send("backup", Path(path))
            self.notify("Creando respaldo local…")

    def run_utility(self, name: str, operation) -> None:
        def worker() -> None:
            try:
                result = operation()
                self.utility_results.put((name, result, None))
            except Exception as exc:
                self.utility_results.put((name, None, str(exc)))
        threading.Thread(target=worker, daemon=True, name=f"safebox-{name}").start()

    def run_integrity_check(self) -> None:
        self.data_diagnostics.configure(text="Comprobando integridad en solo lectura…", text_color=MUTED)
        self.run_utility("integrity", lambda: check_data_integrity(self.backend.data_dir))

    def run_storage_check(self) -> None:
        self.data_diagnostics.configure(text="Calculando espacio ocupado…", text_color=MUTED)
        backups = list(self.backup_paths)
        self.run_utility("storage", lambda: storage_sizes(self.backend.data_dir, backups))

    def run_backup_verification(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="Verificar respaldo ZIP", initialdir=str(self.last_backup_path.parent)
            if self.last_backup_path else None, filetypes=[("ZIP", "*.zip")])
        if path:
            self.data_diagnostics.configure(text="Verificando ZIP…", text_color=MUTED)
            self.run_utility("verify_backup", lambda: (path, verify_local_backup(Path(path))))

    def set_visual_theme(self, name: str) -> None:
        if name not in THEMES:
            return
        self.theme_name = name
        self.apply_visual_palette()

    def toggle_high_contrast(self) -> None:
        self.high_contrast = not self.high_contrast
        self.contrast_button.configure(text=f"Alto contraste: {'sí' if self.high_contrast else 'no'}")
        self.apply_visual_palette()

    def apply_visual_palette(self) -> None:
        global BG, SURFACE, SURFACE_2, BORDER, CARD_BORDER, GREEN_DIM, CYAN, MUTED, TEXT
        old = self.current_palette
        new = dict(THEMES[self.theme_name])
        if self.high_contrast:
            new.update(BG="#050505", SURFACE="#111111", SURFACE_2="#292929",
                       BORDER="#EAEAEA", CARD_BORDER="#D0D0D0", GREEN_DIM="#294439",
                       MUTED="#E0E0E0")
        new_hover = {"Verde tecnológico": "#1DB954", "Azul cibernético": "#389FD4",
                     "Minimalista oscuro": "#B8C0BD"}[self.theme_name]
        colors = {old[key]: value for key, value in new.items()}
        colors[self.current_hover] = new_hover
        for widget in self._walk_widgets(self):
            updates = {}
            for option in ("fg_color", "text_color", "border_color", "hover_color",
                           "button_color", "button_hover_color", "progress_color"):
                try:
                    value = widget.cget(option)
                except (ValueError, AttributeError, TypeError, TclError):
                    continue
                if isinstance(value, str) and value.upper() in {key.upper() for key in colors}:
                    match = next(key for key in colors if key.upper() == value.upper())
                    updates[option] = colors[match]
            if updates:
                try:
                    widget.configure(**updates)
                except (ValueError, TypeError, TclError):
                    pass
        self.current_palette = new
        self.current_hover = new_hover
        BG, SURFACE, SURFACE_2, BORDER = (new[key] for key in ("BG", "SURFACE", "SURFACE_2", "BORDER"))
        CARD_BORDER, GREEN_DIM, CYAN, MUTED, TEXT = (
            new[key] for key in ("CARD_BORDER", "GREEN_DIM", "CYAN", "MUTED", "TEXT"))
        STATE_COLORS["ESCANEANDO"] = MUTED
        STATE_COLORS["AUTORIZADO"] = CYAN
        self.nav_progress.configure(bg=BG)
        self.nav_progress.itemconfigure(self.nav_progress_item, fill=CYAN)
        self.history_chart.configure(bg=SURFACE)
        self.heatmap_chart.configure(bg=SURFACE)
        self.diagnostic_cache.clear()
        self.refresh_diagnostics()
        self.refresh_flow()
        self.refresh_led_indicator()
        self.render_history_chart()
        self.render_heatmap()
        self._rendered_users = None
        self._rendered_recent = None
        self.refresh_users()
        self.refresh_history()
        if self.current_page == "Historial":
            self.refresh_full_history()
        if self.current_page == "Incidentes":
            self.refresh_incidents()
        if self.icons_ready:
            if self.icon_job is not None:
                self.after_cancel(self.icon_job)
            self.start_icons()

    @staticmethod
    def _walk_widgets(root):
        yield root
        for child in root.winfo_children():
            yield from SafeApp._walk_widgets(child)

    def export_filtered_history(self) -> None:
        self.require_admin("Exportar historial filtrado", self.choose_filtered_export)

    def choose_filtered_export(self) -> None:
        try:
            start, end = self.selected_history_dates()
        except ValueError as exc:
            self.notify(f"Fecha inválida: {exc}", error=True)
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Exportar resultados filtrados", defaultextension=".csv",
            filetypes=[("CSV", "*.csv")])
        if path:
            self.backend.send("export_filtered_history", (
                Path(path), self.history_search.get().strip(), self.history_filter.get(), start, end))

    def export_graphs(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self, title="Guardar gráficos para presentación", defaultextension=".png",
            filetypes=[("PNG", "*.png")])
        if path:
            stats = getattr(self, "chart_stats", (0, 0, 0))
            daily, hourly = list(self.daily_rows), list(self.hourly_rows)
            accent = CYAN
            self.run_utility("graphs", lambda: (save_graphs_png(Path(path), stats, daily, hourly, accent), path)[1])

    def render_heatmap(self) -> None:
        if self.current_page != "Historial":
            return
        chart = self.heatmap_chart
        signature = (tuple(self.hourly_rows), chart.winfo_width(), SURFACE, SURFACE_2, CYAN)
        if signature == self.heatmap_drawn:
            return
        self.heatmap_drawn = signature
        chart.delete("all")
        chart.create_text(4, 12, text="MAPA DE CALOR · 7 DÍAS POR HORA LOCAL",
                          anchor="w", fill=TEXT, font=("Segoe UI", 10, "bold"))
        today = datetime.now().astimezone().date()
        days = [(today - timedelta(days=offset)).isoformat() for offset in range(6, -1, -1)]
        counts = {(day, hour): count for day, hour, count in self.hourly_rows}
        highest = max(counts.values(), default=0)
        width = max(chart.winfo_width(), 600)
        cell = max(8, (width - 80) / 24)
        for row, day in enumerate(days):
            y = 29 + row * 10
            chart.create_text(4, y + 4, text=day[5:], anchor="w", fill=MUTED,
                              font=("Segoe UI", 8))
            for hour in range(24):
                count = counts.get((day, hour), 0)
                fill = mix_color(SURFACE_2, CYAN, count / highest) if highest else SURFACE_2
                x = 55 + hour * cell
                chart.create_rectangle(x, y, x + cell - 2, y + 8, fill=fill, outline="")
        if not highest:
            chart.create_text(width - 8, 12, text="Sin actividad", anchor="e", fill=MUTED,
                              font=("Segoe UI", 9))

    def selected_history_dates(self) -> tuple[date | None, date | None]:
        selected = self.date_filter.get()
        today = datetime.now().astimezone().date()
        if selected == "Hoy":
            return today, today
        if selected == "Últimos 7 días":
            return today - timedelta(days=6), today
        if selected == "Últimos 30 días":
            return today - timedelta(days=29), today
        if selected == "Personalizado":
            start = date.fromisoformat(self.date_start.get().strip()) if self.date_start.get().strip() else None
            end = date.fromisoformat(self.date_end.get().strip()) if self.date_end.get().strip() else None
            if start and end and start > end:
                raise ValueError("La fecha inicial debe ser anterior a la final")
            if start is None and end is None:
                raise ValueError("Indica al menos una fecha para el rango personalizado")
            return start, end
        return None, None

    def render_history_chart(self) -> None:
        if self.current_page != "Historial":
            return
        chart = self.history_chart
        signature = (getattr(self, "chart_stats", (0, 0, 0)), tuple(self.daily_rows),
                     chart.winfo_width(), SURFACE, CYAN)
        if signature == self.chart_drawn:
            return
        self.chart_drawn = signature
        chart.delete("all")
        width = max(chart.winfo_width(), 100)
        total, allowed, denied = getattr(self, "chart_stats", (0, 0, 0))
        chart.create_text(4, 12, text=f"{total} INTENTOS  ·  {allowed} AUTORIZADOS  ·  {denied} DENEGADOS",
                          anchor="w", fill=TEXT, font=("Segoe UI", 10, "bold"))
        if not total:
            chart.create_text(4, 67, text="Sin actividad registrada todavía", anchor="w",
                              fill=MUTED, font=("Segoe UI", 11))
            return
        bar_width = max(100, min(270, width // 3))
        chart.create_rectangle(4, 32, 4 + bar_width * allowed / total, 46, fill=CYAN, outline="")
        chart.create_rectangle(4 + bar_width * allowed / total, 32, 4 + bar_width, 46,
                               fill=RED, outline="")
        chart.create_text(4, 55, text=f"{allowed / total:.1%} autorizados · {denied / total:.1%} denegados",
                          anchor="nw", fill=MUTED, font=("Segoe UI", 9))
        daily = self.daily_rows[-14:]
        if daily:
            start_x = bar_width + 34
            chart_width = max(80, width - start_x - 10)
            maximum = max(1, *(a + d for _, a, d in daily))
            slot = chart_width / len(daily)
            for index, (day, a, d) in enumerate(daily):
                x = start_x + index * slot + 2
                h_a, h_d = 54 * a / maximum, 54 * d / maximum
                chart.create_rectangle(x, 88 - h_a, x + max(2, slot - 5), 88,
                                       fill=CYAN, outline="")
                chart.create_rectangle(x, 88 - h_a - h_d, x + max(2, slot - 5), 88 - h_a,
                                       fill=RED, outline="")
                if index in (0, len(daily) - 1):
                    chart.create_text(x, 94, text=day[5:], anchor="nw", fill=MUTED,
                                      font=("Segoe UI", 8))

    def schedule_history_search(self, _event=None) -> None:
        if self.search_job is not None:
            self.after_cancel(self.search_job)
        self.search_job = self.after(250, self.reset_history_search)

    def reset_history_search(self) -> None:
        self.history_offset = 0
        self.request_history()

    def change_history_page(self, direction: int) -> None:
        self.history_offset = max(0, self.history_offset + direction * 30)
        self.request_history()

    def request_history(self) -> None:
        self.search_job = None
        if self.backend_ready:
            try:
                self.history_dates = self.selected_history_dates()
            except ValueError as exc:
                self.history_note.configure(text=f"Fecha inválida: {exc}", text_color=RED)
                return
            self.history_note.configure(text_color=MUTED)
            self.history_note.configure(text="Buscando…")
            self.backend.send("search_history", (
                self.history_search.get().strip(), self.history_filter.get(), self.history_offset,
                *self.history_dates))

    def refresh_full_history(self) -> None:
        for child in self.history_full_list.winfo_children():
            child.destroy()
        rows = self.history_results[:30]
        self.history_note.configure(text=f"Intentos {self.history_offset + 1}–{self.history_offset + len(rows)}"
                                    if rows else "Sin resultados")
        self.history_prev.configure(state="normal" if self.history_offset else "disabled")
        self.history_next.configure(state="normal" if len(self.history_results) > 30 else "disabled")
        if not rows:
            ctk.CTkLabel(self.history_full_list,
                         text="▤  SIN RESULTADOS\nAjusta los filtros o espera el primer intento de acceso.",
                         text_color=MUTED, font=ctk.CTkFont("Segoe UI", 13),
                         justify="left").pack(
                anchor="w", padx=10, pady=22)
            return
        for date, user, result, distance, similarity, capture in rows:
            item = ctk.CTkFrame(self.history_full_list, fg_color=SURFACE_2, corner_radius=12)
            item.pack(fill="x", pady=4, padx=2)
            item.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(item, text="●", text_color=CYAN if result == "AUTORIZADO" else RED,
                         font=ctk.CTkFont("Segoe UI", 17)).grid(
                row=0, column=0, rowspan=2, padx=(14, 2), pady=8)
            ctk.CTkLabel(item, text=user, text_color=TEXT,
                         font=ctk.CTkFont("Segoe UI", 12, "bold")).grid(
                row=0, column=1, sticky="w", padx=12, pady=(9, 0))
            color = CYAN if result == "AUTORIZADO" else RED
            ctk.CTkLabel(item, text=result, text_color=color,
                         font=ctk.CTkFont("Segoe UI", 11, "bold")).grid(
                row=0, column=2, sticky="e", padx=12, pady=(9, 0))
            metric = (f" · SFace coseno {similarity:.3f}" if similarity is not None
                      else f" · LBPH distancia {distance:.1f}" if distance is not None else "")
            ctk.CTkLabel(item, text=date[:19].replace("T", "  ") + metric,
                         text_color=MUTED, font=ctk.CTkFont("Segoe UI", 10)).grid(
                row=1, column=1, sticky="w", padx=12, pady=(0, 9))
            if capture:
                ctk.CTkButton(item, text="Ver captura", width=100, height=27,
                              corner_radius=14, fg_color=BORDER, hover_color="#555555",
                              text_color=TEXT,
                              command=lambda filename=capture: self.show_capture(filename)).grid(
                    row=1, column=2, padx=12, pady=(0, 8))

    def show_capture(self, filename: str) -> None:
        path = self.backend.data_dir / "capturas" / filename
        if path.name != filename or not path.is_file():
            self.notify("La captura ya no existe.", error=True)
            return
        window = ctk.CTkToplevel(self, fg_color=BG)
        window.title("SAFEBOX AI · Captura")
        window.geometry("720x570")
        window.transient(self)
        ctk.CTkLabel(window, text=filename, text_color=MUTED,
                     font=ctk.CTkFont("Segoe UI", 12)).pack(pady=(18, 10))
        try:
            with Image.open(path) as source:
                picture = source.copy()
            picture.thumbnail((680, 480))
            photo = ctk.CTkImage(light_image=picture, dark_image=picture,
                                 size=picture.size)
            label = ctk.CTkLabel(window, text="", image=photo)
            label.image = photo
            label.pack(expand=True)
        except (OSError, ValueError) as exc:
            window.destroy()
            self.notify(f"No se pudo abrir la captura: {exc}", error=True)

    @staticmethod
    def device_label(parent, title: str, initial: str) -> ctk.CTkLabel:
        ctk.CTkLabel(parent, text=title, text_color=MUTED,
                     font=ctk.CTkFont("Segoe UI", 9, "bold")).pack(
            anchor="w", padx=14, pady=(12, 2)
        )
        value = ctk.CTkLabel(parent, text=initial, text_color=TEXT,
                             font=ctk.CTkFont("Segoe UI", 14, "bold"))
        value.pack(anchor="w", padx=14, pady=(0, 12))
        return value

    def update_user_count(self) -> None:
        self.users_count.configure(text=f"USUARIOS REGISTRADOS  {len(self.known_users):02d}")

    def update_clock(self) -> None:
        if self.running:
            self.clock_label.configure(text=datetime.now().strftime("%H:%M:%S"))
            self.clock_job = self.after(1000, self.update_clock)

    def open_camera(self) -> None:
        if self.camera_connected or not self.backend.retry_camera():
            return
        self.camera_badge.configure(text="●  CONECTANDO", text_color=BLUE)
        self.retry_button.grid_remove()

    def on_camera_resize(self, _event=None) -> None:
        canvas = self.camera_canvas
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        if (width, height) == self.last_canvas_size:
            return
        self.last_canvas_size = (width, height)
        canvas.coords(self.border_item, 1, 1, width - 2, height - 2)
        canvas.coords(self.scan_text_item, width - 18, 19)
        if not self.camera_connected or self.camera_paused:
            self.draw_placeholder()
        else:
            self.position_boxes()

    def draw_placeholder(self) -> None:
        canvas = self.camera_canvas
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        signature = (width, height, self.status_state, self.camera_paused)
        if signature == self.placeholder_drawn and canvas.find_withtag("placeholder"):
            return
        self.placeholder_drawn = signature
        canvas.delete("placeholder")
        cx, cy = width // 2, height // 2
        canvas.create_oval(cx - 34, cy - 68, cx + 34, cy,
                           outline=BORDER, width=2, tags="placeholder")
        canvas.create_line(cx - 55, cy + 16, cx + 55, cy + 16,
                           fill=BORDER, width=2, tags="placeholder")
        loading = self.status_state == "INICIANDO"
        canvas.create_text(cx, cy + 55,
                           text="CÁMARA EN PAUSA" if self.camera_paused else
                           "INICIANDO CÁMARA" if loading else
                           "MODELO NO DISPONIBLE" if self.status_state == "ERROR DE MODELO" else
                           "CÁMARA SIN SEÑAL", fill=TEXT,
                           font=("Segoe UI", 17, "bold"), tags="placeholder")
        canvas.create_text(cx, cy + 82,
                           text="Se reanudará al volver al escáner" if self.camera_paused else
                           "Preparando captura…" if loading else
                           "Ejecuta download_models.py y reinicia" if self.status_state == "ERROR DE MODELO" else
                           "Usa «Reintentar cámara» al conectar el dispositivo",
                           fill=MUTED, font=("Segoe UI", 10), tags="placeholder")

    def animate_state_color(self, start: str, target: str, step: int = 0) -> None:
        if not self.running:
            return
        color = mix_color(start, target, (step + 1) / 8)
        self.state_chip.configure(text_color=color)
        self.identity_card.configure(border_color=mix_color(CARD_BORDER, color, 0.55))
        if step < 7:
            self.transition_job = self.after(
                28, lambda: self.animate_state_color(start, target, step + 1)
            )
        else:
            self.transition_job = None

    def set_state(
        self, state: str, detail: str, user: str = "—", similarity: float | None = None
    ) -> None:
        color = STATE_COLORS[state]
        if state == "AUTORIZADO" and self.status_state != "AUTORIZADO":
            self.blue_led_until = time.monotonic() + 3.0
            if self.led_off_job is not None:
                self.after_cancel(self.led_off_job)
            self.led_off_job = self.after(3000, self.finish_blue_led_indicator)
        elif state == "DENEGADO":
            self.blue_led_until = 0.0
            if self.led_off_job is not None:
                self.after_cancel(self.led_off_job)
                self.led_off_job = None
        if state != self.status_state:
            if self.transition_job is not None:
                self.after_cancel(self.transition_job)
                self.transition_job = None
            old = self.status_color
            self.status_color = color
            self.animate_state_color(old, color)
            self.register_icon(self.state_chip, {
                "AUTORIZADO": "SHIELD_CHECK", "DENEGADO": "SHIELD_X",
                "PIN REQUERIDO": "KEY", "ERROR DE MODELO": "ALERT_TRIANGLE",
                "CÁMARA DESCONECTADA": "VIDEO_OFF"}.get(state, "SCAN"),
                "red" if state in {"DENEGADO", "ERROR DE MODELO"} else
                "amber" if state in {"PIN REQUERIDO", "CÁMARA DESCONECTADA"} else "accent", 16)
        self.status_state = state
        self.camera_canvas.itemconfigure(self.scan_text_item,
                                         text=f"●  {state}" if self.camera_connected else "●  BIOMETRIC SCAN")
        self.state_chip.configure(text=state)
        title = user if state in {"AUTORIZADO", "DENEGADO", "REGISTRANDO"} and user != "—" else {
            "INICIANDO": "Iniciando sistema",
            "ESCANEANDO": "Esperando identidad",
            "VERIFICANDO": "Verificando rostro",
            "PIN REQUERIDO": "Verificación en dos pasos",
            "REGISTRANDO": "Capturando rostro",
            "AUTORIZADO": "Identidad verificada",
            "DENEGADO": "Acceso denegado",
            "CÁMARA DESCONECTADA": "Cámara sin señal",
            "ERROR DE MODELO": "Reconocimiento no disponible",
        }[state]
        self.identity_title.configure(text=title)
        self.detail_label.configure(text=detail)
        self.distance_label.configure(text=f"{similarity:.3f}" if similarity is not None else "—")
        scan_text = f"{state} · {user}" if user != "—" else state
        if similarity is not None:
            scan_text += f" · coseno {similarity:.3f}"
        self.scan_result.configure(text=scan_text, text_color=color)
        self.refresh_flow()
        self.refresh_led_indicator()

    def refresh_flow(self) -> None:
        if self.lock_state == "ESPERANDO CIERRE":
            current = 4
        elif self.lock_state == "DESBLOQUEADA":
            current = 3
        elif self.status_state == "AUTORIZADO":
            current = 2
        elif self.status_state == "PIN REQUERIDO":
            current = 1
        else:
            current = 0
        for index, label in enumerate(self.flow_labels):
            active = index == current
            completed = index < current
            color = RED if active and self.status_state == "DENEGADO" else (
                AMBER if active and current == 4 else CYAN if active or completed else MUTED)
            label.configure(text_color=color,
                            fg_color=GREEN_DIM if completed else SURFACE_2)

    def refresh_diagnostics(self) -> None:
        if self.current_page != "Configuración":
            return
        device = ("CONECTADO" if self.device_connected else "SIN RESPUESTA") if self.mode == "hardware" else (
            "LED CONECTADO" if self.led_connected else "SIN RESPUESTA" if self.led_error_seen else
            "CONECTANDO" if self.led_port else "SOLO SIM")
        values = {
            "CÁMARA": ("EN VIVO" if self.camera_connected else "SIN SEÑAL",
                       CYAN if self.camera_connected else AMBER),
            "ROSTRO": ("LISTO" if self.backend_ready else "ERROR" if self.status_state == "ERROR DE MODELO"
                       else "CARGANDO", CYAN if self.backend_ready else RED if self.status_state == "ERROR DE MODELO" else AMBER),
            "ESP32": (device, CYAN if self.device_connected or self.led_connected else
                      RED if device == "SIN RESPUESTA" else AMBER),
            "MG90S": (self.servo_status, CYAN if self.servo_status == "LISTO" else
                      RED if self.servo_status == "ERROR" else AMBER),
            "ACCESO": (f"{self.status_state or 'INICIANDO'} · {self.lock_state}",
                       RED if self.lock_state == "ERROR" or self.status_state == "DENEGADO" else
                       CYAN if self.status_state == "AUTORIZADO" else AMBER),
        }
        for title, value in values.items():
            if self.diagnostic_cache.get(title) != value:
                self.diagnostic_labels[title].configure(text=value[0], text_color=value[1])
                self.diagnostic_cache[title] = value

    def finish_blue_led_indicator(self) -> None:
        self.led_off_job = None
        self.blue_led_until = 0.0
        self.refresh_led_indicator()

    def refresh_led_indicator(self) -> None:
        if self.mode == "hardware" and self.lock_state == "ERROR":
            led, color = "ESTADO DESCONOCIDO", AMBER
        elif self.status_state == "DENEGADO":
            led, color = "ROJO · ORDENADO", RED
        elif time.monotonic() < self.blue_led_until:
            led, color = "AZUL · 3 s", BLUE
        else:
            led, color = "APAGADO", MUTED
        self.led_value.configure(text=led, text_color=color)
        self.led_lamp.configure(text_color=color)
        self.register_icon(self.led_value, "BULB", "red" if color == RED else
                           "blue" if color == BLUE else "muted", 17)

    def update_lock(self, state: str, reason: str) -> None:
        self.lock_state = state
        if self.mode == "simulacion":
            label = (f"SIM {state}\nFÍSICO DESCONOCIDO" if self.physical_unknown else
                     f"{state} · SOLO SIM")
        else:
            label = ("BLOQUEO ORDENADO\nPOSICIÓN NO VERIFICADA" if state == "BLOQUEADA" else
                     "LIBERACIÓN ORDENADA\nPOSICIÓN NO VERIFICADA" if state == "DESBLOQUEADA" else
                     "CIERRE PENDIENTE\nPOSICIÓN NO VERIFICADA" if state == "ESPERANDO CIERRE" else state)
        self.lock_value.configure(text=label, text_color=RED if state == "ERROR" else
                                  CYAN if state == "DESBLOQUEADA" else AMBER if state in {"ESPERANDO CIERRE", "AUTORIZACIÓN PENDIENTE"} else TEXT)
        lock_symbol = "PESTILLO" if state == "BLOQUEADA" else (
            "LIBERACIÓN ORDENADA" if state == "DESBLOQUEADA" else
            "CIERRE PENDIENTE" if state == "ESPERANDO CIERRE" else
            "ESTADO DESCONOCIDO" if state == "ERROR" else "VERIFICANDO")
        self.register_icon(self.lock_icon, "LOCK_OPEN" if state == "DESBLOQUEADA" else
                           "ALERT_TRIANGLE" if state == "ERROR" else "LOCK",
                           "red" if state == "ERROR" else "accent" if state == "DESBLOQUEADA" else "amber", 18)
        self.lock_icon.configure(text=lock_symbol,
                                 text_color=RED if state == "ERROR" else
                                 CYAN if state == "DESBLOQUEADA" else AMBER)
        if state in {"DESBLOQUEADA", "ESPERANDO CIERRE"}:
            self.lock_action_label.configure(
                text="CIERRA LA PUERTA MANUALMENTE Y CONFIRMA" if state == "ESPERANDO CIERRE"
                else "DESLIZA LA PUERTA; CIÉRRALA ANTES DE CONFIRMAR",
                text_color=AMBER)
            self.close_button.grid()
            self.lock_action.grid()
        elif self.mode == "hardware" and state == "ERROR":
            self.lock_action_label.configure(
                text="ESTADO FÍSICO DESCONOCIDO · REVISA COM Y MG90S EN CONFIGURACIÓN",
                text_color=RED)
            self.close_button.grid_remove()
            self.lock_action.grid()
        else:
            self.lock_action.grid_remove()
        self.refresh_led_indicator()
        self.refresh_flow()
        notice = (state, reason) if reason and state in {"ESPERANDO CIERRE", "ERROR"} else None
        if notice is not None and notice != self.last_lock_notice:
            self.notify(reason, error=state == "ERROR")
        self.last_lock_notice = notice

    def confirm_door_closed(self) -> None:
        self.require_admin("Confirmar cierre", lambda: self.backend.send("confirm_closed"))

    def test_servo(self) -> None:
        if self.mode != "hardware" or self.servo_status != "NO CONFIGURADO":
            self.notify("La prueba requiere modo hardware con el servo aún sin calibrar", error=True)
            return
        dialog = ctk.CTkInputDialog(
            title="SAFEBOX AI · Prueba MG90S",
            text="Servo desacoplado: empieza con 1500 µs; después avanza hasta 50 µs por prueba (1000–2000).")
        value = dialog.get_input()
        if value is None:
            return
        try:
            pulse = int(value.strip())
            if not 1000 <= pulse <= 2000:
                raise ValueError
        except ValueError:
            self.notify("Introduce un pulso entre 1000 y 2000 µs", error=True)
            return
        self.backend.send("test_servo", pulse)

    def stop_servo_signal(self) -> None:
        if self.mode != "hardware" or self.servo_status != "NO CONFIGURADO":
            self.notify("La parada de prueba solo está disponible antes de calibrar", error=True)
            return
        self.backend.send("servo_stop")

    def save_servo_config(self) -> None:
        if self.mode != "hardware" or self.servo_status != "DEMO":
            self.notify(f"Conecta {self.backend.port} en modo hardware DEMO antes de ajustar el MG90S", error=True)
            return
        try:
            pulses = (int(self.servo_open_entry.get().strip()),
                      int(self.servo_lock_entry.get().strip()))
            if (pulses[0] == pulses[1] or
                    any(not 1000 <= pulse <= 2000 for pulse in pulses)):
                raise ValueError
        except ValueError:
            self.notify("Usa dos pulsos distintos entre 1000 y 2000 µs", error=True)
            return
        self.servo_save_button.configure(state="disabled")
        self.backend.send("set_servo_config", pulses)

    def open_registration(self, name: str = "", authorized: bool = False) -> None:
        if not authorized:
            self.require_admin("Registrar rostro",
                               lambda: self.open_registration(name, True))
            return
        if self.register_window is not None and self.register_window.winfo_exists():
            self.register_window.focus()
            return
        if self.modal_close_job is not None:
            self.after_cancel(self.modal_close_job)
            self.modal_close_job = None
        if self.current_page not in {"Inicio", "Escáner"}:
            self.navigate("Inicio")
        window = ctk.CTkToplevel(self, fg_color=BG)
        self.register_window = window
        self.sync_capture_activity()
        window.title("SAFEBOX AI · Nuevo usuario")
        window.geometry("460x440")
        window.resizable(False, False)
        window.transient(self)
        window.grab_set()
        window.protocol("WM_DELETE_WINDOW", self.cancel_enrollment)
        window.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(window, text="Registrar identidad", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 23, "bold")).grid(
            row=0, column=0, sticky="w", padx=28, pady=(25, 5)
        )
        ctk.CTkLabel(window, text="Nombre y PIN · un rostro frente a la cámara · 20 muestras.\nBuena luz, mirada al frente y pequeños cambios de ángulo.",
                     text_color=MUTED, font=ctk.CTkFont("Segoe UI", 11),
                     wraplength=375, justify="left").grid(
            row=1, column=0, sticky="w", padx=28
        )
        self.register_entry = ctk.CTkEntry(
            window, placeholder_text="Nombre del usuario", height=43,
            fg_color=SURFACE_2, border_color=BORDER, text_color=TEXT,
            corner_radius=22,
        )
        self.register_entry.grid(row=2, column=0, sticky="ew", padx=28, pady=(23, 15))
        if name:
            self.register_entry.insert(0, name)
        self.register_pin = ctk.CTkEntry(window, placeholder_text="PIN de acceso · 6 a 12 dígitos",
                                          show="•", height=40, corner_radius=20)
        self.register_pin.grid(row=3, column=0, sticky="ew", padx=28, pady=(0, 8))
        self.register_pin_confirm = ctk.CTkEntry(window, placeholder_text="Repetir PIN de acceso",
                                                  show="•", height=40, corner_radius=20)
        self.register_pin_confirm.grid(row=4, column=0, sticky="ew", padx=28, pady=(0, 12))
        self.enrollment_progress = ctk.CTkProgressBar(
            window, height=9, progress_color=CYAN, fg_color=SURFACE_2,
        )
        self.enrollment_progress.grid(row=5, column=0, sticky="ew", padx=28)
        self.enrollment_progress.set(0)
        self.enrollment_hint = ctk.CTkLabel(
            window, text=f"0 / {SAMPLES_REQUIRED} muestras · esperando captura", text_color=MUTED,
            font=ctk.CTkFont("Consolas", 11),
        )
        self.enrollment_hint.grid(row=6, column=0, sticky="w", padx=28, pady=(8, 18))
        actions = ctk.CTkFrame(window, fg_color="transparent")
        actions.grid(row=7, column=0, sticky="ew", padx=28)
        actions.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(actions, text="Cancelar", height=39, fg_color=SURFACE_2,
                      hover_color=BORDER, text_color=TEXT,
                      corner_radius=20,
                      command=self.cancel_enrollment).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.capture_button = ctk.CTkButton(
            actions, text="Iniciar captura", height=39,
            fg_color=CYAN, hover_color="#1DB954", text_color=BG,
            corner_radius=20,
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
            command=self.start_enrollment,
        )
        self.capture_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.register_icon(self.capture_button, "SCAN", "text", 18)
        self.register_entry.focus_set()

    def start_enrollment(self, authorized: bool = False) -> None:
        if not authorized:
            self.require_admin("Registrar rostro", lambda: self.start_enrollment(True))
            return
        name = self.register_entry.get().strip()
        access_pin = self.register_pin.get()
        if access_pin != self.register_pin_confirm.get() or not re.fullmatch(r"[0-9]{6,12}", access_pin):
            self.enrollment_hint.configure(text="PIN de acceso: 6 a 12 dígitos iguales.", text_color=RED)
            return
        if not re.fullmatch(r"[\w .-]{2,40}", name, flags=re.UNICODE):
            self.enrollment_hint.configure(text="Usa 2 a 40 letras, números, espacios, puntos o guiones.", text_color=RED)
            return
        if not self.backend_ready:
            self.enrollment_hint.configure(text="El motor facial aún está iniciando.", text_color=AMBER)
            return
        if any(active and existing.casefold() == name.casefold()
               for _, existing, active in self.people):
            self.enrollment_hint.configure(text="Ese nombre ya tiene un perfil SFace.", text_color=RED)
            return
        if not self.camera_connected:
            self.enrollment_hint.configure(text="La cámara no está disponible.", text_color=RED)
            return
        self.enrollment_hint.configure(text="Preparando captura…", text_color=MUTED)
        self.capture_button.configure(state="disabled", text="Preparando…")
        self.register_entry.configure(state="disabled")
        self.register_pin.configure(state="disabled")
        self.register_pin_confirm.configure(state="disabled")
        self.backend.send("start_enroll", (name, access_pin))

    def cancel_enrollment(self) -> None:
        if self.modal_close_job is not None:
            self.after_cancel(self.modal_close_job)
            self.modal_close_job = None
        self.backend.send("cancel_enroll")
        if self.register_window is not None and self.register_window.winfo_exists():
            self.register_pin.configure(state="normal")
            self.register_pin_confirm.configure(state="normal")
            self.register_pin.delete(0, "end")
            self.register_pin_confirm.delete(0, "end")
            self.register_window.grab_release()
            self.forget_icons(self.register_window)
            self.register_window.destroy()
        self.register_window = None
        if self.running:
            self.sync_capture_activity()

    def close_registration_after_success(self) -> None:
        self.modal_close_job = None
        if self.register_window is not None and self.register_window.winfo_exists():
            self.register_pin.configure(state="normal")
            self.register_pin_confirm.configure(state="normal")
            self.register_pin.delete(0, "end")
            self.register_pin_confirm.delete(0, "end")
            self.register_window.grab_release()
            self.forget_icons(self.register_window)
            self.register_window.destroy()
        self.register_window = None
        if self.running:
            self.sync_capture_activity()

    def refresh_history(self) -> None:
        if self.current_page not in {"Inicio", "Escáner"}:
            return
        fingerprint = tuple(self.history_rows)
        if fingerprint == self._rendered_recent:
            return
        self._rendered_recent = fingerprint
        for child in self.history_list.winfo_children():
            child.destroy()
        rows = self.history_rows
        if not rows:
            ctk.CTkLabel(
                self.history_list, text="Aún no hay intentos registrados.",
                text_color=MUTED, font=ctk.CTkFont("Segoe UI", 10),
            ).pack(anchor="w", padx=8, pady=(2, 0))
            return
        for date, user, result, distance, similarity in rows:
            item = ctk.CTkFrame(self.history_list, fg_color=SURFACE_2, corner_radius=8)
            item.pack(fill="x", pady=(0, 6))
            item.grid_columnconfigure(0, weight=1)
            color = CYAN if result == "AUTORIZADO" else RED
            ctk.CTkLabel(item, text=user, text_color=TEXT,
                         font=ctk.CTkFont("Segoe UI", 11, "bold")).grid(
                row=0, column=0, sticky="w", padx=10, pady=(6, 0)
            )
            ctk.CTkLabel(item, text=result, text_color=color,
                         font=ctk.CTkFont("Segoe UI", 9, "bold")).grid(
                row=0, column=1, sticky="e", padx=10, pady=(6, 0)
            )
            metric = (f"  ·  SFace {similarity:.3f}" if similarity is not None
                      else f"  ·  LBPH {distance:.1f}" if distance is not None else "")
            ctk.CTkLabel(item, text=date[:10] + "  " + date[11:19] + metric,
                         text_color=MUTED, font=ctk.CTkFont("Consolas", 9)).grid(
                row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 6)
            )

    def poll_backend(self) -> None:
        if not self.running:
            return
        for _ in range(8):
            try:
                name, result, error = self.utility_results.get_nowait()
            except queue.Empty:
                break
            if error:
                self.data_diagnostics.configure(text=f"{name}: {error}", text_color=RED)
                self.record_incident("Datos", f"{name}: {error}")
            elif name == "integrity":
                self.data_diagnostics.configure(
                    text="Integridad: sin hallazgos" if not result else
                    "Integridad: " + "\n".join(result[:12]) +
                    (f"\n… y {len(result)-12} hallazgos más" if len(result) > 12 else ""),
                    text_color=CYAN if not result else AMBER)
            elif name == "storage":
                self.data_diagnostics.configure(
                    text="Espacio ocupado · " + "  ·  ".join(
                        f"{key}: {value / 1024 / 1024:.2f} MB" for key, value in result.items()),
                    text_color=TEXT)
            elif name == "verify_backup":
                path, (valid, findings) = result
                self.data_diagnostics.configure(
                    text=f"Respaldo {'válido' if valid else 'con problemas'}: {path}" +
                    (" · " + "; ".join(findings) if findings else " · ZIP legible y archivos esperados presentes"),
                    text_color=CYAN if valid else RED)
                if not valid:
                    self.record_incident("Respaldo", "; ".join(findings))
            elif name == "graphs":
                self.notify(f"Gráficos PNG guardados en {result}")
        for _ in range(5):
            try:
                success, label, button, error = self.admin_results.get_nowait()
            except queue.Empty:
                break
            if error:
                if label.winfo_exists():
                    label.configure(text=error, text_color=RED)
                if button.winfo_exists():
                    button.configure(state="normal")
            else:
                success()
        for _ in range(60):
            try:
                event = self.backend.events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "ready":
                self.backend_ready = True
                self.known_users = event[1]
                self.history_rows = event[2]
                self.people = event[3]
                self.update_user_count()
                self.refresh_history()
                self.refresh_users()
                self.settings_engine.configure(text="Listo · YuNet + SFace", text_color=CYAN)
                if self.current_page == "Historial":
                    self.request_history()
                if self.camera_connected:
                    self.set_state("ESCANEANDO", "Esperando un rostro frente a la cámara")
            elif kind == "camera":
                self.camera_connected = event[1]
                if self.camera_connected:
                    self.camera_badge.configure(
                        text="●  VISTA PREVIA · SIN RECONOCIMIENTO" if not self.backend_ready else "●  EN VIVO",
                        text_color=AMBER if not self.backend_ready else CYAN)
                    self.settings_camera.configure(text="Conectada", text_color=CYAN)
                    self.settings_retry.configure(state="disabled")
                    self.retry_button.grid_remove()
                    self.camera_canvas.delete("placeholder")
                    if self.backend_ready and self.status_state != "ERROR DE MODELO":
                        self.set_state("ESCANEANDO", "Esperando un rostro frente a la cámara")
                else:
                    if event[2] != "Cambiando cámara…":
                        self.record_incident("Cámara", event[2])
                    self.camera_badge.configure(text="●  SIN SEÑAL", text_color=AMBER)
                    self.settings_camera.configure(text=f"Desconectada · {event[2]}", text_color=AMBER)
                    self.settings_retry.configure(state="normal")
                    self.retry_button.grid()
                    self.camera_canvas.itemconfigure(self.image_item, image="")
                    self.camera_canvas.itemconfigure(self.cam_text_item, state="hidden")
                    self.camera_canvas.itemconfigure(self.scan_text_item, state="hidden")
                    self.camera_canvas.itemconfigure(self.scan_line_item, state="hidden")
                    self.hud_visible = False
                    self.scan_line_visible = False
                    self.photo = None
                    self.photo_size = None
                    self.boxes = []
                    self.face_labels = []
                    self.position_boxes()
                    self.faces_label.configure(text="ROSTROS DETECTADOS  00")
                    if self.status_state != "ERROR DE MODELO":
                        self.set_state("CÁMARA DESCONECTADA", event[2])
                    self.draw_placeholder()
            elif kind == "camera_paused":
                self.camera_paused = event[1]
                if self.camera_paused:
                    self.camera_badge.configure(text="●  EN PAUSA", text_color=MUTED)
                    self.settings_camera.configure(text="Pausada por pestaña", text_color=MUTED)
                    self.camera_canvas.itemconfigure(self.image_item, image="")
                    self.camera_canvas.itemconfigure(self.cam_text_item, state="hidden")
                    self.camera_canvas.itemconfigure(self.scan_text_item, state="hidden")
                    self.camera_canvas.itemconfigure(self.scan_line_item, state="hidden")
                    self.hud_visible = False
                    self.scan_line_visible = False
                    self.photo = None
                    self.photo_size = None
                    self.last_frame_seq = 0
                    self.boxes = []
                    self.face_labels = []
                    self.position_boxes()
                    self.draw_placeholder()
                else:
                    self.camera_badge.configure(text="●  REANUDANDO", text_color=BLUE)
                    self.settings_camera.configure(text="Reanudando cámara…", text_color=AMBER)
                    self.draw_placeholder()
            elif kind == "state":
                if self.camera_connected:
                    self.set_state(event[1], event[2], event[3], event[4])
                    if event[1] in {"PIN REQUERIDO", "AUTORIZADO"} and event[3] not in {"—", "Desconocido"}:
                        if event[3] not in self.session_users:
                            self.session_users.add(event[3])
                            self.update_session_summary()
                    labels = event[6] if len(event) > 6 else []
                    if event[5] != self.boxes or labels != self.face_labels:
                        self.boxes = event[5]
                        self.face_labels = labels
                        self.faces_label.configure(text=f"ROSTROS DETECTADOS  {len(self.boxes):02d}")
                        self.position_boxes()
            elif kind == "history":
                self.history_rows = event[1]
                self.refresh_history()
                if self.current_page == "Historial":
                    self.request_history()
            elif kind == "stats":
                total, allowed, denied, latest = event[1]
                self.chart_stats = (total, allowed, denied)
                self.render_history_chart()
                last = latest[:19].replace("T", " ") if latest else "—"
                self.stats_label.configure(text=f"Intentos {total}   ·   Autorizados {allowed}   ·   Denegados {denied}   ·   Último {last}")
            elif kind == "daily_activity":
                self.daily_rows = event[1]
                self.render_history_chart()
            elif kind == "hourly_activity":
                self.hourly_rows = event[1]
                self.render_heatmap()
            elif kind == "access_recorded":
                if event[2] in self.session_counts:
                    self.session_counts[event[2]] += 1
                    self.update_session_summary()
            elif kind == "lock":
                self.update_lock(event[1], event[2])
            elif kind == "device":
                mode, port = event[1], event[2]
                self.mode = mode
                self.device_connected = mode == "hardware"
                self.physical_unknown = event[3]
                if mode == "hardware":
                    self.led_connected = False
                badge = ("●  HARDWARE CONECTADO" if mode == "hardware" else
                         "●  LED ESP32 CONECTADO" if self.led_port and self.led_connected else
                         "●  LED ESP32 SIN CONEXIÓN" if self.led_port and self.led_error_seen else
                         "●  LED ESP32 CONECTANDO" if self.led_port else
                         "●  SIM · FÍSICO DESCONOCIDO" if self.physical_unknown else "●  SOLO SIMULACIÓN")
                self.mode_badge.configure(text=badge,
                                          text_color=RED if mode == "simulacion" and self.led_port and self.led_error_seen else
                                          AMBER if mode == "simulacion" and self.physical_unknown else CYAN)
                device_text = (f"Hardware · {port} · sin sensor de puerta" if mode == "hardware" else
                               self.led_error_message or f"Cerradura simulada · LEDs {self.led_port} "
                               f"{'conectados' if self.led_connected else 'conectando'}" if self.led_port else
                               "Simulación · estado físico desconocido" if self.physical_unknown else
                               "Simulación · sin hardware conectado")
                self.settings_device.configure(text=device_text,
                                               text_color=RED if mode == "simulacion" and self.led_port and self.led_error_seen else TEXT)
                if self.pending_settings is not None:
                    values, self.pending_settings = self.pending_settings, None
                    self.commit_settings(values)
            elif kind == "device_error":
                self.pending_settings = None
                self.device_connected = False
                self.physical_unknown = True
                self.mode_badge.configure(text="●  HARDWARE · ERROR", text_color=RED)
                self.settings_device.configure(text=event[1], text_color=RED)
                self.notify(event[1], error=True)
                self.record_incident("ESP32", event[1])
            elif kind == "led_device":
                if self.mode != "simulacion":
                    continue
                self.led_connected = event[1]
                self.led_error_message = None if event[1] else event[2]
                self.mode_badge.configure(text=("●  LED ESP32 CONECTADO" if event[1] else
                                                "●  LED ESP32 SIN CONEXIÓN"),
                                          text_color=CYAN if event[1] else RED)
                self.settings_device.configure(text=(f"Cerradura simulada · LEDs {self.led_port} conectados"
                                                     if event[1] else event[2]),
                                               text_color=TEXT if event[1] else RED)
                if event[1]:
                    self.led_error_seen = False
                elif not self.led_error_seen:
                    self.notify(event[2], error=True)
                    self.led_error_seen = True
                if not event[1]:
                    self.record_incident("ESP32", event[2])
            elif kind == "servo_status":
                self.servo_status = event[1].replace("_", " ")
                color = CYAN if self.servo_status == "LISTO" else RED if self.servo_status == "ERROR" else AMBER
                self.servo_value.configure(text=f"MG90S · {self.servo_status}", text_color=color)
                self.settings_servo.configure(text=(self.servo_status +
                    (" · PWM aceptado, posición física no verificada" if self.servo_status == "LISTO" else
                     " · modo DEMO; posición física no verificada" if self.servo_status == "DEMO" else
                     " · calibra el pestillo desacoplado antes de usarlo" if self.servo_status == "NO CONFIGURADO" else "")),
                    text_color=color)
            elif kind == "servo_config":
                self.servo_pulses = (event[1], event[2])
                for entry, pulse in ((self.servo_open_entry, event[1]),
                                     (self.servo_lock_entry, event[2])):
                    entry.delete(0, "end")
                    entry.insert(0, str(pulse))
                self.servo_save_button.configure(state="normal")
            elif kind == "servo_config_saved":
                self.notify("Recorrido MG90S guardado en la ESP32; no se ordenó movimiento")
            elif kind == "servo_config_error":
                self.servo_save_button.configure(state="normal")
                self.notify(event[1], error=True)
                self.record_incident("MG90S", event[1])
            elif kind == "servo_test_result":
                self.notify(event[2], error=not event[1])
            elif kind == "configuration_error":
                self.pending_settings = None
                self.notify(event[1], error=True)
                self.record_incident("Configuración", event[1])
            elif kind == "export_done":
                self.notify(f"{event[1]} intentos exportados a CSV")
            elif kind == "export_filtered_done":
                self.notify(f"{event[1]} intentos filtrados exportados a {event[2]}")
            elif kind == "export_filtered_error":
                self.notify(f"No se pudo exportar: {event[1]}", error=True)
                self.record_incident("Datos", event[1])
            elif kind == "backup_done":
                self.notify(f"Respaldo validado: {event[1]} · {event[2]} archivos. Contiene datos sensibles.")
                self.last_backup_path = Path(event[1])
                self.backup_paths.append(self.last_backup_path)
                backup_path = event[1]
                self.run_utility("verify_backup", lambda path=backup_path: (
                    path, verify_local_backup(Path(path))))
            elif kind == "backup_error":
                self.notify(f"No se creó el respaldo: {event[1]}", error=True)
                self.record_incident("Respaldo", event[1])
            elif kind == "history_search":
                if (event[1], event[2], event[3], event[4], event[5]) == (
                        self.history_search.get().strip(), self.history_filter.get(),
                        self.history_offset, *self.history_dates):
                    self.history_results = event[6]
                    if self.current_page == "Historial":
                        self.refresh_full_history()
            elif kind == "users":
                self.people = event[1]
                self.refresh_users()
            elif kind == "pins":
                self.access_pin_ids = set(event[1])
                self.refresh_users()
            elif kind == "pin_set":
                self.notify(f"PIN de acceso guardado para {event[1]}")
            elif kind == "pin_required":
                self.show_access_pin(event[1], event[2], event[3],
                                     event[4] if len(event) > 4 else None)
            elif kind == "pin_failed":
                if self.access_window is not None and self.access_window.winfo_exists():
                    self.access_entry.delete(0, "end")
                    self.access_error.configure(text=event[1], text_color=RED)
                    self.access_button.configure(
                        state="normal" if time.monotonic() < self.access_deadline else "disabled")
            elif kind == "pin_expired":
                self.close_access_pin(False)
                self.notify(event[1], error=True)
            elif kind == "pin_success":
                self.close_access_pin(False)
                self.notify(f"Acceso autorizado: {event[1]}")
            elif kind == "welcome":
                threading.Thread(target=play_welcome_sound, daemon=True,
                                 name="safebox-welcome").start()
            elif kind == "user_deleted":
                self.people = event[2]
                self.known_users = [name for _, name, _ in self.people]
                self.update_user_count()
                self.refresh_users()
                self.notify(f"Se eliminó a {event[1]} y se actualizó el registro.")
            elif kind == "user_error":
                self.notify(event[1], error=True)
            elif kind == "enroll_started":
                if self.register_window is not None and self.register_window.winfo_exists():
                    self.capture_button.configure(text="Capturando…")
            elif kind == "enroll_progress":
                if self.register_window is not None and self.register_window.winfo_exists():
                    self.enrollment_progress.set(event[1] / SAMPLES_REQUIRED)
                    label = ("Guardando perfil SFace…" if event[1] >= SAMPLES_REQUIRED else
                             f"{event[1]} / {SAMPLES_REQUIRED} muestras · {round(event[1] / SAMPLES_REQUIRED * 100)} %")
                    self.enrollment_hint.configure(text=label)
            elif kind == "enroll_hint":
                if self.register_window is not None and self.register_window.winfo_exists():
                    self.enrollment_hint.configure(text=event[1], text_color=AMBER)
            elif kind == "enroll_done":
                self.known_users = event[2]
                self.update_user_count()
                self.backend.send("list_users")
                self.notify(f"{event[1]} registrado correctamente." if event[3] else
                            f"Rostro de {event[1]} guardado; falta asignar PIN de acceso.", error=not event[3])
                if self.register_window is not None and self.register_window.winfo_exists():
                    self.enrollment_hint.configure(text=(f"{event[1]} registrado correctamente" if event[3] else
                                                         "Rostro guardado; asigna PIN en Usuarios"))
                    self.modal_close_job = self.after(650, self.close_registration_after_success)
            elif kind == "enroll_error":
                if self.register_window is not None and self.register_window.winfo_exists():
                    self.enrollment_hint.configure(text=event[1], text_color=RED)
                    self.capture_button.configure(state="normal", text="Reintentar")
                    self.register_entry.configure(state="normal")
                    self.register_pin.configure(state="normal")
                    self.register_pin_confirm.configure(state="normal")
            elif kind == "storage_error":
                self.detail_label.configure(text=f"No se guardó el intento: {event[1]}")
                self.notify(f"No se guardó el intento: {event[1]}", error=True)
                self.record_incident("Datos", event[1])
            elif kind == "engine_error":
                self.backend_ready = False
                self.device_connected = False
                self.physical_unknown = True
                self.set_state("ERROR DE MODELO", event[1])
                self.settings_engine.configure(text=event[1], text_color=RED)
                self.camera_badge.configure(text="●  VISTA PREVIA · SIN RECONOCIMIENTO" if self.camera_connected
                                            else "●  RECONOCIMIENTO DETENIDO", text_color=RED)
                self.close_access_pin(False)
                if not self.camera_connected:
                    self.draw_placeholder()
                if self.mode == "hardware":
                    self.mode_badge.configure(text="●  HARDWARE SIN CONEXIÓN", text_color=RED)
                self.record_incident("Modelos", event[1])
        self.refresh_diagnostics()
        self.refresh_assistant()
        if (self.camera_connected and self.current_page in {"Inicio", "Escáner"} and
                time.monotonic() - self.preview_last_at >= self.preview_interval):
            latest = self.backend.latest_frame(self.last_frame_seq)
            if latest is not None:
                self.last_frame_seq, frame = latest
                self.show_frame(frame)
                self.preview_last_at = time.monotonic()
        self.poll_job = self.after(33 if self.camera_connected and self.current_page in {"Inicio", "Escáner"}
                                   else 100, self.poll_backend)

    def show_frame(self, frame: np.ndarray) -> None:
        canvas = self.camera_canvas
        width, height = max(2, canvas.winfo_width()), max(2, canvas.winfo_height())
        scale = min(1.0, width / frame.shape[1], height / frame.shape[0])
        image_width = max(1, int(frame.shape[1] * scale))
        image_height = max(1, int(frame.shape[0] * scale))
        offset_x = (width - image_width) // 2
        offset_y = (height - image_height) // 2
        resized = (frame if (image_width, image_height) == (frame.shape[1], frame.shape[0]) else
                   cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        if self.photo is None or self.photo_size != (image_width, image_height):
            self.photo = ImageTk.PhotoImage(image)
            self.photo_size = (image_width, image_height)
            canvas.itemconfigure(self.image_item, image=self.photo)
        else:
            self.photo.paste(image)
        layout = (offset_x, offset_y, image_width, image_height, scale)
        if layout != self.image_layout:
            self.image_layout = layout
            canvas.coords(self.image_item, offset_x, offset_y)
            self.position_boxes()
        if not self.hud_visible:
            canvas.itemconfigure(self.cam_text_item, state="normal")
            canvas.itemconfigure(self.scan_text_item, state="normal")
            self.hud_visible = True
        self.render_tick += 1
        if self.render_tick % 3 == 0:
            active_color = STATE_COLORS.get(self.status_state, CYAN)
            pulse = 0.72 + 0.28 * abs(np.sin(time.monotonic() * 2.5))
            pulse_color = mix_color(BORDER, active_color, pulse)
            canvas.itemconfigure(self.scan_text_item, fill=pulse_color)
            for group in self.corner_items:
                for item in group:
                    canvas.itemconfigure(item, fill=pulse_color)
        scan_y = offset_y + int((time.monotonic() * 54) % max(1, image_height))
        scanning = self.status_state in {"ESCANEANDO", "VERIFICANDO", "REGISTRANDO"}
        if scanning:
            canvas.coords(self.scan_line_item, offset_x + 12, scan_y,
                          offset_x + image_width - 12, scan_y)
        if scanning != self.scan_line_visible:
            self.scan_line_visible = scanning
            canvas.itemconfigure(self.scan_line_item, state="normal" if scanning else "hidden")

    def position_boxes(self) -> None:
        canvas = self.camera_canvas
        offset_x, offset_y, _, _, scale = self.image_layout

        while len(self.label_items) < len(self.boxes):
            background = canvas.create_rectangle(0, 0, 0, 0, fill=SURFACE_2,
                                                 outline=BORDER, state="hidden")
            text = canvas.create_text(0, 0, anchor="nw", fill=TEXT,
                                      font=("Segoe UI", 10, "bold"), state="hidden")
            self.label_items.append((background, text))
        for index, group in enumerate(self.corner_items):
            if index >= len(self.boxes) or not self.camera_connected:
                for item in group:
                    canvas.itemconfigure(item, state="hidden")
                continue
            x, y, w, h = self.boxes[index]
            left = offset_x + int(x * scale)
            top = offset_y + int(y * scale)
            right = offset_x + int((x + w) * scale)
            bottom = offset_y + int((y + h) * scale)
            length = min(26, max(12, (right - left) // 4))
            points = []
            for corner_x, dx in ((left, 1), (right, -1)):
                for corner_y, dy in ((top, 1), (bottom, -1)):
                    points.append((corner_x, corner_y, corner_x + dx * length, corner_y))
                    points.append((corner_x, corner_y, corner_x, corner_y + dy * length))
            for item, coords in zip(group, points):
                canvas.coords(item, *coords)
                canvas.itemconfigure(item, state="normal")

        occupied = []
        canvas_width = max(1, canvas.winfo_width())
        for index, (background, text) in enumerate(self.label_items):
            if index >= len(self.boxes) or not self.camera_connected:
                canvas.itemconfigure(background, state="hidden")
                canvas.itemconfigure(text, state="hidden")
                continue
            x, y, _, _ = self.boxes[index]
            name, phase = (self.face_labels[index] if index < len(self.face_labels) else
                           ("Desconocido", "DESCONOCIDO"))
            caption = f"{name} · {phase}" if phase != "DESCONOCIDO" else name
            color = CYAN if phase == "AUTORIZADO" else AMBER if phase == "PIN PENDIENTE" else BLUE if phase in {"IDENTIDAD", "VERIFICANDO"} else MUTED
            canvas.itemconfigure(text, text=caption, fill=color, state="normal")
            label_width = min(max(8, canvas_width - 8), max(90, len(caption) * 7 + 14))
            left = max(4, min(canvas_width - label_width - 4, offset_x + int(x * scale)))
            top = max(26, offset_y + int(y * scale) - 25)
            while any(left < right and left + label_width > other_left and
                      top < bottom and top + 20 > other_top
                      for other_left, other_top, right, bottom in occupied):
                top += 23
            occupied.append((left, top, left + label_width, top + 20))
            canvas.coords(background, left, top, left + label_width, top + 20)
            canvas.coords(text, left + 7, top + 2)
            canvas.itemconfigure(background, state="normal")

    def close(self) -> None:
        if not self.running:
            return
        self.running = False
        self.admin_until = 0.0
        for job in (self.transition_job, self.led_off_job, self.backend_start_job, self.poll_job,
                    self.clock_job, self.modal_close_job, self.search_job, self.notice_clear_job,
                    self.nav_transition_job, self.intro_job, self.resize_job, self.icon_job):
            if job is not None:
                self.after_cancel(job)
        if getattr(self, "intro_overlay", None) is not None:
            self.intro_overlay.destroy()
            self.intro_overlay = None
        if self.register_window is not None and self.register_window.winfo_exists():
            self.register_pin.configure(state="normal")
            self.register_pin_confirm.configure(state="normal")
            self.register_pin.delete(0, "end")
            self.register_pin_confirm.delete(0, "end")
            self.register_window.grab_release()
            self.register_window.destroy()
            self.register_window = None
        self.close_access_pin(False)
        self.close_pin_window()
        self.withdraw()
        self.backend.stop()
        self.close_deadline = time.monotonic() + 10.0
        self.finish_close()

    def finish_close(self) -> None:
        if self.backend.stopped() or time.monotonic() >= self.close_deadline:
            self.destroy()
        else:
            self.after(40, self.finish_close)


def self_test() -> None:
    face_detector, recognizer = load_models()
    blank = np.zeros((320, 320, 3), dtype=np.uint8)
    if len(detect_faces(face_detector, blank)):
        raise RuntimeError("YuNet detectó un rostro en una imagen vacía")
    feature = recognizer.feature(np.zeros((112, 112, 3), dtype=np.uint8))
    if feature.size != 128 or not np.isfinite(feature).all():
        raise RuntimeError("SFace no produjo un vector válido")
    print(f"OK: OpenCV {cv2.__version__}, CustomTkinter {ctk.__version__}, YuNet + SFace CPU")


def main() -> None:
    parser = argparse.ArgumentParser(description="SAFEBOX AI · dashboard biométrico local")
    parser.add_argument("--self-test", action="store_true", help="verifica software sin cámara")
    parser.add_argument("--data-dir", type=Path, default=BASE / "datos")
    parser.add_argument("--mode", choices=("simulacion", "hardware"))
    parser.add_argument("--port", help="puerto de la WeMos (usa el guardado o COM4 por defecto)")
    parser.add_argument("--led-port", help="puerto LED en simulación; usa COMx u OFF")
    parser.add_argument("--similarity-threshold", type=float,
                        help="umbral coseno SFace entre 0.5 y 0.9 (predeterminado: 0.5)")
    parser.add_argument("--camera-index", type=int, help="índice de cámara")
    parser.add_argument("--authorization-seconds", type=int, help="tiempo para confirmar cierre")
    args = parser.parse_args()
    if args.self_test:
        try:
            self_test()
        except (FileNotFoundError, RuntimeError, cv2.error) as exc:
            raise SystemExit(f"Prueba fallida: {exc}") from exc
        return
    ctk.set_appearance_mode("dark")
    try:
        settings = SettingsStore(args.data_dir)
        values = dict(settings.values)
        values["mode"] = args.mode if args.mode is not None else "hardware"
        if values["mode"] == "hardware":
            values["port"] = args.port if args.port is not None else (values["port"] or "COM4")
        for key, supplied in (("mode", args.mode), ("port", args.port),
                              ("similarity_threshold", args.similarity_threshold),
                              ("camera_index", args.camera_index),
                              ("authorization_seconds", args.authorization_seconds)):
            if supplied is not None:
                values[key] = supplied
        settings.validate(values)
        led_port = (args.led_port.strip().upper() if args.led_port is not None else
                    "COM4" if values["mode"] == "simulacion" else None)
        if led_port == "OFF":
            led_port = None
        if led_port and values["mode"] != "simulacion":
            raise ValueError("--led-port requiere --mode simulacion para no enviar comandos al servo")
        app = SafeApp(args.data_dir, values["mode"], values["port"],
                      values["similarity_threshold"], values["camera_index"],
                      values["authorization_seconds"], settings,
                      led_port)
    except Exception as exc:
        raise SystemExit(f"No se pudo iniciar SAFEBOX AI: {exc}") from exc
    app.mainloop()


if __name__ == "__main__":
    main()
