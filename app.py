"""SAFEBOX AI: dashboard local de control de acceso biométrico."""

from __future__ import annotations

import argparse
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from tkinter import Canvas, filedialog

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from core import (DEFAULT_SIMILARITY_THRESHOLD, ENROLL_SAMPLES, SafeBackend,
                  detect_faces, load_models)
from control import AdminManager, SettingsStore


BASE = Path(__file__).resolve().parent
SAMPLES_REQUIRED = ENROLL_SAMPLES

BG = "#121212"
SURFACE = "#181818"
SURFACE_2 = "#252525"
BORDER = "#383838"
TEXT = "#FFFFFF"
MUTED = "#B3B3B3"
CYAN = "#1ED760"
BLUE = "#539DF5"
RED = "#F3727F"
AMBER = "#FFA42B"

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
        self.admin = AdminManager(data_dir)
        self.admin_results: queue.Queue = queue.Queue()
        self.admin_until = 0.0
        self.admin_window = None
        self.pending_settings = None
        self.led_port = led_port
        self.led_connected = False
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
        self.backend_ready = False
        self.known_users: list[str] = []
        self.people: list[tuple[str, str, bool]] = []
        self.access_pin_ids: set[str] = set()
        self.access_window = None
        self.access_request = None
        self.access_track = None
        self.history_rows: list[tuple] = []
        self.history_results: list[tuple] = []
        self.history_offset = 0
        self.current_page = "Inicio"
        self.boxes: list[tuple[int, int, int, int]] = []
        self.face_labels: list[tuple[str, str]] = []
        self.last_frame_seq = 0
        self.status_state = ""
        self.status_color = CYAN
        self.transition_job = None
        self.backend_start_job = None
        self.poll_job = None
        self.clock_job = None
        self.modal_close_job = None
        self.register_window: ctk.CTkToplevel | None = None
        self.photo = None
        self.photo_size = None
        self.preview_photo = None
        self.search_job = None

        self.build_ui()
        self.refresh_history()
        self.set_state("INICIANDO", "Preparando cámara y reconocimiento…")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.backend_start_job = self.after(20, self.backend.start)
        self.poll_job = self.after(40, self.poll_backend)
        self.clock_job = self.after(1000, self.update_clock)
        if not self.admin.configured:
            self.after(100, self.prompt_create_pin)

    @staticmethod
    def card(parent, radius: int = 16) -> ctk.CTkFrame:
        return ctk.CTkFrame(
            parent, fg_color=SURFACE, corner_radius=radius,
        )

    def build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        navigation = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0, width=210)
        navigation.grid(row=0, column=0, rowspan=3, sticky="nsew")
        navigation.grid_propagate(False)
        ctk.CTkLabel(navigation, text="◈  SAFEBOX AI", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 20, "bold")).pack(
            anchor="w", padx=20, pady=(29, 6))
        ctk.CTkLabel(navigation, text="CONTROL BIOMÉTRICO", text_color=MUTED,
                     font=ctk.CTkFont("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=22, pady=(0, 34))
        self.nav_buttons = {}
        for label, glyph in (("Inicio", "⌂"), ("Escáner", "◎"),
                             ("Usuarios", "♙"), ("Historial", "▤"),
                             ("Configuración", "⚙")):
            button = ctk.CTkButton(
                navigation, text=f"{glyph}   {label}", anchor="w", height=43,
                corner_radius=22, fg_color="transparent", hover_color=SURFACE_2,
                text_color=MUTED, font=ctk.CTkFont("Segoe UI", 13, "bold"),
                command=lambda page=label: self.navigate(page),
            )
            button.pack(fill="x", padx=13, pady=3)
            self.nav_buttons[label] = button
        ctk.CTkLabel(navigation, text="DATOS LOCALES · SAFEBOX", text_color=MUTED,
                     font=ctk.CTkFont("Segoe UI", 10)).pack(
            side="bottom", anchor="w", padx=22, pady=22)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=1, sticky="ew", padx=24, pady=(18, 16))
        header.grid_columnconfigure(0, weight=1)
        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="w")
        self.page_title = ctk.CTkLabel(brand, text="Panel de control", text_color=TEXT,
                                       font=ctk.CTkFont("Segoe UI", 27, "bold"))
        self.page_title.pack(side="left")
        ctk.CTkLabel(header, text="SAFEBOX AI  /  ACCESO BIOMÉTRICO", text_color=MUTED,
                     font=ctk.CTkFont("Segoe UI", 10, "bold")).grid(
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
            fg_color=SURFACE_2, corner_radius=18, width=164, height=34,
            font=ctk.CTkFont("Segoe UI", 10, "bold"))
        self.mode_badge.pack(side="left")

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

        side = ctk.CTkFrame(body, fg_color="transparent")
        side.grid(row=0, column=1, sticky="nsew")
        self.dashboard_side = side
        self.camera_card = camera_card
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(2, weight=1)

        identity = self.card(side)
        identity.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        identity.grid_columnconfigure(0, weight=1)
        self.state_chip = ctk.CTkLabel(
            identity, text="●  ESCANEANDO", text_color=CYAN,
            fg_color=SURFACE_2, corner_radius=9, width=130, height=30,
            font=ctk.CTkFont("Segoe UI", 10, "bold"),
        )
        self.state_chip.grid(row=0, column=0, sticky="w", padx=18, pady=(18, 12))
        self.identity_title = ctk.CTkLabel(
            identity, text="Esperando identidad", text_color=TEXT,
            font=ctk.CTkFont("Segoe UI", 23, "bold"), anchor="w",
        )
        self.identity_title.grid(row=1, column=0, sticky="ew", padx=18)
        self.detail_label = ctk.CTkLabel(
            identity, text="Sitúate frente a la cámara", text_color=MUTED,
            font=ctk.CTkFont("Segoe UI", 11), anchor="w",
        )
        self.detail_label.grid(row=2, column=0, sticky="ew", padx=18, pady=(2, 14))
        metric = ctk.CTkFrame(identity, fg_color=SURFACE_2, corner_radius=10)
        metric.grid(row=3, column=0, sticky="ew", padx=17, pady=(0, 16))
        metric.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(metric, text="SIMILITUD COSENO · SFACE", text_color=MUTED,
                     font=ctk.CTkFont("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(10, 0)
        )
        self.distance_label = ctk.CTkLabel(metric, text="—", text_color=TEXT,
                                            font=ctk.CTkFont("Consolas", 21, "bold"))
        self.distance_label.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 10))
        self.threshold_label = ctk.CTkLabel(metric, text=f"UMBRAL  ≥ {self.similarity_threshold:.3f}   ·   MAYOR ES MEJOR",
                     text_color=MUTED, font=ctk.CTkFont("Consolas", 9))
        self.threshold_label.grid(
            row=0, column=1, rowspan=2, sticky="e", padx=14
        )

        hardware_row = ctk.CTkFrame(side, fg_color="transparent")
        hardware_row.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        hardware_row.grid_columnconfigure((0, 1), weight=1)
        lock_card = self.card(hardware_row, 13)
        lock_card.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        led_card = self.card(hardware_row, 13)
        led_card.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.lock_value = self.device_label(lock_card, "CERRADURA VIRTUAL",
                                            "BLOQUEADA · SOLO SIM" if self.mode == "simulacion" else "SIN CONFIRMAR")
        self.servo_value = self.device_label(lock_card, "MG90S · GPIO 18", self.servo_status)
        self.led_value = self.device_label(led_card, "LED VIRTUAL", "APAGADO")
        self.close_button = ctk.CTkButton(
            lock_card, text="Confirmar cierre", height=29, corner_radius=16,
            fg_color=SURFACE_2, hover_color=BORDER, text_color=TEXT,
            font=ctk.CTkFont("Segoe UI", 10, "bold"),
            command=self.confirm_door_closed)
        self.close_button.pack(fill="x", padx=10, pady=(0, 10))
        self.close_button.pack_forget()

        activity = self.card(side)
        activity.grid(row=2, column=0, sticky="nsew")
        activity.grid_columnconfigure(0, weight=1)
        activity.grid_rowconfigure(1, weight=1)
        activity_head = ctk.CTkFrame(activity, fg_color="transparent")
        activity_head.grid(row=0, column=0, sticky="ew", padx=17, pady=(14, 7))
        activity_head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(activity_head, text="ACTIVIDAD RECIENTE", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(activity_head, text="LOCAL", text_color=MUTED,
                     font=ctk.CTkFont("Consolas", 9)).grid(row=0, column=1, sticky="e")
        self.history_list = ctk.CTkScrollableFrame(
            activity, fg_color="transparent", scrollbar_button_color=BORDER,
        )
        self.history_list.grid(row=1, column=0, sticky="nsew", padx=11, pady=(0, 8))

        self.register_button = ctk.CTkButton(
            side, text="+  Registrar nuevo usuario", height=43,
            fg_color=CYAN, hover_color="#1DB954", text_color=BG,
            corner_radius=22,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
            command=self.open_registration,
        )
        self.register_button.grid(row=3, column=0, sticky="ew", pady=(12, 0))

        footer = ctk.CTkFrame(self, fg_color="transparent")
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
                                         font=ctk.CTkFont("Segoe UI", 10))
        self.notice_label.grid(row=1, column=0, columnspan=2, sticky="w")
        self.build_secondary_pages()
        self.navigate("Inicio")

    def build_secondary_pages(self) -> None:
        self.pages = {}
        for title, heading, subtitle in (
            ("Usuarios", "Identidades autorizadas", "Gestiona los rostros registrados en este equipo"),
            ("Historial", "Registro de accesos", "Consulta los intentos guardados localmente"),
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
        users_card.grid_rowconfigure(1, weight=1)
        users_head = ctk.CTkFrame(users_card, fg_color="transparent")
        users_head.grid(row=0, column=0, sticky="ew", padx=22, pady=(20, 8))
        users_head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(users_head, text="Personas registradas", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 17, "bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(users_head, text="+  Registrar rostro", width=170, height=38,
                      corner_radius=20, fg_color=CYAN, hover_color="#1DB954",
                      text_color=BG, command=self.open_registration).grid(row=0, column=1)
        self.users_list = ctk.CTkScrollableFrame(users_card, fg_color="transparent")
        self.users_list.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 16))
        self.refresh_users()

        history = self.pages["Historial"]
        filters = self.card(history)
        filters.grid(row=2, column=0, sticky="nsew")
        filters.grid_columnconfigure(0, weight=1)
        filters.grid_rowconfigure(3, weight=1)
        self.stats_label = ctk.CTkLabel(
            filters, text="Cargando estadísticas…", text_color=TEXT,
            font=ctk.CTkFont("Segoe UI", 12, "bold"))
        self.stats_label.grid(row=0, column=0, sticky="w", padx=24, pady=(15, 0))
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
        ctk.CTkButton(search_row, text="Exportar CSV", width=115, height=40,
                      corner_radius=20, fg_color=BORDER, hover_color="#505050",
                      text_color=TEXT, command=self.export_history).grid(row=0, column=2, padx=(10, 0))
        self.history_note = ctk.CTkLabel(
            filters, text="Cargando historial…", text_color=MUTED,
            font=ctk.CTkFont("Segoe UI", 11))
        self.history_note.grid(row=2, column=0, sticky="w", padx=24, pady=(0, 8))
        self.history_full_list = ctk.CTkScrollableFrame(filters, fg_color="transparent")
        self.history_full_list.grid(row=3, column=0, sticky="nsew", padx=17, pady=(0, 16))
        pager = ctk.CTkFrame(filters, fg_color="transparent")
        pager.grid(row=4, column=0, sticky="e", padx=20, pady=(0, 16))
        self.history_prev = ctk.CTkButton(pager, text="Anterior", width=95, height=30,
                                           corner_radius=16, fg_color=SURFACE_2,
                                           hover_color=BORDER, command=lambda: self.change_history_page(-1))
        self.history_prev.pack(side="left", padx=(0, 8))
        self.history_next = ctk.CTkButton(pager, text="Siguiente", width=95, height=30,
                                           corner_radius=16, fg_color=SURFACE_2,
                                           hover_color=BORDER, command=lambda: self.change_history_page(1))
        self.history_next.pack(side="left")

        settings = self.pages["Configuración"]
        settings_card = ctk.CTkScrollableFrame(settings, fg_color=SURFACE, corner_radius=16)
        settings_card.grid(row=2, column=0, sticky="nsew")
        settings_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(settings_card, text="Conexiones y datos", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 18, "bold")).grid(
            row=0, column=0, sticky="w", padx=24, pady=(16, 8))
        self.settings_camera = self.setting_line(settings_card, 1, "Cámara", "Iniciando…")
        self.settings_engine = self.setting_line(settings_card, 2, "Reconocimiento", "Iniciando…")
        self.setting_line(settings_card, 3, "Modelo facial", "YuNet + SFace · CPU")
        mode_text = (f"Cerradura simulada · LEDs {self.led_port} conectando" if self.led_port else
                     "Simulación · sin hardware conectado" if self.mode == "simulacion" else
                     f"Hardware · esperando conexión ({self.backend.port})")
        self.settings_device = self.setting_line(settings_card, 4, "Modo", mode_text)
        self.settings_servo = self.setting_line(settings_card, 14, "MG90S", self.servo_status)
        self.setting_line(settings_card, 5, "Datos", str(self.backend.data_dir))
        self.threshold_entry = self.setting_input(settings_card, 6, "Umbral SFace", f"{self.similarity_threshold:.3f}")
        self.duration_entry = self.setting_input(settings_card, 7, "Autorización (segundos)", str(self.authorization_seconds))
        self.camera_choice = ctk.CTkOptionMenu(settings_card, values=[f"Cámara {i}" for i in range(10)],
                                               fg_color=SURFACE_2, button_color=BORDER)
        self.camera_choice.set(f"Cámara {self.camera_index}")
        self.camera_choice.grid(row=8, column=0, sticky="ew", padx=24, pady=4)
        self.mode_choice = ctk.CTkOptionMenu(settings_card, values=["Simulación", "Hardware"],
                                             fg_color=SURFACE_2, button_color=BORDER)
        self.mode_choice.set("Simulación" if self.mode == "simulacion" else "Hardware")
        self.mode_choice.grid(row=9, column=0, sticky="ew", padx=24, pady=4)
        self.port_entry = self.setting_input(settings_card, 10, "Puerto serial (COMx)",
                                             self.backend.port or self.led_port or "")
        actions = ctk.CTkFrame(settings_card, fg_color="transparent")
        actions.grid(row=11, column=0, sticky="w", padx=24, pady=12)
        ctk.CTkButton(actions, text="Guardar ajustes y comprobar conexión", width=265,
                      height=38, corner_radius=20, fg_color=CYAN, hover_color="#1DB954",
                      text_color=BG, command=self.save_settings).pack(side="left", padx=(0, 8))
        ctk.CTkButton(actions, text="Cambiar PIN admin", width=145, height=38, corner_radius=20,
                      fg_color=SURFACE_2, hover_color=BORDER, command=self.change_pin).pack(side="left", padx=4)
        ctk.CTkButton(actions, text="Cerrar sesión", width=120, height=38, corner_radius=20,
                      fg_color=SURFACE_2, hover_color=BORDER, command=self.lock_admin_session).pack(side="left", padx=4)
        self.settings_retry = ctk.CTkButton(
            settings_card, text="Reintentar cámara", width=170,
            height=38, corner_radius=20, fg_color=SURFACE_2,
            hover_color=BORDER, text_color=TEXT, command=self.open_camera)
        self.settings_retry.grid(row=12, column=0, sticky="w", padx=24, pady=(4, 8))
        ctk.CTkLabel(settings_card,
                     text="El modo hardware requiere ESP32 y ACK serial. Confirma el cierre antes de volver a bloquear.",
                     text_color=MUTED, font=ctk.CTkFont("Segoe UI", 11),
                     wraplength=650, justify="left").grid(
            row=13, column=0, sticky="w", padx=24, pady=(0, 14))
        servo_actions = ctk.CTkFrame(settings_card, fg_color="transparent")
        servo_actions.grid(row=15, column=0, sticky="w", padx=24, pady=(0, 12))
        ctk.CTkButton(servo_actions, text="Probar MG90S", width=150, height=36,
                      corner_radius=18, fg_color=SURFACE_2, hover_color=BORDER,
                      command=lambda: self.require_admin("Calibrar MG90S", self.test_servo)).pack(side="left", padx=(0, 8))
        ctk.CTkButton(servo_actions, text="Detener PWM", width=140, height=36,
                      corner_radius=18, fg_color=SURFACE_2, hover_color=BORDER,
                      command=self.stop_servo_signal).pack(side="left")
        self.servo_open_entry = self.setting_input(settings_card, 16, "Autorización · µs", "—")
        self.servo_lock_entry = self.setting_input(settings_card, 17, "Retorno · µs", "—")
        self.servo_save_button = ctk.CTkButton(
            settings_card, text="Guardar recorrido MG90S", width=210, height=36,
            corner_radius=18, fg_color=SURFACE_2, hover_color=BORDER,
            command=lambda: self.require_admin("Ajustar MG90S", self.save_servo_config))
        self.servo_save_button.grid(row=18, column=0, sticky="w", padx=24, pady=(6, 4))
        ctk.CTkLabel(settings_card,
                     text="Solo DEMO sin puerta · 1000–2000 µs · guardar no mueve el servo."
                          " Mide otras posiciones antes de instalar el pestillo real.",
                     text_color=MUTED, font=ctk.CTkFont("Segoe UI", 11),
                     wraplength=650, justify="left").grid(
            row=19, column=0, sticky="w", padx=24, pady=(0, 14))

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
        self.current_page = page
        self.page_title.configure(text="Panel de control" if page == "Inicio" else page)
        for name, button in self.nav_buttons.items():
            active = name == page
            button.configure(fg_color=SURFACE_2 if active else "transparent",
                             text_color=TEXT if active else MUTED)
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
            elif page == "Historial":
                self.request_history()

    def refresh_users(self) -> None:
        for child in self.users_list.winfo_children():
            child.destroy()
        if not self.people:
            ctk.CTkLabel(self.users_list,
                         text="Aún no hay identidades. Registra el primer rostro para empezar.",
                         text_color=MUTED, font=ctk.CTkFont("Segoe UI", 13)).pack(
                anchor="w", padx=12, pady=24)
            return
        for person_id, name, active in self.people:
            item = ctk.CTkFrame(self.users_list, fg_color=SURFACE_2, corner_radius=12)
            item.pack(fill="x", pady=5)
            item.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(item, text=f"●  {name}", text_color=TEXT if active else AMBER,
                         font=ctk.CTkFont("Segoe UI", 13, "bold")).grid(
                row=0, column=0, sticky="w", padx=18, pady=(9, 0))
            pin_text = "PIN de acceso activo" if person_id in self.access_pin_ids else "PIN de acceso pendiente"
            ctk.CTkLabel(item,
                         text=(f"SFace activo · {pin_text}" if active else
                               f"Registro LBPH anterior · requiere nuevo enrolamiento · {pin_text}"),
                         text_color=MUTED, font=ctk.CTkFont("Segoe UI", 10)).grid(
                row=1, column=0, sticky="w", padx=18, pady=(0, 9))
            if not active:
                ctk.CTkButton(item, text="Reenrolar", width=95, height=32,
                              corner_radius=17, fg_color=CYAN, hover_color="#1DB954",
                              text_color=BG,
                              command=lambda label=name: self.open_registration(label)).grid(
                    row=0, column=1, rowspan=2, padx=(4, 0))
            ctk.CTkButton(item, text="PIN acceso", width=95, height=32,
                          corner_radius=17, fg_color=SURFACE, hover_color=BORDER,
                          text_color=TEXT, command=lambda uid=person_id, label=name:
                          self.require_admin("PIN de acceso", lambda: self.set_access_pin(uid, label))).grid(
                    row=0, column=2, rowspan=2, padx=(4, 0))
            ctk.CTkButton(item, text="Eliminar", width=95, height=32,
                          corner_radius=17, fg_color="#423035", hover_color="#653B44",
                          text_color=RED, command=lambda uid=person_id, label=name:
                          self.confirm_delete(uid, label)).grid(row=0, column=3, rowspan=2, padx=14)

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
        self.notice_label.configure(text=message, text_color=RED if error else CYAN)
        self.after(5000, lambda: self.notice_label.configure(text="") if self.running else None)

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

    def set_access_pin(self, person_id: str, name: str) -> None:
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
            self.backend.send("set_access_pin", (person_id, pin))
            self.close_pin_window()
        ctk.CTkButton(window, text="Guardar PIN", height=40, corner_radius=20,
                      fg_color=CYAN, hover_color="#1DB954", text_color=BG,
                      command=save).grid(row=4, column=0, sticky="ew", padx=26, pady=8)
        first.focus_set()

    def show_access_pin(self, person_id: str, name: str, track_id: int) -> None:
        if (self.admin_window is not None and self.admin_window.winfo_exists()) or (
                self.register_window is not None and self.register_window.winfo_exists()):
            self.backend.send("cancel_access")
            return
        self.close_access_pin(False)
        window = ctk.CTkToplevel(self, fg_color=BG)
        self.access_window = window
        self.access_request = person_id
        self.access_track = track_id
        window.title("SAFEBOX AI · Segundo factor")
        window.geometry("420x245")
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
        self.access_button = ctk.CTkButton(window, text="Verificar y abrir", height=40,
                                           corner_radius=20, fg_color=CYAN,
                                           hover_color="#1DB954", text_color=BG,
                                           command=self.submit_access_pin)
        self.access_button.grid(row=4, column=0, sticky="ew", padx=26, pady=(6, 10))
        self.access_entry.bind("<Return>", lambda _event: self.submit_access_pin())
        self.access_entry.focus_set()

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
        if cancel and self.access_request is not None:
            self.backend.send("cancel_access")
        if self.access_window is not None and self.access_window.winfo_exists():
            self.access_entry.delete(0, "end")
            self.access_window.grab_release()
            self.access_window.destroy()
        self.access_window = None
        self.access_request = None
        self.access_track = None

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
            self.history_note.configure(text="Buscando…")
            self.backend.send("search_history", (
                self.history_search.get().strip(), self.history_filter.get(), self.history_offset))

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
                         text="No hay intentos que coincidan con la búsqueda.",
                         text_color=MUTED, font=ctk.CTkFont("Segoe UI", 13)).pack(
                anchor="w", padx=10, pady=22)
            return
        for date, user, result, distance, similarity, capture in rows:
            item = ctk.CTkFrame(self.history_full_list, fg_color=SURFACE_2, corner_radius=12)
            item.pack(fill="x", pady=4)
            item.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(item, text=user, text_color=TEXT,
                         font=ctk.CTkFont("Segoe UI", 12, "bold")).grid(
                row=0, column=0, sticky="w", padx=16, pady=(9, 0))
            color = CYAN if result == "AUTORIZADO" else RED
            ctk.CTkLabel(item, text=result, text_color=color,
                         font=ctk.CTkFont("Segoe UI", 11, "bold")).grid(
                row=0, column=1, sticky="e", padx=12, pady=(9, 0))
            metric = (f" · SFace coseno {similarity:.3f}" if similarity is not None
                      else f" · LBPH distancia {distance:.1f}" if distance is not None else "")
            ctk.CTkLabel(item, text=date[:19].replace("T", "  ") + metric,
                         text_color=MUTED, font=ctk.CTkFont("Segoe UI", 10)).grid(
                row=1, column=0, sticky="w", padx=16, pady=(0, 9))
            if capture:
                ctk.CTkButton(item, text="Ver captura", width=100, height=27,
                              corner_radius=14, fg_color=BORDER, hover_color="#555555",
                              text_color=TEXT,
                              command=lambda filename=capture: self.show_capture(filename)).grid(
                    row=1, column=1, padx=12, pady=(0, 8))

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
        self.camera_badge.configure(text="●  CONECTANDO", text_color=BLUE)
        self.retry_button.grid_remove()
        self.backend.retry_camera()

    def on_camera_resize(self, _event=None) -> None:
        canvas = self.camera_canvas
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        canvas.coords(self.border_item, 1, 1, width - 2, height - 2)
        canvas.coords(self.scan_text_item, width - 18, 19)
        if not self.camera_connected:
            self.draw_placeholder()
        else:
            self.position_boxes()

    def draw_placeholder(self) -> None:
        canvas = self.camera_canvas
        canvas.delete("placeholder")
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        cx, cy = width // 2, height // 2
        canvas.create_oval(cx - 34, cy - 68, cx + 34, cy,
                           outline=BORDER, width=2, tags="placeholder")
        canvas.create_line(cx - 55, cy + 16, cx + 55, cy + 16,
                           fill=BORDER, width=2, tags="placeholder")
        loading = self.status_state == "INICIANDO"
        canvas.create_text(cx, cy + 55,
                           text="INICIANDO CÁMARA" if loading else
                           "MODELO NO DISPONIBLE" if self.status_state == "ERROR DE MODELO" else
                           "CÁMARA SIN SEÑAL", fill=TEXT,
                           font=("Segoe UI", 17, "bold"), tags="placeholder")
        canvas.create_text(cx, cy + 82,
                           text="Preparando captura…" if loading else
                           "Ejecuta download_models.py y reinicia" if self.status_state == "ERROR DE MODELO" else
                           "Usa «Reintentar cámara» al conectar el dispositivo",
                           fill=MUTED, font=("Segoe UI", 10), tags="placeholder")

    def animate_state_color(self, start: str, target: str, step: int = 0) -> None:
        if not self.running:
            return
        color = mix_color(start, target, (step + 1) / 8)
        self.state_chip.configure(text_color=color)
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
        if state != self.status_state:
            if self.transition_job is not None:
                self.after_cancel(self.transition_job)
                self.transition_job = None
            old = self.status_color
            self.status_color = color
            self.animate_state_color(old, color)
        self.status_state = state
        self.state_chip.configure(text=f"●  {state}")
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
        led = "AZUL" if self.lock_state in {"DESBLOQUEADA", "ESPERANDO CIERRE"} else "ROJO" if state == "DENEGADO" else "APAGADO"
        self.led_value.configure(text=led, text_color=BLUE if led == "AZUL" else RED if led == "ROJO" else MUTED)

    def update_lock(self, state: str, reason: str) -> None:
        self.lock_state = state
        if self.mode == "simulacion":
            label = (f"SIM {state}\nFÍSICO DESCONOCIDO" if self.physical_unknown else
                     f"{state} · SOLO SIM")
        else:
            label = ("BLOQUEO ORDENADO\nPOSICIÓN NO VERIFICADA" if state == "BLOQUEADA" else
                     "APERTURA ORDENADA\nPOSICIÓN NO VERIFICADA" if state == "DESBLOQUEADA" else
                     "CIERRE PENDIENTE\nPOSICIÓN NO VERIFICADA" if state == "ESPERANDO CIERRE" else state)
        self.lock_value.configure(text=label, text_color=RED if state == "ERROR" else
                                  CYAN if state == "DESBLOQUEADA" else AMBER if state in {"ESPERANDO CIERRE", "AUTORIZACIÓN PENDIENTE"} else TEXT)
        if state in {"DESBLOQUEADA", "ESPERANDO CIERRE"}:
            self.close_button.pack(fill="x", padx=10, pady=(0, 10))
        else:
            self.close_button.pack_forget()
        if reason and state in {"ESPERANDO CIERRE", "ERROR"}:
            self.notify(reason, error=state == "ERROR")

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
            self.notify("Conecta COM4 en modo hardware DEMO antes de ajustar el MG90S", error=True)
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
        window = ctk.CTkToplevel(self, fg_color=BG)
        self.register_window = window
        window.title("SAFEBOX AI · Nuevo usuario")
        window.geometry("440x420")
        window.resizable(False, False)
        window.transient(self)
        window.grab_set()
        window.protocol("WM_DELETE_WINDOW", self.cancel_enrollment)
        window.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(window, text="Registrar identidad", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 23, "bold")).grid(
            row=0, column=0, sticky="w", padx=28, pady=(25, 5)
        )
        ctk.CTkLabel(window, text="Mantén una sola persona frente a la cámara durante la captura.",
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
            window, height=7, progress_color=CYAN, fg_color=SURFACE_2,
        )
        self.enrollment_progress.grid(row=5, column=0, sticky="ew", padx=28)
        self.enrollment_progress.set(0)
        self.enrollment_hint = ctk.CTkLabel(
            window, text="0 / 20 muestras", text_color=MUTED,
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
        self.backend.send("cancel_enroll")
        if self.register_window is not None and self.register_window.winfo_exists():
            self.register_pin.configure(state="normal")
            self.register_pin_confirm.configure(state="normal")
            self.register_pin.delete(0, "end")
            self.register_pin_confirm.delete(0, "end")
            self.register_window.grab_release()
            self.register_window.destroy()
        self.register_window = None

    def close_registration_after_success(self) -> None:
        self.modal_close_job = None
        if self.register_window is not None and self.register_window.winfo_exists():
            self.register_pin.configure(state="normal")
            self.register_pin_confirm.configure(state="normal")
            self.register_pin.delete(0, "end")
            self.register_pin_confirm.delete(0, "end")
            self.register_window.grab_release()
            self.register_window.destroy()
        self.register_window = None

    def refresh_history(self) -> None:
        for child in self.history_list.winfo_children():
            child.destroy()
        rows = self.history_rows
        if not rows:
            ctk.CTkLabel(
                self.history_list, text="Aún no hay intentos registrados.",
                text_color=MUTED, font=ctk.CTkFont("Segoe UI", 11),
            ).pack(anchor="w", padx=8, pady=12)
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
                if self.status_state == "ERROR DE MODELO":
                    continue
                self.camera_connected = event[1]
                if self.camera_connected:
                    self.camera_badge.configure(text="●  EN VIVO", text_color=CYAN)
                    self.settings_camera.configure(text="Conectada", text_color=CYAN)
                    self.retry_button.grid_remove()
                    self.camera_canvas.delete("placeholder")
                    if self.backend_ready:
                        self.set_state("ESCANEANDO", "Esperando un rostro frente a la cámara")
                else:
                    self.camera_badge.configure(text="●  SIN SEÑAL", text_color=AMBER)
                    self.settings_camera.configure(text="Desconectada", text_color=AMBER)
                    self.retry_button.grid()
                    self.camera_canvas.itemconfigure(self.image_item, image="")
                    self.photo = None
                    self.boxes = []
                    self.face_labels = []
                    self.position_boxes()
                    self.faces_label.configure(text="ROSTROS DETECTADOS  00")
                    self.set_state("CÁMARA DESCONECTADA", event[2])
                    self.draw_placeholder()
            elif kind == "state":
                if self.camera_connected:
                    self.set_state(event[1], event[2], event[3], event[4])
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
                last = latest[:19].replace("T", " ") if latest else "—"
                self.stats_label.configure(text=f"Intentos {total}   ·   Autorizados {allowed}   ·   Denegados {denied}   ·   Último {last}")
            elif kind == "lock":
                self.update_lock(event[1], event[2])
            elif kind == "device":
                mode, port = event[1], event[2]
                self.mode = mode
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
                self.physical_unknown = True
                self.mode_badge.configure(text="●  HARDWARE · ERROR", text_color=RED)
                self.settings_device.configure(text=event[1], text_color=RED)
                self.notify(event[1], error=True)
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
            elif kind == "servo_status":
                self.servo_status = event[1].replace("_", " ")
                color = CYAN if self.servo_status == "LISTO" else RED if self.servo_status == "ERROR" else AMBER
                self.servo_value.configure(text=self.servo_status, text_color=color)
                self.settings_servo.configure(text=(self.servo_status +
                    (" · PWM aceptado, posición física no verificada" if self.servo_status == "LISTO" else
                     " · demostración sin puerta; pestillo no calibrado" if self.servo_status == "DEMO" else
                     " · configura calibration.h tras probar sin puerta" if self.servo_status == "NO CONFIGURADO" else "")),
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
            elif kind == "servo_test_result":
                self.notify(event[2], error=not event[1])
            elif kind == "configuration_error":
                self.pending_settings = None
                self.notify(event[1], error=True)
            elif kind == "export_done":
                self.notify(f"{event[1]} intentos exportados a CSV")
            elif kind == "history_search":
                if (event[1], event[2], event[3]) == (self.history_search.get().strip(),
                                                     self.history_filter.get(), self.history_offset):
                    self.history_results = event[4]
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
                self.show_access_pin(event[1], event[2], event[3])
            elif kind == "pin_failed":
                if self.access_window is not None and self.access_window.winfo_exists():
                    self.access_entry.delete(0, "end")
                    self.access_error.configure(text=event[1], text_color=RED)
                    self.access_button.configure(state="normal")
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
                    label = "Guardando perfil SFace…" if event[1] >= SAMPLES_REQUIRED else f"{event[1]} / {SAMPLES_REQUIRED} muestras"
                    self.enrollment_hint.configure(text=label)
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
            elif kind == "engine_error":
                self.backend_ready = False
                self.set_state("ERROR DE MODELO", event[1])
                self.settings_engine.configure(text=event[1], text_color=RED)
                self.camera_badge.configure(text="●  RECONOCIMIENTO DETENIDO", text_color=RED)
                self.retry_button.configure(state="disabled")
                self.settings_retry.configure(state="disabled")
                self.camera_connected = False
                self.camera_canvas.itemconfigure(self.image_item, image="")
                self.photo = None
                self.boxes = []
                self.face_labels = []
                self.position_boxes()
                self.draw_placeholder()
                if self.mode == "hardware":
                    self.mode_badge.configure(text="●  HARDWARE SIN CONEXIÓN", text_color=RED)
                self.backend.stop()
        if self.camera_connected and self.current_page in {"Inicio", "Escáner"}:
            latest = self.backend.latest_frame(self.last_frame_seq)
            if latest is not None:
                self.last_frame_seq, frame = latest
                self.show_frame(frame)
        self.poll_job = self.after(12 if self.camera_connected else 120, self.poll_backend)

    def show_frame(self, frame: np.ndarray) -> None:
        canvas = self.camera_canvas
        width, height = max(2, canvas.winfo_width()), max(2, canvas.winfo_height())
        scale = min(1.0, width / frame.shape[1], height / frame.shape[0])
        image_width = max(1, int(frame.shape[1] * scale))
        image_height = max(1, int(frame.shape[0] * scale))
        offset_x = (width - image_width) // 2
        offset_y = (height - image_height) // 2
        resized = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
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
        canvas.itemconfigure(self.cam_text_item, state="normal")
        canvas.itemconfigure(self.scan_text_item, state="normal")
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
        if self.status_state in {"ESCANEANDO", "VERIFICANDO", "REGISTRANDO"}:
            canvas.coords(self.scan_line_item, offset_x + 12, scan_y,
                          offset_x + image_width - 12, scan_y)
            canvas.itemconfigure(self.scan_line_item, state="normal")
        else:
            canvas.itemconfigure(self.scan_line_item, state="hidden")

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
        for job in (self.transition_job, self.backend_start_job, self.poll_job,
                    self.clock_job, self.modal_close_job, self.search_job):
            if job is not None:
                self.after_cancel(job)
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
    parser.add_argument("--port", help="puerto de la WeMos (predeterminado: COM4 en hardware)")
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
            values["port"] = args.port if args.port is not None else "COM4"
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
