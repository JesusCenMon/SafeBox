"""SAFEBOX AI: dashboard local de control de acceso biométrico."""

from __future__ import annotations

import argparse
import queue
import re
import time
from datetime import datetime
from pathlib import Path
from tkinter import Canvas, messagebox

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from core import MAX_DISTANCE, SafeBackend, detector


BASE = Path(__file__).resolve().parent
SAMPLES_REQUIRED = 20

BG = "#070D18"
SURFACE = "#0E192B"
SURFACE_2 = "#14243A"
BORDER = "#203651"
TEXT = "#F0F7FF"
MUTED = "#A6BBD1"
CYAN = "#52E4EC"
BLUE = "#5B9CFF"
RED = "#FF667F"
AMBER = "#FFC985"

STATE_COLORS = {
    "INICIANDO": BLUE,
    "ESCANEANDO": CYAN,
    "VERIFICANDO": BLUE,
    "REGISTRANDO": BLUE,
    "AUTORIZADO": CYAN,
    "DENEGADO": RED,
    "CÁMARA DESCONECTADA": AMBER,
}


def mix_color(start: str, end: str, amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    a = [int(start[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(end[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * amount):02X}" for x, y in zip(a, b))


class SafeApp(ctk.CTk):
    def __init__(self, data_dir: Path, mode: str, port: str | None):
        super().__init__(fg_color=BG)
        self.title("SAFEBOX AI · Dashboard biométrico")
        self.geometry("1220x720+40+30")
        self.minsize(1060, 670)

        self.backend = SafeBackend(data_dir, mode, port)
        self.mode = mode
        self.running = True
        self.camera_connected = False
        self.backend_ready = False
        self.known_users: list[str] = []
        self.history_rows: list[tuple] = []
        self.boxes: list[tuple[int, int, int, int]] = []
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

        self.build_ui()
        self.refresh_history()
        self.set_state("INICIANDO", "Preparando cámara y reconocimiento…")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.backend_start_job = self.after(20, self.backend.start)
        self.poll_job = self.after(40, self.poll_backend)
        self.clock_job = self.after(1000, self.update_clock)

    @staticmethod
    def card(parent, radius: int = 16) -> ctk.CTkFrame:
        return ctk.CTkFrame(
            parent, fg_color=SURFACE, corner_radius=radius,
            border_width=1, border_color=BORDER,
        )

    def build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(18, 16))
        header.grid_columnconfigure(0, weight=1)
        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(brand, text="◈", text_color=CYAN,
                     font=ctk.CTkFont("Segoe UI Symbol", 31)).pack(side="left", padx=(0, 11))
        ctk.CTkLabel(brand, text="SAFEBOX", text_color=TEXT,
                     font=ctk.CTkFont("Segoe UI", 28, "bold")).pack(side="left")
        ctk.CTkLabel(brand, text=" AI", text_color=CYAN,
                     font=ctk.CTkFont("Segoe UI", 28, "bold")).pack(side="left")
        ctk.CTkLabel(header, text="CENTRO DE CONTROL BIOMÉTRICO", text_color=MUTED,
                     font=ctk.CTkFont("Segoe UI", 10, "bold")).grid(
            row=1, column=0, sticky="w", padx=46, pady=(0, 0)
        )
        header_right = ctk.CTkFrame(header, fg_color="transparent")
        header_right.grid(row=0, column=1, rowspan=2, sticky="e")
        self.clock_label = ctk.CTkLabel(header_right, text="", text_color=MUTED,
                                        font=ctk.CTkFont("Consolas", 12))
        self.clock_label.pack(side="left", padx=(0, 18))
        self.clock_label.configure(text=datetime.now().strftime("%H:%M:%S"))
        mode_label = "SIMULACIÓN ACTIVA" if self.mode == "simulacion" else "SERIAL ACTIVO"
        ctk.CTkLabel(header_right, text=f"●  {mode_label}", text_color=CYAN,
                     fg_color=SURFACE_2, corner_radius=9, width=164, height=34,
                     font=ctk.CTkFont("Segoe UI", 10, "bold")).pack(side="left")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=28, pady=(0, 14))
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

        self.camera_canvas = Canvas(camera_card, bg="#081321", highlightthickness=0,
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
        self.image_layout = (0, 0, 1, 1, 1.0)
        self.render_tick = 0

        camera_foot = ctk.CTkFrame(camera_card, fg_color="transparent")
        camera_foot.grid(row=2, column=0, sticky="ew", padx=19, pady=(12, 15))
        camera_foot.grid_columnconfigure(0, weight=1)
        self.faces_label = ctk.CTkLabel(camera_foot, text="ROSTROS DETECTADOS  00",
                                        text_color=MUTED, font=ctk.CTkFont("Consolas", 10))
        self.faces_label.grid(row=0, column=0, sticky="w")
        self.retry_button = ctk.CTkButton(
            camera_foot, text="Reintentar cámara", width=128, height=28,
            fg_color=SURFACE_2, hover_color=BORDER, text_color=CYAN,
            font=ctk.CTkFont("Segoe UI", 10, "bold"), command=self.open_camera,
        )
        self.retry_button.grid(row=0, column=1, sticky="e")
        self.retry_button.grid_remove()

        side = ctk.CTkFrame(body, fg_color="transparent")
        side.grid(row=0, column=1, sticky="nsew")
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
        ctk.CTkLabel(metric, text="DISTANCIA LBPH", text_color=MUTED,
                     font=ctk.CTkFont("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(10, 0)
        )
        self.distance_label = ctk.CTkLabel(metric, text="—", text_color=TEXT,
                                            font=ctk.CTkFont("Consolas", 21, "bold"))
        self.distance_label.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 10))
        ctk.CTkLabel(metric, text=f"UMBRAL  ≤ {MAX_DISTANCE:.1f}   ·   MENOR ES MEJOR",
                     text_color=MUTED, font=ctk.CTkFont("Consolas", 9)).grid(
            row=0, column=1, rowspan=2, sticky="e", padx=14
        )

        hardware_row = ctk.CTkFrame(side, fg_color="transparent")
        hardware_row.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        hardware_row.grid_columnconfigure((0, 1), weight=1)
        lock_card = self.card(hardware_row, 13)
        lock_card.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        led_card = self.card(hardware_row, 13)
        led_card.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.lock_value = self.device_label(lock_card, "CERRADURA VIRTUAL", "BLOQUEADA")
        self.led_value = self.device_label(led_card, "LED VIRTUAL", "APAGADO")

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
            fg_color=CYAN, hover_color="#37C6D3", text_color=BG,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
            command=self.open_registration,
        )
        self.register_button.grid(row=3, column=0, sticky="ew", pady=(12, 0))

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=28, pady=(0, 13))
        footer.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(footer, text="SAFEBOX AI  /  RECONOCIMIENTO LOCAL  /  DATOS EN ESTE EQUIPO",
                     text_color=MUTED, font=ctk.CTkFont("Consolas", 9)).grid(
            row=0, column=0, sticky="w"
        )
        self.users_count = ctk.CTkLabel(footer, text="", text_color=MUTED,
                                        font=ctk.CTkFont("Consolas", 9))
        self.users_count.grid(row=0, column=1, sticky="e")
        self.update_user_count()

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
                           text="INICIANDO CÁMARA" if loading else "CÁMARA SIN SEÑAL", fill=TEXT,
                           font=("Segoe UI", 17, "bold"), tags="placeholder")
        canvas.create_text(cx, cy + 82,
                           text="Preparando captura…" if loading else "Usa «Reintentar cámara» al conectar el dispositivo",
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
        self, state: str, detail: str, user: str = "—", distance: float | None = None
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
            "REGISTRANDO": "Capturando rostro",
            "AUTORIZADO": "Identidad verificada",
            "DENEGADO": "Acceso denegado",
            "CÁMARA DESCONECTADA": "Cámara sin señal",
        }[state]
        self.identity_title.configure(text=title)
        self.detail_label.configure(text=detail)
        self.distance_label.configure(text=f"{distance:.1f}" if distance is not None else "—")
        unlocked = state == "AUTORIZADO"
        self.lock_value.configure(text="DESBLOQUEADA" if unlocked else "BLOQUEADA",
                                  text_color=CYAN if unlocked else TEXT)
        led = "AZUL" if unlocked else "ROJO" if state == "DENEGADO" else "APAGADO"
        self.led_value.configure(text=led, text_color=BLUE if led == "AZUL" else RED if led == "ROJO" else MUTED)

    def open_registration(self) -> None:
        if self.register_window is not None and self.register_window.winfo_exists():
            self.register_window.focus()
            return
        window = ctk.CTkToplevel(self, fg_color=BG)
        self.register_window = window
        window.title("SAFEBOX AI · Nuevo usuario")
        window.geometry("440x320")
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
            fg_color=SURFACE, border_color=BORDER, text_color=TEXT,
        )
        self.register_entry.grid(row=2, column=0, sticky="ew", padx=28, pady=(23, 15))
        self.enrollment_progress = ctk.CTkProgressBar(
            window, height=7, progress_color=CYAN, fg_color=SURFACE_2,
        )
        self.enrollment_progress.grid(row=3, column=0, sticky="ew", padx=28)
        self.enrollment_progress.set(0)
        self.enrollment_hint = ctk.CTkLabel(
            window, text="0 / 20 muestras", text_color=MUTED,
            font=ctk.CTkFont("Consolas", 11),
        )
        self.enrollment_hint.grid(row=4, column=0, sticky="w", padx=28, pady=(8, 18))
        actions = ctk.CTkFrame(window, fg_color="transparent")
        actions.grid(row=5, column=0, sticky="ew", padx=28)
        actions.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(actions, text="Cancelar", height=39, fg_color=SURFACE_2,
                      hover_color=BORDER, text_color=TEXT,
                      command=self.cancel_enrollment).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.capture_button = ctk.CTkButton(
            actions, text="Iniciar captura", height=39,
            fg_color=CYAN, hover_color="#37C6D3", text_color=BG,
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
            command=self.start_enrollment,
        )
        self.capture_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.register_entry.focus_set()

    def start_enrollment(self) -> None:
        name = self.register_entry.get().strip()
        if not re.fullmatch(r"[\w .-]{2,40}", name, flags=re.UNICODE):
            messagebox.showwarning(
                "Nombre inválido", "Usa entre 2 y 40 letras, números, espacios, puntos o guiones.",
                parent=self.register_window,
            )
            return
        if not self.backend_ready:
            messagebox.showinfo("Cargando", "El motor facial aún está iniciando.",
                                parent=self.register_window)
            return
        if any(existing.casefold() == name.casefold() for existing in self.known_users):
            messagebox.showwarning("Nombre repetido", "Ese nombre ya está registrado.",
                                   parent=self.register_window)
            return
        if not self.camera_connected:
            messagebox.showerror("Cámara", "No se pudo abrir la cámara.",
                                 parent=self.register_window)
            return
        self.capture_button.configure(state="disabled", text="Preparando…")
        self.register_entry.configure(state="disabled")
        self.backend.send("start_enroll", name)

    def cancel_enrollment(self) -> None:
        self.backend.send("cancel_enroll")
        if self.register_window is not None and self.register_window.winfo_exists():
            self.register_window.grab_release()
            self.register_window.destroy()
        self.register_window = None

    def close_registration_after_success(self) -> None:
        self.modal_close_job = None
        if self.register_window is not None and self.register_window.winfo_exists():
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
        for date, user, result, distance in rows:
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
            metric = f"  ·  LBPH {distance:.1f}" if distance is not None else ""
            ctk.CTkLabel(item, text=date[:10] + "  " + date[11:19] + metric,
                         text_color=MUTED, font=ctk.CTkFont("Consolas", 9)).grid(
                row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 6)
            )

    def poll_backend(self) -> None:
        if not self.running:
            return
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
                self.update_user_count()
                self.refresh_history()
                if self.camera_connected:
                    self.set_state("ESCANEANDO", "Esperando un rostro frente a la cámara")
            elif kind == "camera":
                self.camera_connected = event[1]
                if self.camera_connected:
                    self.camera_badge.configure(text="●  EN VIVO", text_color=CYAN)
                    self.retry_button.grid_remove()
                    self.camera_canvas.delete("placeholder")
                    if self.backend_ready:
                        self.set_state("ESCANEANDO", "Esperando un rostro frente a la cámara")
                else:
                    self.camera_badge.configure(text="●  SIN SEÑAL", text_color=AMBER)
                    self.retry_button.grid()
                    self.camera_canvas.itemconfigure(self.image_item, image="")
                    self.photo = None
                    self.boxes = []
                    self.position_boxes()
                    self.faces_label.configure(text="ROSTROS DETECTADOS  00")
                    self.set_state("CÁMARA DESCONECTADA", event[2])
                    self.draw_placeholder()
            elif kind == "state":
                if self.camera_connected:
                    self.set_state(event[1], event[2], event[3], event[4])
                    if event[5] != self.boxes:
                        self.boxes = event[5]
                        self.faces_label.configure(text=f"ROSTROS DETECTADOS  {len(self.boxes):02d}")
                        self.position_boxes()
            elif kind == "history":
                self.history_rows = event[1]
                self.refresh_history()
            elif kind == "enroll_started":
                if self.register_window is not None and self.register_window.winfo_exists():
                    self.capture_button.configure(text="Capturando…")
            elif kind == "enroll_progress":
                if self.register_window is not None and self.register_window.winfo_exists():
                    self.enrollment_progress.set(event[1] / SAMPLES_REQUIRED)
                    label = "Guardando y entrenando…" if event[1] >= SAMPLES_REQUIRED else f"{event[1]} / {SAMPLES_REQUIRED} muestras"
                    self.enrollment_hint.configure(text=label)
            elif kind == "enroll_done":
                self.known_users = event[2]
                self.update_user_count()
                if self.register_window is not None and self.register_window.winfo_exists():
                    self.enrollment_hint.configure(text=f"{event[1]} registrado correctamente")
                    self.modal_close_job = self.after(650, self.close_registration_after_success)
            elif kind == "enroll_error":
                if self.register_window is not None and self.register_window.winfo_exists():
                    self.enrollment_hint.configure(text="No se completó el registro")
                    self.capture_button.configure(state="normal", text="Reintentar")
                    self.register_entry.configure(state="normal")
                    messagebox.showerror("Registro", event[1], parent=self.register_window)
            elif kind == "storage_error":
                self.detail_label.configure(text=f"No se guardó el intento: {event[1]}")
            elif kind == "engine_error":
                self.backend_ready = False
                self.detail_label.configure(text=f"Error del motor: {event[1]}")
                self.backend.stop()
        if self.camera_connected:
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

    def close(self) -> None:
        if not self.running:
            return
        self.running = False
        for job in (self.transition_job, self.backend_start_job, self.poll_job,
                    self.clock_job, self.modal_close_job):
            if job is not None:
                self.after_cancel(job)
        if self.register_window is not None and self.register_window.winfo_exists():
            self.register_window.grab_release()
            self.register_window.destroy()
            self.register_window = None
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
    detector()
    model = cv2.face.LBPHFaceRecognizer_create()
    image = np.tile(np.arange(160, dtype=np.uint8), (160, 1))
    model.train([image], np.array([7], dtype=np.int32))
    label, _ = model.predict(image)
    if label != 7:
        raise RuntimeError("Falló el reconocedor LBPH")
    print(f"OK: OpenCV {cv2.__version__}, CustomTkinter {ctk.__version__}, Pillow y LBPH")


def main() -> None:
    parser = argparse.ArgumentParser(description="SAFEBOX AI · dashboard biométrico local")
    parser.add_argument("--self-test", action="store_true", help="verifica software sin cámara")
    parser.add_argument("--data-dir", type=Path, default=BASE / "datos")
    parser.add_argument("--mode", choices=("simulacion", "hardware"), default="simulacion")
    parser.add_argument("--port", help="puerto de la WeMos, por ejemplo COM3")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    ctk.set_appearance_mode("dark")
    try:
        app = SafeApp(args.data_dir, args.mode, args.port)
    except Exception as exc:
        raise SystemExit(f"No se pudo iniciar SAFEBOX AI: {exc}") from exc
    app.mainloop()


if __name__ == "__main__":
    main()
