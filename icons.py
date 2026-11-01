"""Iconos Tabler con caché y fallback; no requiere una pantalla para cargar Pillow."""

from __future__ import annotations

from collections import OrderedDict


class IconProvider:
    def __init__(self, limit: int = 128):
        self.limit = limit
        self._pil: OrderedDict[tuple, object] = OrderedDict()
        self._tk: OrderedDict[tuple, object] = OrderedDict()
        self._api = None
        self._unavailable = False

    def pillow(self, name: str, size: int = 20, color: str = "#FFFFFF",
               stroke_width: float = 1.8):
        key = (name, size, color, stroke_width)
        if key in self._pil:
            self._pil.move_to_end(key)
            return self._pil[key]
        if self._unavailable:
            return None
        try:
            if self._api is None:
                from pytablericons import OutlineIcon, TablerIcons
                self._api = (OutlineIcon, TablerIcons)
            outline, tabler = self._api
            picture = tabler.load(getattr(outline, name), size=size,
                                  color=color, stroke_width=stroke_width)
        except ImportError:
            self._unavailable = True
            return None
        except (AttributeError, OSError, ValueError, RuntimeError):
            return None
        self._pil[key] = picture
        if len(self._pil) > self.limit:
            self._pil.popitem(last=False)
        return picture

    def ctk_image(self, name: str, size: int = 20, color: str = "#FFFFFF",
                  stroke_width: float = 1.8):
        key = (name, size, color, stroke_width)
        if key in self._tk:
            self._tk.move_to_end(key)
            return self._tk[key]
        picture = self.pillow(name, size, color, stroke_width)
        if picture is None:
            return None
        try:
            import customtkinter as ctk
            image = ctk.CTkImage(light_image=picture, dark_image=picture,
                                 size=(size, size))
        except (ImportError, OSError, ValueError, RuntimeError):
            return None
        self._tk[key] = image
        if len(self._tk) > self.limit:
            self._tk.popitem(last=False)
        return image
