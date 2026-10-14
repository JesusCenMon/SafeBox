"""Descarga los modelos oficiales de OpenCV Zoo para inferencia local en CPU."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from core import MODEL_DIR, SFACE_FILE, YUNET_FILE


MODELS = (
    (YUNET_FILE, "face_detection_yunet", 232589,
     "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    (SFACE_FILE, "face_recognition_sface", 9896933,
     "2b0e941e6f16cc048c20aee0c8e31f569118f65d702914540f7bfdc14048d78a"),
)
BASE_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/"
            "47534e27c9851bb1128ccc0102f1145e27f23f98/models")


def download() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    for filename, folder, expected_size, expected_hash in MODELS:
        destination = MODEL_DIR / filename
        if (destination.is_file() and destination.stat().st_size == expected_size
                and hashlib.sha256(destination.read_bytes()).hexdigest() == expected_hash):
            print(f"Ya existe: {filename}")
            continue
        temporary = destination.with_suffix(".download")
        print(f"Descargando {filename}…", flush=True)
        try:
            url = f"{BASE_URL}/{folder}/{filename}"
            def part(index: int) -> bytes:
                start = index * (expected_size // 8)
                end = expected_size - 1 if index == 7 else (index + 1) * (expected_size // 8) - 1
                request = Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise OSError("El servidor no aceptó la descarga parcial")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise OSError("Descarga parcial incompleta")
                return data
            with ThreadPoolExecutor(max_workers=8) as pool, temporary.open("wb") as output:
                for chunk in pool.map(part, range(8)):
                    output.write(chunk)
            if (temporary.stat().st_size != expected_size or
                    hashlib.sha256(temporary.read_bytes()).hexdigest() != expected_hash):
                raise ValueError(f"Descarga incompleta o inválida: {filename}")
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    print("Modelos YuNet y SFace listos en", Path(MODEL_DIR))


if __name__ == "__main__":
    download()
