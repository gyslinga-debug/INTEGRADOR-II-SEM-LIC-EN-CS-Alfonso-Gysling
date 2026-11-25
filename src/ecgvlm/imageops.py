"""Operaciones de imagen compartidas por la auditoría, el preprocesamiento y la demo.

La demo reutiliza exactamente el mismo preprocesamiento del entrenamiento.
"""
from __future__ import annotations

from PIL import Image


def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        return background
    return img.convert("RGB")


def crop_margins(img: Image.Image, top: float = 0.0, bottom: float = 0.0,
                 left: float = 0.0, right: float = 0.0) -> Image.Image:
    """Recorta fracciones de cada borde (p. ej., top=0.10 elimina la cabecera impresa)."""
    w, h = img.size
    box = (int(round(w * left)), int(round(h * top)),
           int(round(w * (1 - right))), int(round(h * (1 - bottom))))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        raise ValueError(f"Recorte inválido {box} para tamaño {img.size}")
    return img.crop(box)


def letterbox(img: Image.Image, size: int, fill=(255, 255, 255)) -> Image.Image:
    """Ajusta a size x size sin deformar (ADR-06): escala el lado mayor y rellena con blanco."""
    img = to_rgb(img)
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def prepare(img: Image.Image, pre_cfg: dict) -> Image.Image:
    """Recorte de cabecera y márgenes + letterbox, según la sección 'preprocess' del YAML."""
    img = to_rgb(img)
    img = crop_margins(img, pre_cfg.get("crop_top", 0.0), pre_cfg.get("crop_bottom", 0.0),
                       pre_cfg.get("crop_left", 0.0), pre_cfg.get("crop_right", 0.0))
    return letterbox(img, int(pre_cfg["size"]))
