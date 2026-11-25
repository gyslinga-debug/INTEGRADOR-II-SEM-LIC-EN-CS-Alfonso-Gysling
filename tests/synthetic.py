"""Imágenes sintéticas con forma de ECG para pruebas (sin datos reales).

Cada imagen tiene una cabecera de texto, una cuadrícula rosada y 4 tiras de trazado negro.
La forma de onda depende de la clase (solo para que los modelos tengan algo que aprender):
  normal       latidos regulares
  mi           elevación del segmento ST
  abnormal_hb  intervalos RR irregulares
  history_mi   onda Q profunda y onda T invertida
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FOLDERS = {
    "normal": "Normal Person ECG Images (284x12=3408)",
    "mi": "ECG Images of Myocardial Infarction Patients (240x12=2880)",
    "abnormal_hb": "ECG Images of Patient that have abnormal heartbeat (233x12=2796)",
    "history_mi": "ECG Images of Patient that have History of MI (172x12=2064)",
}


def _beat(t: np.ndarray, label: str) -> np.ndarray:
    """Un latido en t ∈ [0, 1)."""
    g = lambda mu, s, a: a * np.exp(-((t - mu) ** 2) / (2 * s ** 2))  # noqa: E731
    p, r, s = g(0.18, 0.025, 0.12), g(0.40, 0.012, 1.0), g(0.43, 0.012, -0.25)
    q = g(0.37, 0.010, -0.15)
    tw = g(0.68, 0.05, 0.28)
    st = np.zeros_like(t)
    if label == "mi":
        st = 0.35 * ((t > 0.44) & (t < 0.62))
        tw = g(0.66, 0.05, 0.45)
    elif label == "history_mi":
        q = g(0.37, 0.012, -0.55)
        tw = g(0.68, 0.05, -0.25)
    return p + q + r + s + st + tw


def ecg_image(label: str, rng: np.random.Generator, size=(480, 340)) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    for x in range(0, w, 8):
        d.line([(x, 0), (x, h)], fill=(250, 210, 210) if x % 40 else (240, 160, 160), width=1)
    for y in range(0, h, 8):
        d.line([(0, y), (w, y)], fill=(250, 210, 210) if y % 40 else (240, 160, 160), width=1)
    d.rectangle([0, 0, w, int(0.09 * h)], fill=(255, 255, 255))
    d.text((6, 4), f"ID {int(rng.integers(10000, 99999))}  25mm/s 10mm/mV  EDAN SE-3", fill=(0, 0, 0))
    rows = 4
    top = int(0.12 * h)
    row_h = (h - top) / rows
    hr = rng.uniform(55, 95)
    for k in range(rows):
        y0 = top + row_h * (k + 0.6)
        xs = np.arange(4, w - 4)
        sec = xs / 100.0 + rng.uniform(0, 1)
        period = 60.0 / hr
        if label == "abnormal_hb":
            beats = np.cumsum(rng.uniform(0.55, 1.45, 64) * period)
            idx = np.searchsorted(beats, sec)
            start = np.where(idx > 0, beats[np.maximum(idx - 1, 0)], 0)
            length = beats[np.minimum(idx, len(beats) - 1)] - start
            phase = np.clip((sec - start) / np.maximum(length, 1e-6), 0, 0.999)
        else:
            phase = (sec / period) % 1.0
        amp = row_h * rng.uniform(0.38, 0.48)
        ys = y0 - amp * _beat(phase, label) + rng.normal(0, 0.6, len(xs))
        d.line(list(zip(xs.tolist(), ys.tolist())), fill=(10, 10, 10), width=2)
    return img


def make_dataset(out_dir: str | Path, n_per_class: int = 20, seed: int = 0,
                 with_duplicates: bool = True, size=(480, 340)) -> Path:
    """Crea out_dir/<carpeta Mendeley>/*.jpg. Con with_duplicates agrega un duplicado exacto
    y una copia recomprimida (casi-duplicado) de la primera imagen 'normal'."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    for label, folder in FOLDERS.items():
        fdir = out_dir / folder
        fdir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            ecg_image(label, rng, size).save(fdir / f"{label}_{i:03d}.jpg", quality=95)
    if with_duplicates:
        src = out_dir / FOLDERS["normal"] / "normal_000.jpg"
        (out_dir / FOLDERS["normal"] / "normal_dup_exact.jpg").write_bytes(src.read_bytes())
        with Image.open(src) as im:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=70)
        (out_dir / FOLDERS["normal"] / "normal_dup_recompressed.jpg").write_bytes(buf.getvalue())
    return out_dir


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "smoke/raw"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    make_dataset(target, n)
    print(f"Dataset sintético en {target}: {4 * n} imágenes + 2 duplicados")
