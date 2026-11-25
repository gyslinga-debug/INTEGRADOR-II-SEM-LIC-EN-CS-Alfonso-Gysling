"""Robustez ante variaciones de captura simuladas (H3): 8 perturbaciones del conjunto de test.

Simulan fotos de celular y escaneos de baja calidad. Para cada modelo se reporta
Δ macro-F1 (puntos porcentuales) = macro-F1 perturbado - macro-F1 limpio,
y para BLIP también la tasa de texto válido.

Uso:
  python -m ecgvlm.robustness --classifier models/vit_b16_s42.pt --blip models/blip_ft_s42
Salidas: results/robustness.csv y results/robustness_examples/<perturbación>.png
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from .captions import CLASSES
from .config import add_common_args, get_device, load_config, load_manifest, set_seed
from .data import load_rgb
from .evaluate import macro_f1, validity


def _blur(img, rng):
    return img.filter(ImageFilter.GaussianBlur(radius=1.5))


def _jpeg(img, rng):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=30)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _rotation(img, rng):
    return img.rotate(5, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))


def _perspective(img, rng):
    import torchvision.transforms.functional as F

    w, h = img.size
    d = 0.06
    start = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    end = [[int(w * d), int(h * d / 2)], [w - 1 - int(w * d / 2), 0], [w - 1, h - 1], [0, h - 1 - int(h * d)]]
    return F.perspective(img, start, end, fill=[255, 255, 255])


def _low_light(img, rng):
    return ImageEnhance.Brightness(img).enhance(0.55)


def _downsample(img, rng):
    w, h = img.size
    small = img.resize((max(1, int(w * 0.4)), max(1, int(h * 0.4))), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.BILINEAR)


def _noise(img, rng):
    a = np.asarray(img, dtype=np.float32)
    a = a + rng.normal(0, 12.0, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _low_contrast(img, rng):
    return ImageEnhance.Contrast(img).enhance(0.5)


PERTURBATIONS = {
    "desenfoque_gauss_s1.5": _blur,
    "jpeg_q30": _jpeg,
    "rotacion_5": _rotation,
    "perspectiva": _perspective,
    "baja_iluminacion": _low_light,
    "submuestreo_x0.4": _downsample,
    "ruido_gaussiano": _noise,
    "bajo_contraste": _low_contrast,
}
# Perturbaciones "moderadas" que usa el criterio de H3 (Δ ≤ 10 pp)
H3_CRITERION = ["desenfoque_gauss_s1.5", "jpeg_q30", "rotacion_5"]


def perturb_all(images, name: str, seed: int):
    rng = np.random.default_rng(seed)
    fn = PERTURBATIONS[name]
    return [fn(img, rng) for img in images]


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--classifier", nargs="*", default=[], help="Uno o más .pt de clasificadores")
    parser.add_argument("--blip", default=None, help="Carpeta del BLIP ajustado")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="Usa solo N imágenes (pruebas rápidas)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg["robustness"]["seed"]
    set_seed(seed)
    device = get_device()
    df = load_manifest(cfg)
    df = df[df.split == args.split].reset_index(drop=True)
    if args.limit:
        df = df.groupby("label", sort=True).head(max(1, args.limit // len(CLASSES))).reset_index(drop=True)
    clean = [load_rgb(p) for p in df["proc_path"]]
    results_dir = Path(cfg["paths"]["results_dir"])
    ex_dir = results_dir / "robustness_examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    variants = {"limpio": clean}
    for name in PERTURBATIONS:
        variants[name] = perturb_all(clean, name, seed)
        variants[name][0].save(ex_dir / f"{name}.png")
    clean[0].save(ex_dir / "limpio.png")

    rows = []
    labels = df["label"].tolist()
    for ckpt in args.classifier:
        from .models import load_classifier
        from .train_classifier import make_loader, predict_probs

        model, meta = load_classifier(ckpt, device)
        run = Path(ckpt).stem
        base = None
        for name, imgs in variants.items():
            loader = make_loader(df, meta["img_size"], None, False, cfg["classifier"]["batch_size"], 0, images=imgs)
            pred = [CLASSES[i] for i in predict_probs(model, loader, device).argmax(1)]
            f1 = macro_f1(labels, pred)
            base = f1 if name == "limpio" else base
            rows.append({"model": run, "perturbation": name, "macro_f1": f1,
                         "delta_pp": 100 * (f1 - base), "valid_rate": None})
            print(f"{run:>18} | {name:<22} macro-F1={f1:.4f} Δ={100 * (f1 - base):+.1f} pp")
    if args.blip:
        from .models import load_blip
        from .train_blip import predict_split

        model, processor = load_blip(args.blip, device)
        run = Path(args.blip).name
        base = None
        for name, imgs in variants.items():
            pred = predict_split(model, processor, df, cfg, device, images=imgs)
            f1 = macro_f1(labels, pred["pred"])
            base = f1 if name == "limpio" else base
            valid = validity(pred["text"].tolist())["valid_rate"]
            rows.append({"model": run, "perturbation": name, "macro_f1": f1,
                         "delta_pp": 100 * (f1 - base), "valid_rate": valid})
            print(f"{run:>18} | {name:<22} macro-F1={f1:.4f} Δ={100 * (f1 - base):+.1f} pp válido={valid:.1%}")
    out = pd.DataFrame(rows)
    out["h3_criterion"] = out["perturbation"].isin(H3_CRITERION)
    out.to_csv(results_dir / "robustness.csv", index=False)
    print(f"-> {results_dir / 'robustness.csv'}")


if __name__ == "__main__":
    main()
