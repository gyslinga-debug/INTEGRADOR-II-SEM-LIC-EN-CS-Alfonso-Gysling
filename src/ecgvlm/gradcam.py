"""Grad-CAM para los clasificadores (explicabilidad y control de atajos).

Pondera las activaciones de la última capa (ResNet: layer4; ViT: tokens de la última capa
reorganizados en una grilla) por el gradiente medio de la clase.

Además del mapa se calcula un indicador automático: la fracción de la "energía" del mapa que cae
sobre el trazado (tinta oscura, dilatada unos píxeles) frente a la fracción del área que ocupa
el trazado (lift = energía / área). La revisión manual (columna 'en_trazado_manual': si/no)
es la que define el criterio de la E2 (≥ 80 % de mapas centrados en el trazado).

Uso: python -m ecgvlm.gradcam --classifier models/vit_b16_s42.pt [--per_class 5]
Salidas: results/gradcam_<run>/<image_id>.png y gradcam_review.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .captions import CLASS_ES, CLASSES
from .config import add_common_args, get_device, load_config, load_manifest, save_json
from .data import classifier_transform, load_rgb
from .evaluate import sample_per_class
from .models import gradcam_layer, load_classifier


class GradCAM:
    def __init__(self, model, model_name: str):
        self.model, self.name = model, model_name
        self.layer = gradcam_layer(model, model_name)
        self.act = None
        self.grad = None
        self.handle = self.layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.act = output
        output.register_hook(lambda g: setattr(self, "grad", g))

    def remove(self):
        self.handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        self.model.eval()
        x = x.requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, 1)[0].detach().cpu().numpy()
            idx = int(probs.argmax()) if class_idx is None else class_idx
            self.model.zero_grad(set_to_none=True)
            logits[0, idx].backward()
        act, grad = self.act.detach(), self.grad.detach()
        if self.name.startswith("vit"):  # (B, 1+N, D) -> (B, D, h, w), sin el token CLS
            act, grad = act[:, 1:], grad[:, 1:]
            side = int(round(act.shape[1] ** 0.5))
            act = act.reshape(1, side, side, -1).permute(0, 3, 1, 2)
            grad = grad.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = (cam / cam.max()).cpu().numpy() if cam.max() > 0 else cam.cpu().numpy()
        return cam, idx, probs


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    base = img.convert("RGB").resize(cam.shape[::-1])
    heat = (colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    return Image.blend(base, Image.fromarray(heat), alpha)


def trace_lift(img: Image.Image, cam: np.ndarray, ink_threshold: int = 110, dilate: int = 3) -> tuple[float, float]:
    from scipy.ndimage import binary_dilation

    gray = np.asarray(img.convert("L").resize(cam.shape[::-1]), dtype=np.uint8)
    ink = binary_dilation(gray < ink_threshold, iterations=dilate)
    area = float(ink.mean())
    energy = float((cam * ink).sum() / max(cam.sum(), 1e-8))
    return energy, (energy / area if area > 0 else 0.0)


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--per_class", type=int, default=5)
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    device = get_device()
    model, meta = load_classifier(args.classifier, device)
    run = Path(args.classifier).stem
    df = load_manifest(cfg)
    df = df[df.split == args.split]
    sample = sample_per_class(df, args.per_class, seed=0)
    out_dir = Path(cfg["paths"]["results_dir"]) / f"gradcam_{run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tf = classifier_transform(meta["img_size"], None, False)
    cammer = GradCAM(model, meta["model_name"])
    rows = []
    for _, r in sample.iterrows():
        img = load_rgb(r["proc_path"])
        cam, idx, probs = cammer(tf(img).unsqueeze(0).to(device))
        energy, lift = trace_lift(img, cam, cfg["audit"]["ink_threshold"])
        side = Image.new("RGB", (cam.shape[1] * 2, cam.shape[0]), "white")
        side.paste(img.resize(cam.shape[::-1]), (0, 0))
        side.paste(overlay(img, cam), (cam.shape[1], 0))
        side.save(out_dir / f"{r['image_id']}.png")
        rows.append({"image_id": r["image_id"], "label": r["label"], "pred": CLASSES[idx],
                     "prob": float(probs[idx]), "cam_energy_on_trace": round(energy, 4),
                     "trace_lift": round(lift, 3), "en_trazado_manual": ""})
    cammer.remove()
    review = pd.DataFrame(rows)
    review.to_csv(out_dir / "gradcam_review.csv", index=False)
    summary = {"run": run, "n_maps": len(review), "trace_lift_median": float(review["trace_lift"].median()),
               "share_lift_above_1": float((review["trace_lift"] > 1.0).mean()),
               "note": "Proxy automático. Complete 'en_trazado_manual' (si/no) para el criterio ≥ 80 %."}
    save_json(summary, out_dir / "gradcam_summary.json")
    print(f"{len(review)} mapas en {out_dir} | lift mediano={summary['trace_lift_median']:.2f} "
          f"| {summary['share_lift_above_1']:.0%} con más energía sobre el trazado que su área")
    print("Clases:", {c: CLASS_ES[c] for c in CLASSES})


if __name__ == "__main__":
    main()
