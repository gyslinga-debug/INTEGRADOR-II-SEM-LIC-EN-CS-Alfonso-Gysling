"""Líneas base de clasificación: B1 ResNet50, B2 ViT-B/16 y B3 ViT + plantilla.

Entrenamiento: AdamW, coseno, entropía cruzada ponderada por clase, fp16 en GPU,
early stopping por macro-F1 en validación (el test queda sellado hasta el final).

Uso:
  python -m ecgvlm.train_classifier --model vit_b16 --seed 42
  python -m ecgvlm.train_classifier --model resnet50 --seed 43
  python -m ecgvlm.train_classifier --model vit_b16 --cv 5        # validación cruzada agrupada

Salidas: models/<modelo>_s<semilla>.pt y results/<modelo>_s<semilla>/{pred_val,pred_test}.csv,
metrics_*.json, train_log.csv, meta.json. Para modelos ViT también results/<modelo>_template_s<semilla>/ (B3).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader

from .captions import CLASSES, REFERENCE_CAPTION
from .config import add_common_args, get_device, load_config, load_manifest, save_json, set_seed
from .data import ImageDataset, class_weights, classifier_transform
from .evaluate import evaluate_file, macro_f1
from .models import build_classifier, count_params, estimate_gflops, save_classifier


def make_loader(df, img_size, aug, train, batch_size, num_workers, images=None):
    ds = ImageDataset(df, classifier_transform(img_size, aug, train), images=images)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False)


@torch.no_grad()
def predict_probs(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


def model_lr(c: dict, model_name: str) -> float:
    """Tasa de aprendizaje del modelo: lr_by_model[modelo] si existe; si no, lr."""
    return float((c.get("lr_by_model") or {}).get(model_name, c["lr"]))


def warmup_cosine(warmup: int, total: int):
    """Factor por época: subida lineal durante 'warmup' épocas y luego decaimiento coseno."""
    import math

    def factor(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return factor


def train_model(cfg: dict, model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                device, pretrained: bool | None = None, verbose: bool = True):
    c = cfg["classifier"]
    pretrained = c["pretrained"] if pretrained is None else pretrained
    img_size = c["img_size"]
    model = build_classifier(model_name, len(CLASSES), pretrained=pretrained, img_size=img_size).to(device)
    train_loader = make_loader(train_df, img_size, cfg["augment"], True, c["batch_size"], c["num_workers"])
    val_loader = make_loader(val_df, img_size, None, False, c["batch_size"], c["num_workers"])

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_df["label_idx"].tolist(), len(CLASSES)).to(device))
    lr = model_lr(c, model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=c["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(c.get("warmup_epochs", 0), c["epochs"]))
    if verbose:
        print(f"  lr={lr:g}, calentamiento={c.get('warmup_epochs', 0)} épocas, máx. {c['epochs']} épocas")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1, best_state, best_epoch, bad, log = -1.0, None, 0, 0, []
    for epoch in range(1, c["epochs"] + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
            n += len(y)
        scheduler.step()
        probs = predict_probs(model, val_loader, device)
        val_pred = [CLASSES[i] for i in probs.argmax(1)]
        f1 = macro_f1(val_df["label"], val_pred)
        acc = float(np.mean(np.array(val_pred) == val_df["label"].values))
        secs = time.time() - t0
        log.append({"epoch": epoch, "train_loss": total / max(1, n), "val_macro_f1": f1, "val_accuracy": acc,
                    "lr": scheduler.get_last_lr()[0], "seconds": round(secs, 2)})
        if verbose:
            print(f"  época {epoch:02d} pérdida={total / max(1, n):.4f} val macro-F1={f1:.4f} ({secs:.1f}s)")
        if f1 > best_f1:
            best_f1, best_epoch, bad = f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= c["patience"]:
                if verbose:
                    print(f"  early stopping en la época {epoch} (mejor: {best_epoch})")
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(log), best_f1, best_epoch


def write_predictions(df: pd.DataFrame, probs: np.ndarray, path: Path) -> pd.DataFrame:
    out = pd.DataFrame({"image_id": df["image_id"].values, "label": df["label"].values,
                        "pred": [CLASSES[i] for i in probs.argmax(1)]})
    for j, c in enumerate(CLASSES):
        out[f"prob_{c}"] = probs[:, j]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def run_single(cfg, model_name, seed, pretrained=None, skip_existing=False):
    results_dir, models_dir = Path(cfg["paths"]["results_dir"]), Path(cfg["paths"]["models_dir"])
    run = f"{model_name}_s{seed}"
    out_dir = results_dir / run
    if skip_existing and (out_dir / "pred_test.csv").exists():
        print(f"[omitido] {run}: ya existe {out_dir / 'pred_test.csv'}")
        return
    set_seed(seed)
    device = get_device()
    df = load_manifest(cfg)
    train_df, val_df, test_df = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    print(f"== {run} en {device} | train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    t0 = time.time()
    model, log, best_f1, best_epoch = train_model(cfg, model_name, train_df, val_df, device, pretrained)
    train_secs = time.time() - t0
    c = cfg["classifier"]
    save_classifier(models_dir / f"{run}.pt", model, model_name, c["img_size"], {"seed": seed, "classes": CLASSES})

    out_dir.mkdir(parents=True, exist_ok=True)
    log.to_csv(out_dir / "train_log.csv", index=False)
    for split_name, split_df in (("val", val_df), ("test", test_df)):
        loader = make_loader(split_df, c["img_size"], None, False, c["batch_size"], c["num_workers"])
        t1 = time.time()
        probs = predict_probs(model, loader, device)
        infer_ms = 1000 * (time.time() - t1) / max(1, len(split_df))
        pred = write_predictions(split_df, probs, out_dir / f"pred_{split_name}.csv")
        evaluate_file(out_dir / f"pred_{split_name}.csv", n_boot=cfg["evaluate"]["bootstrap"])
        if model_name.startswith("vit") and split_name == "test":
            # B3: el ViT predice la clase y se agrega una plantilla fija (texto "gratis", sin VLM)
            tpl_dir = results_dir / f"{model_name}_template_s{seed}"
            tpl = pred.assign(text=[REFERENCE_CAPTION[p] for p in pred["pred"]])
            tpl_dir.mkdir(parents=True, exist_ok=True)
            tpl.to_csv(tpl_dir / "pred_test.csv", index=False)
            evaluate_file(tpl_dir / "pred_test.csv", n_boot=cfg["evaluate"]["bootstrap"])
    meta = {"run": run, "model": model_name, "seed": seed, "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "params": count_params(model), "gflops_per_image": estimate_gflops(model, c["img_size"], device),
            "img_size": c["img_size"], "lr": model_lr(c, model_name), "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
            "epochs_run": int(len(log)), "train_seconds": round(train_secs, 1),
            "mean_epoch_seconds": round(float(log["seconds"].mean()), 2),
            "inference_ms_per_image": round(infer_ms, 2), "config": cfg["_config_path"],
            "hyperparameters": cfg["classifier"], "augment": cfg["augment"]}
    save_json(meta, out_dir / "meta.json")
    print(f"== {run} listo: mejor val macro-F1={best_f1:.4f} (época {best_epoch}); resultados en {out_dir}")


def run_cv(cfg, model_name, k, seed, pretrained=None):
    set_seed(seed)
    device = get_device()
    df = load_manifest(cfg)
    pool = df[df.split.isin(["train", "val"])].reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for fold, (tr, va) in enumerate(sgkf.split(pool, pool["label_idx"], pool["group_id"]), start=1):
        print(f"== CV {model_name} partición {fold}/{k} (train={len(tr)} val={len(va)})")
        set_seed(seed + fold)
        _, log, best_f1, best_epoch = train_model(cfg, model_name, pool.iloc[tr], pool.iloc[va].reset_index(drop=True),
                                                  device, pretrained, verbose=False)
        best = log.loc[log["epoch"] == best_epoch].iloc[0]
        folds.append({"fold": fold, "n_train": int(len(tr)), "n_val": int(len(va)), "macro_f1": best_f1,
                      "accuracy": float(best["val_accuracy"]), "best_epoch": int(best_epoch)})
        print(f"   macro-F1={best_f1:.4f}")
    f1s = np.array([f["macro_f1"] for f in folds])
    summary = {"model": model_name, "k": k, "seed": seed, "folds": folds,
               "macro_f1_mean": float(f1s.mean()), "macro_f1_std": float(f1s.std(ddof=1)) if k > 1 else 0.0}
    out = Path(cfg["paths"]["results_dir"]) / f"{model_name}_cv{k}" / "cv_summary.json"
    save_json(summary, out)
    print(f"== CV {model_name}: macro-F1 = {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f} -> {out}")


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--model", default="vit_b16", help="resnet50 | vit_b16 (resnet18 | vit_tiny para pruebas)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cv", type=int, default=0, help="k particiones de validación cruzada (0 = no)")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--skip_existing", action="store_true", help="Omite la corrida si ya existe pred_test.csv")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seeds"][0]
    pretrained = False if args.no_pretrained else None
    if args.cv:
        run_cv(cfg, args.model, args.cv, seed, pretrained)
    else:
        run_single(cfg, args.model, seed, pretrained, args.skip_existing)


if __name__ == "__main__":
    main()
