"""BLIP: B4 sin ajustar (zero-shot) y P = BLIP ajustado (BLIP-FT).

Ajuste fino: pérdida de modelado de lenguaje con teacher forcing; en cada época se muestrea
una paráfrasis por imagen (aumento de texto). El primer token se fija en [DEC] para alinear
entrenamiento e inferencia. Selección por macro-F1 en validación (texto -> clase con el parser),
early stopping, beam search (k = 3, máximo 40 tokens).

Uso:
  python -m ecgvlm.train_blip --zero_shot            # B4 -> results/blip_zs/
  python -m ecgvlm.train_blip --seed 42              # P  -> results/blip_ft_s42/, models/blip_ft_s42/
  python -m ecgvlm.train_blip --seed 42 --freeze_vision   # si falta memoria o cuota de GPU
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .captions import parse_caption
from .config import add_common_args, get_device, load_config, load_manifest, save_json, set_seed
from .data import BlipDataset, blip_collate
from .evaluate import evaluate_file, macro_f1, validity
from .models import blip_param_counts, build_blip


@torch.no_grad()
def generate_texts(model, processor, images, device, num_beams=3, max_new_tokens=40, batch_size=16) -> list[str]:
    model.eval()
    texts: list[str] = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model.generate(pixel_values=inputs["pixel_values"], num_beams=num_beams,
                                 max_new_tokens=max_new_tokens)
        texts.extend(t.strip() for t in processor.batch_decode(out, skip_special_tokens=True))
    return texts


def predict_split(model, processor, df: pd.DataFrame, cfg: dict, device, images=None) -> pd.DataFrame:
    b = cfg["blip"]
    if images is None:
        ds = BlipDataset(df, train=False, img_size=b["img_size"])
        images = [ds[i][0] for i in range(len(ds))]
    texts = generate_texts(model, processor, images, device, b["num_beams"], b["max_new_tokens"], b["batch_size"] * 2)
    return pd.DataFrame({"image_id": df["image_id"].values, "label": df["label"].values,
                         "pred": [parse_caption(t) for t in texts], "text": texts})


def _load_images(df: pd.DataFrame, img_size: int) -> list:
    ds = BlipDataset(df, train=False, img_size=img_size)
    return [ds[i][0] for i in range(len(ds))]


def run_zero_shot(cfg, skip_existing=False):
    out_dir = Path(cfg["paths"]["results_dir"]) / "blip_zs"
    if skip_existing and (out_dir / "pred_test.csv").exists():
        print(f"[omitido] blip_zs: ya existe {out_dir / 'pred_test.csv'}")
        return
    set_seed(cfg["seeds"][0])
    device = get_device()
    df = load_manifest(cfg)
    model, processor = build_blip(cfg["blip"]["model_name"], cfg["blip"]["img_size"])
    model.to(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        sdf = df[df.split == split].reset_index(drop=True)
        pred = predict_split(model, processor, sdf, cfg, device)
        pred.to_csv(out_dir / f"pred_{split}.csv", index=False)
        evaluate_file(out_dir / f"pred_{split}.csv", n_boot=cfg["evaluate"]["bootstrap"])
    save_json({"run": "blip_zs", "model_name": cfg["blip"]["model_name"], "params": blip_param_counts(model),
               "device": str(device)}, out_dir / "meta.json")
    print(f"== blip_zs listo -> {out_dir}")


def run_finetune(cfg, seed, freeze_vision=None, skip_existing=False):
    b = cfg["blip"]
    run = f"blip_ft_s{seed}"
    results_dir, models_dir = Path(cfg["paths"]["results_dir"]), Path(cfg["paths"]["models_dir"])
    out_dir, model_dir = results_dir / run, models_dir / run
    if skip_existing and (out_dir / "pred_test.csv").exists():
        print(f"[omitido] {run}: ya existe {out_dir / 'pred_test.csv'}")
        return
    set_seed(seed)
    device = get_device()
    df = load_manifest(cfg)
    train_df, val_df, test_df = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    model, processor = build_blip(b["model_name"], b["img_size"])
    freeze_vision = b["freeze_vision"] if freeze_vision is None else freeze_vision
    if freeze_vision:
        for p in model.vision_model.parameters():
            p.requires_grad = False
    model.to(device)
    bos = model.config.text_config.bos_token_id
    print(f"== {run} en {device} | train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"| visión congelada={freeze_vision} | muestreo balanceado={b.get('balanced_sampling', False)}")

    sampler = None
    if b.get("balanced_sampling", False):
        # Sobremuestreo: cada clase aporta lo mismo por época (la pérdida de lenguaje no tiene pesos por clase)
        counts = train_df["label"].value_counts()
        weights = torch.tensor([1.0 / counts[l] for l in train_df["label"]], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_df), replacement=True,
                                        generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(BlipDataset(train_df, True, b["img_size"], cfg["augment"]), batch_size=b["batch_size"],
                              shuffle=sampler is None, sampler=sampler, num_workers=b["num_workers"],
                              collate_fn=blip_collate(processor, bos))
    val_images = _load_images(val_df, b["img_size"])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=b["lr"], weight_decay=b["weight_decay"])
    from transformers import get_cosine_schedule_with_warmup

    total_steps = max(1, len(train_loader) * b["epochs"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(b.get("warmup_ratio", 0.0) * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1, best_epoch, bad, log = -1.0, 0, 0, []
    t_start = time.time()
    for epoch in range(1, b["epochs"] + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(pixel_values=batch["pixel_values"].to(device), input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
            scaler.scale(out.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += out.loss.item() * len(batch["class"])
            n += len(batch["class"])
        val_pred = predict_split(model, processor, val_df, cfg, device, images=val_images)
        f1 = macro_f1(val_pred["label"], val_pred["pred"])
        valid = validity(val_pred["text"].tolist())["valid_rate"]
        secs = time.time() - t0
        log.append({"epoch": epoch, "train_loss": total / max(1, n), "val_macro_f1": f1,
                    "val_valid_rate": valid, "seconds": round(secs, 2)})
        print(f"  época {epoch:02d} pérdida={total / max(1, n):.4f} val macro-F1={f1:.4f} "
              f"texto válido={valid:.1%} ({secs:.1f}s) ej.: \"{val_pred['text'].iloc[0]}\"")
        if f1 > best_f1:
            best_f1, best_epoch, bad = f1, epoch, 0
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            processor.save_pretrained(model_dir)
        else:
            bad += 1
            if bad >= b["patience"]:
                print(f"  early stopping en la época {epoch} (mejor: {best_epoch})")
                break
    train_secs = time.time() - t_start

    from transformers import BlipForConditionalGeneration

    model = BlipForConditionalGeneration.from_pretrained(model_dir).to(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(out_dir / "train_log.csv", index=False)
    infer_ms = None
    for split, sdf in (("val", val_df), ("test", test_df)):
        t1 = time.time()
        pred = predict_split(model, processor, sdf, cfg, device)
        if split == "test":
            infer_ms = round(1000 * (time.time() - t1) / max(1, len(sdf)), 2)
        pred.to_csv(out_dir / f"pred_{split}.csv", index=False)
        evaluate_file(out_dir / f"pred_{split}.csv", n_boot=cfg["evaluate"]["bootstrap"])
    save_json({"run": run, "seed": seed, "device": str(device),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "model_name": b["model_name"], "params": blip_param_counts(model), "freeze_vision": freeze_vision,
               "best_epoch": best_epoch, "best_val_macro_f1": best_f1, "epochs_run": len(log),
               "train_seconds": round(train_secs, 1),
               "mean_epoch_seconds": round(float(np.mean([r["seconds"] for r in log])), 2),
               "inference_ms_per_image": infer_ms, "config": cfg["_config_path"], "hyperparameters": b},
              out_dir / "meta.json")
    print(f"== {run} listo: mejor val macro-F1={best_f1:.4f} (época {best_epoch}) -> {out_dir}")


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--zero_shot", action="store_true", help="B4: BLIP sin ajustar")
    parser.add_argument("--freeze_vision", action="store_true", help="Congela el encoder visual (ahorra memoria)")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.zero_shot:
        run_zero_shot(cfg, args.skip_existing)
    else:
        seed = args.seed if args.seed is not None else cfg["seeds"][0]
        run_finetune(cfg, seed, True if args.freeze_vision else None, args.skip_existing)


if __name__ == "__main__":
    main()
