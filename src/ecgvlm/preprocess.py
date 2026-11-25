"""Auditoría + recorte de cabecera + letterbox + partición estratificada y agrupada.

Salidas en paths.work_dir:
  images/<image_id>.png    imagen procesada (PNG sin pérdida)
  manifest.csv             única fuente de verdad: clase, partición, grupo, huellas
  preprocess_summary.json  conteos por clase y partición
(copias de los resúmenes también en paths.results_dir)

Uso: python -m ecgvlm.preprocess [--config ruta.yaml] [--force]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .audit import run_audit
from .captions import CLASS_TO_IDX, CLASSES, REFERENCE_CAPTION
from .config import add_common_args, load_config, save_json
from .imageops import prepare

SPLITS = ["train", "val", "test"]


def grouped_stratified_split(df: pd.DataFrame, fractions=(0.7, 0.15, 0.15), seed: int = 42) -> pd.Series:
    """Asigna train/val/test por clase, moviendo grupos completos de casi-duplicados.

    Dentro de cada clase los grupos se barajan y se asignan, del más grande al más pequeño,
    a la partición con mayor déficit respecto de su cuota. Así ningún grupo queda repartido
    entre particiones (evita la fuga train -> test) y la proporción por clase se conserva.
    Complejidad O(G · n) para G grupos.
    """
    rng = np.random.default_rng(seed)
    fractions = np.asarray(fractions, dtype=float)
    fractions = fractions / fractions.sum()
    split = pd.Series(index=df.index, dtype=object)
    for _, sub in df.groupby("label", sort=True):
        sizes = sub.groupby("group_id").size()
        order = sizes.index.values[rng.permutation(len(sizes))]
        order = sizes.loc[order].sort_values(ascending=False, kind="stable").index
        targets = fractions * len(sub)
        counts = np.zeros(len(SPLITS))
        for g in order:
            k = int(np.argmax(targets - counts))
            members = sub.index[sub["group_id"] == g]
            split.loc[members] = SPLITS[k]
            counts[k] += len(members)
    return split


def run_preprocess(cfg: dict, force: bool = False, verbose: bool = True) -> pd.DataFrame:
    pre = cfg["preprocess"]
    work_dir = Path(cfg["paths"]["work_dir"])
    img_dir = work_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    raw, report = run_audit(cfg, verbose=verbose)
    df = raw[(raw["readable"]) & (raw["duplicate_of"] == "")].copy().reset_index(drop=True)

    proc_paths = []
    for k, r in df.iterrows():
        out = img_dir / f"{r['image_id']}.png"
        if force or not out.exists():
            with Image.open(r["abs_path"]) as im:
                prepare(im, pre).save(out, format="PNG")
        proc_paths.append(str(out))
        if verbose and (k + 1) % 100 == 0:
            print(f"  procesadas {k + 1}/{len(df)} imágenes")
    df["proc_path"] = proc_paths
    df["label_idx"] = df["label"].map(CLASS_TO_IDX).astype(int)
    df["split"] = grouped_stratified_split(df, pre["split"], pre.get("split_seed", 42))
    df["caption"] = [REFERENCE_CAPTION[l] if s != "train" else "" for l, s in zip(df["label"], df["split"])]

    cols = ["image_id", "proc_path", "rel_path", "label", "label_idx", "split", "group_id",
            "md5", "dhash", "width", "height", "caption"]
    manifest = df[cols]
    manifest.to_csv(work_dir / "manifest.csv", index=False)

    table = pd.crosstab(manifest["label"], manifest["split"]).reindex(index=CLASSES, columns=SPLITS, fill_value=0)
    leaked = manifest.groupby("group_id")["split"].nunique()
    summary = {
        "n_images": int(len(manifest)),
        "by_class_and_split": {c: table.loc[c].to_dict() for c in CLASSES},
        "by_split": manifest["split"].value_counts().reindex(SPLITS, fill_value=0).to_dict(),
        "groups_split_across_partitions": int((leaked > 1).sum()),
        "image_size": pre["size"],
        "crop": {k: pre.get(k, 0.0) for k in ("crop_top", "crop_bottom", "crop_left", "crop_right")},
        "audit": {k: report[k] for k in ("total_files", "exact_duplicates_removed", "kept_images")},
    }
    save_json(summary, work_dir / "preprocess_summary.json")
    save_json(summary, Path(cfg["paths"]["results_dir"]) / "preprocess_summary.json")
    if verbose:
        print("\nImágenes por clase y partición:\n" + table.to_string())
        print(f"Grupos repartidos entre particiones (debe ser 0): {summary['groups_split_across_partitions']}")
    return manifest


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--force", action="store_true", help="Reprocesa aunque existan los PNG.")
    args = parser.parse_args(argv)
    run_preprocess(load_config(args.config), force=args.force)


if __name__ == "__main__":
    main()
