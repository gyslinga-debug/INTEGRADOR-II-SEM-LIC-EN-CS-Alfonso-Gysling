"""Auditoría de calidad de datos (Entrega 2, sección 3.2).

- Completitud: cada archivo se abre con PIL; los ilegibles se excluyen y se reportan.
- Unicidad exacta: MD5 por archivo (tabla hash); se conserva un ejemplar por grupo.
- Casi-duplicados: dHash de 256 bits calculado SOLO sobre el trazado (tinta oscura binarizada,
  sin la cuadrícula rosada), distancias de Hamming vectorizadas y union-find. Solo se fusionan
  pares de la misma clase; los pares casi idénticos de clases distintas se reportan como
  conflictos de etiqueta. Grupos mayores que max_group_size se disuelven (salvaguarda frente
  al fallo detectado en la E2, donde un grupo absorbía las 4 clases).

Uso: python -m ecgvlm.audit [--config ruta.yaml]
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .captions import CLASSES, folder_to_label
from .config import add_common_args, load_config, save_json
from .imageops import crop_margins, to_rgb

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ------------------------------------------------------------------ estructuras de datos
class UnionFind:
    """Conjuntos disjuntos con compresión de camino y unión por rango: O(m · α(n))."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # compresión de camino
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# ------------------------------------------------------------------ huellas
def md5_file(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def trace_mask(img: Image.Image, ink_threshold: int) -> np.ndarray:
    """True donde hay tinta oscura (trazado y texto); la cuadrícula rosada queda fuera."""
    gray = np.asarray(to_rgb(img).convert("L"), dtype=np.uint8)
    return gray < ink_threshold


def trace_dhash(img: Image.Image, hash_size: int = 16, ink_threshold: int = 110,
                pre_cfg: dict | None = None) -> np.ndarray:
    """dHash de hash_size² bits sobre el trazado binarizado (vector uint8 de 0/1)."""
    if pre_cfg:
        img = crop_margins(to_rgb(img), pre_cfg.get("crop_top", 0.0), pre_cfg.get("crop_bottom", 0.0),
                           pre_cfg.get("crop_left", 0.0), pre_cfg.get("crop_right", 0.0))
    ink = trace_mask(img, ink_threshold)
    trace = np.where(ink, 0, 255).astype(np.uint8)
    small = Image.fromarray(trace).resize((hash_size + 1, hash_size), Image.Resampling.BOX)
    a = np.asarray(small, dtype=np.float32)
    return (a[:, 1:] > a[:, :-1]).astype(np.uint8).ravel()


def bits_to_hex(bits: np.ndarray) -> str:
    return np.packbits(bits).tobytes().hex()


def hamming_matrix(bits: np.ndarray) -> np.ndarray:
    """Distancias de Hamming de todos los pares vía producto matricial: O(n² · b)."""
    b = bits.astype(np.float32)
    d = b @ (1.0 - b).T + (1.0 - b) @ b.T
    return np.rint(d).astype(np.int32)


def find_near_duplicate_groups(bits: np.ndarray, labels: list[str], threshold: int,
                               max_group_size: int) -> tuple[np.ndarray, list[dict], dict]:
    """Agrupa casi-duplicados de la misma clase. Devuelve (group_index, conflictos, stats)."""
    n = len(labels)
    uf = UnionFind(n)
    conflicts: list[dict] = []
    if n > 1:
        dist = hamming_matrix(bits)
        ii, jj = np.where(np.triu(dist <= threshold, k=1))
        for i, j in zip(ii.tolist(), jj.tolist()):
            if labels[i] == labels[j]:
                uf.union(i, j)
            else:
                conflicts.append({"i": i, "j": j, "distance": int(dist[i, j])})
        masked = dist + np.eye(n, dtype=np.int32) * 10_000
        nearest = masked.min(axis=1)
    else:
        nearest = np.zeros(n, dtype=np.int32)
    roots = np.array([uf.find(i) for i in range(n)])
    sizes = Counter(roots.tolist())
    dissolved = 0
    for r, s in sizes.items():
        if s > max_group_size:  # salvaguarda: un grupo gigante indica que el hash no discrimina
            dissolved += 1
            for i in np.where(roots == r)[0]:
                roots[i] = i
    _, group_index = np.unique(roots, return_inverse=True)
    final_sizes = Counter(group_index.tolist())
    stats = {
        "pairs_merged_same_class": int(sum(s - 1 for s in final_sizes.values())),
        "groups_total": int(len(final_sizes)),
        "groups_with_duplicates": int(sum(1 for s in final_sizes.values() if s > 1)),
        "largest_group": int(max(final_sizes.values()) if final_sizes else 0),
        "groups_dissolved_too_large": dissolved,
        "nearest_neighbor_distance": {
            "min": int(nearest.min()) if n else 0,
            "p05": float(np.percentile(nearest, 5)) if n else 0.0,
            "median": float(np.median(nearest)) if n else 0.0,
            "histogram": np.histogram(nearest, bins=range(0, 257, 8))[0].tolist() if n else [],
        },
    }
    return group_index, conflicts, stats


# ------------------------------------------------------------------ inventario
def label_for(path: Path, raw_dir: Path) -> str | None:
    """Clase según la carpeta ancestro más cercana cuyo nombre se reconoce."""
    rel_parts = path.relative_to(raw_dir).parts[:-1]
    for part in reversed(rel_parts):
        label = folder_to_label(part)
        if label is not None:
            return label
    return None


def list_images(raw_dir: str | Path) -> list[dict]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de datos crudos: {raw_dir}")
    items = []
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            label = label_for(p, raw_dir)
            if label is not None:
                items.append({"rel_path": str(p.relative_to(raw_dir)), "abs_path": str(p), "label": label})
    return items


def run_audit(cfg: dict, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    a = cfg["audit"]
    raw_dir = Path(cfg["paths"]["raw_dir"])
    work_dir = Path(cfg["paths"]["work_dir"])
    items = list_images(raw_dir)
    if not items:
        raise RuntimeError(f"No se encontraron imágenes con clase reconocible en {raw_dir}")

    rows, bits_list = [], []
    for k, it in enumerate(items):
        row = {"image_id": f"img_{k:04d}", **it, "md5": md5_file(it["abs_path"]), "readable": True,
               "width": 0, "height": 0, "dhash": "", "duplicate_of": "", "error": ""}
        try:
            with Image.open(it["abs_path"]) as im:
                im.load()
                row["width"], row["height"] = im.size
                bits = trace_dhash(im, a["hash_size"], a["ink_threshold"], cfg["preprocess"])
            row["dhash"] = bits_to_hex(bits)
        except Exception as e:  # noqa: BLE001 - se reporta cualquier archivo ilegible
            row["readable"], row["error"] = False, f"{type(e).__name__}: {e}"
            bits = None
        rows.append(row)
        bits_list.append(bits)
        if verbose and (k + 1) % 100 == 0:
            print(f"  auditadas {k + 1}/{len(items)} imágenes")
    df = pd.DataFrame(rows)

    # Duplicados exactos (tabla hash MD5 -> ids)
    by_md5: dict[str, list[int]] = defaultdict(list)
    for i, r in df.iterrows():
        if r["readable"]:
            by_md5[r["md5"]].append(i)
    exact_groups = [ids for ids in by_md5.values() if len(ids) > 1]
    exact_conflicts = []
    for ids in exact_groups:
        keep = ids[0]
        labels = set(df.loc[ids, "label"])
        if len(labels) > 1:
            exact_conflicts.append({"image_ids": df.loc[ids, "image_id"].tolist(), "labels": sorted(labels)})
        for i in ids[1:]:
            df.at[i, "duplicate_of"] = df.at[keep, "image_id"]

    kept = df.index[(df["readable"]) & (df["duplicate_of"] == "")].tolist()
    kept_bits = np.stack([bits_list[i] for i in kept]) if kept else np.zeros((0, a["hash_size"] ** 2), np.uint8)
    kept_labels = df.loc[kept, "label"].tolist()
    group_index, conflicts, stats = find_near_duplicate_groups(
        kept_bits, kept_labels, a["hamming_threshold"], a["max_group_size"])
    df["group_id"] = ""
    for local, i in enumerate(kept):
        df.at[i, "group_id"] = f"g_{group_index[local]:04d}"
    near_conflicts = [{
        "image_a": df.at[kept[c["i"]], "image_id"], "label_a": kept_labels[c["i"]],
        "image_b": df.at[kept[c["j"]], "image_id"], "label_b": kept_labels[c["j"]],
        "distance": c["distance"]} for c in conflicts]

    counts_folder = df.groupby("label").size().reindex(CLASSES, fill_value=0)
    counts_kept = df.loc[kept].groupby("label").size().reindex(CLASSES, fill_value=0)
    report = {
        "raw_dir": str(raw_dir),
        "total_files": int(len(df)),
        "counts_by_class_files": counts_folder.to_dict(),
        "unreadable": df.loc[~df["readable"], ["image_id", "rel_path", "error"]].to_dict("records"),
        "exact_duplicates_removed": int((df["duplicate_of"] != "").sum()),
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_label_conflicts": exact_conflicts,
        "kept_images": int(len(kept)),
        "counts_by_class_kept": counts_kept.to_dict(),
        "class_balance_ratio_max_min": float(counts_kept.max() / max(1, counts_kept.min())),
        "near_duplicates": {"hash_bits": a["hash_size"] ** 2, "hamming_threshold": a["hamming_threshold"],
                            "max_group_size": a["max_group_size"], **stats},
        "near_duplicate_label_conflicts": near_conflicts,
        "resolutions": df.loc[df["readable"]].groupby(["width", "height"]).size()
                         .sort_values(ascending=False).head(10).reset_index(name="n").to_dict("records"),
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(work_dir / "manifest_raw.csv", index=False)
    save_json(report, work_dir / "audit_report.json")
    results_dir = Path(cfg["paths"]["results_dir"])
    save_json(report, results_dir / "audit_report.json")
    _save_histogram(stats, a["hamming_threshold"], results_dir / "audit_hamming_hist.png")
    if verbose:
        print(f"Auditoría: {report['total_files']} archivos, {len(report['unreadable'])} ilegibles, "
              f"{report['exact_duplicates_removed']} duplicados exactos, "
              f"{stats['groups_with_duplicates']} grupos de casi-duplicados, "
              f"{len(near_conflicts)} conflictos de etiqueta.")
        print("Conteo por clase (conservadas):", report["counts_by_class_kept"])
    return df, report


def _save_histogram(stats: dict, threshold: int, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    hist = stats["nearest_neighbor_distance"]["histogram"]
    if not hist:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(0, 256, 8), hist, width=7, align="edge", color="#888888")
    ax.axvline(threshold, color="#1f6feb", linestyle="--", label=f"umbral = {threshold}")
    ax.set_xlabel("Distancia de Hamming al vecino más cercano (de 256 bits)")
    ax.set_ylabel("Imágenes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args(argv)
    run_audit(load_config(args.config))


if __name__ == "__main__":
    main()
