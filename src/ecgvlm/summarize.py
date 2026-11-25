"""Tabla resumen (media ± d.e. entre semillas) y verificación de los criterios de la E2 (tabla 8).

Uso: python -m ecgvlm.summarize
Salidas en results/: summary.csv, summary.md, criteria.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .captions import CLASSES
from .config import add_common_args, load_config, load_json, load_manifest
from .evaluate import bootstrap_ci, classification_metrics

DISPLAY = {
    "majority": "B0 Mayoritaria",
    "resnet50": "B1 ResNet50",
    "vit_b16": "B2 ViT-B/16",
    "vit_b16_template": "B3 ViT + plantilla",
    "blip_zs": "B4 BLIP sin ajustar",
    "blip_ft": "P BLIP-FT",
}
ORDER = list(DISPLAY)


def _run_key(run: str) -> tuple[str, int | None]:
    m = re.match(r"^(.*)_s(\d+)$", run)
    return (m.group(1), int(m.group(2))) if m else (run, None)


def collect(results_dir: Path) -> pd.DataFrame:
    rows = []
    for mpath in sorted(results_dir.glob("*/metrics_test.json")):
        key, seed = _run_key(mpath.parent.name)
        m = load_json(mpath)
        t = m.get("text", {})
        rows.append({"model": key, "seed": seed, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "ci_lo": m["macro_f1_ci95"][0], "ci_hi": m["macro_f1_ci95"][1], "recall_mi": m["recall_mi"],
                     "bleu4": t.get("bleu4"), "rougeL": t.get("rougeL"), "meteor": t.get("meteor"),
                     "valid_rate": t.get("valid_rate")})
    return pd.DataFrame(rows)


def majority_baseline(cfg) -> dict:
    df = load_manifest(cfg)
    majority = df[df.split == "train"]["label"].value_counts().idxmax()
    test = df[df.split == "test"]
    pred = [majority] * len(test)
    m = classification_metrics(test["label"], pred)
    lo, hi = bootstrap_ci(test["label"], pred, n_boot=cfg["evaluate"]["bootstrap"])
    return {"model": "majority", "seed": None, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "ci_lo": lo, "ci_hi": hi, "recall_mi": m["recall_mi"], "bleu4": None, "rougeL": None,
            "meteor": None, "valid_rate": None}


def _fmt(mean, std, n, pct=False):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "—"
    if pct:
        return f"{100 * mean:.1f} %" + (f" ± {100 * std:.1f}" if n > 1 else "")
    return f"{mean:.3f}" + (f" ± {std:.3f}" if n > 1 else "")


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "macro_f1", "ci_lo", "ci_hi", "recall_mi", "bleu4", "rougeL", "meteor", "valid_rate"]
    out = []
    keys = [k for k in ORDER if k in set(runs["model"])] + sorted(set(runs["model"]) - set(ORDER))
    for key in keys:
        g = runs[runs["model"] == key]
        row = {"model": key, "name": DISPLAY.get(key, key), "n_seeds": int(len(g))}
        for c in metrics:
            vals = pd.to_numeric(g[c], errors="coerce").dropna()
            row[f"{c}_mean"] = float(vals.mean()) if len(vals) else None
            row[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(row)
    return pd.DataFrame(out)


def to_markdown(agg: pd.DataFrame) -> str:
    head = "| Modelo | Exactitud | Macro-F1 | IC 95 % | Recall IM | BLEU-4 | ROUGE-L | METEOR | Texto válido |"
    lines = [head, "|" + " --- |" * 9]
    for _, r in agg.iterrows():
        n = r["n_seeds"]
        ci = "—" if r["ci_lo_mean"] is None else f"[{r['ci_lo_mean']:.3f}, {r['ci_hi_mean']:.3f}]"
        lines.append("| " + " | ".join([
            r["name"], _fmt(r["accuracy_mean"], r["accuracy_std"], n), _fmt(r["macro_f1_mean"], r["macro_f1_std"], n),
            ci, _fmt(r["recall_mi_mean"], r["recall_mi_std"], n), _fmt(r["bleu4_mean"], r["bleu4_std"], n),
            _fmt(r["rougeL_mean"], r["rougeL_std"], n), _fmt(r["meteor_mean"], r["meteor_std"], n),
            _fmt(r["valid_rate_mean"], r["valid_rate_std"], n, pct=True)]) + " |")
    return "\n".join(lines)


def check_criteria(agg: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    a = agg.set_index("model")

    def get(model, col):
        if model in a.index and a.at[model, col] is not None and not pd.isna(a.at[model, col]):
            return float(a.at[model, col])
        return None

    def ge(v, t):
        return None if v is None else v >= t

    vit, ft = get("vit_b16", "macro_f1_mean"), get("blip_ft", "macro_f1_mean")
    recall, valid = get("blip_ft", "recall_mi_mean"), get("blip_ft", "valid_rate_mean")
    meteor, std = get("blip_ft", "meteor_mean"), get("blip_ft", "macro_f1_std")
    n_ft = int(a.at["blip_ft", "n_seeds"]) if "blip_ft" in a.index else 0
    rows = [
        ("Reproducción de la línea base: macro-F1 ViT-B/16", "≥ 0,90", vit, ge(vit, 0.90)),
        ("H1: macro-F1 BLIP-FT ≥ macro-F1 ViT − 0,03", "≥ ViT − 0,03", ft,
         None if (vit is None or ft is None) else ft >= vit - 0.03),
        ("Recall de IM (BLIP-FT)", "≥ 0,90", recall, ge(recall, 0.90)),
        ("H2: texto válido (BLIP-FT)", "≥ 95 %", valid, ge(valid, 0.95)),
        ("METEOR de BLIP-FT", "≥ 0,80", meteor, ge(meteor, 0.80)),
        ("Estabilidad: d.e. macro-F1 entre semillas (BLIP-FT)", "≤ 0,03", std if n_ft > 1 else None,
         (std <= 0.03) if (n_ft > 1 and std is not None) else None),
    ]

    rob_path = results_dir / "robustness.csv"
    if rob_path.exists():
        rob = pd.read_csv(rob_path)
        h3 = rob[rob["h3_criterion"] & rob["model"].str.startswith("blip_ft")]
        worst = float(h3["delta_pp"].min()) if len(h3) else None
        rows.append(("H3: Δ macro-F1 BLIP-FT en desenfoque, JPEG y rotación 5°", "caída ≤ 10 pp", worst,
                     None if worst is None else worst >= -10))
    gc = sorted(results_dir.glob("gradcam_*/gradcam_review.csv"))
    if gc:
        rev = pd.read_csv(gc[0])
        manual = rev["en_trazado_manual"].astype(str).str.lower().str.strip()
        filled = manual.isin(["si", "sí", "no"])
        if filled.any():
            share = float(manual[filled].isin(["si", "sí"]).mean())
            rows.append(("Explicabilidad: mapas Grad-CAM en el trazado (revisión manual)", "≥ 80 %", share, share >= 0.80))
        else:
            rows.append(("Explicabilidad: mapas Grad-CAM en el trazado", "≥ 80 %", None, None))
    scores = sorted(results_dir.glob("*/review_scores.json"))
    if scores:
        s = load_json(scores[0])
        vals = [v for v in s["mean"].values() if v is not None]
        rows.append(("Revisión cualitativa: puntaje medio mínimo entre criterios", "≥ 1,5 / 2",
                     min(vals) if vals else None, s.get("meets_threshold_1_5")))
    out = pd.DataFrame(rows, columns=["criterio", "umbral", "valor", "cumple"])
    out["cumple"] = out["cumple"].map({True: "sí", False: "no"}).fillna("pendiente")
    return out


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    results_dir = Path(cfg["paths"]["results_dir"])
    runs = collect(results_dir)
    runs = pd.concat([pd.DataFrame([majority_baseline(cfg)]), runs], ignore_index=True)
    runs.to_csv(results_dir / "runs.csv", index=False)
    agg = aggregate(runs)
    agg.to_csv(results_dir / "summary.csv", index=False)
    md = to_markdown(agg)
    cv = sorted(results_dir.glob("*_cv*/cv_summary.json"))
    if cv:
        md += "\n\nValidación cruzada:\n" + "\n".join(
            f"- {load_json(p)['model']} ({load_json(p)['k']} particiones): macro-F1 = "
            f"{load_json(p)['macro_f1_mean']:.3f} ± {load_json(p)['macro_f1_std']:.3f}" for p in cv)
    crit = check_criteria(agg, results_dir)
    crit.to_csv(results_dir / "criteria.csv", index=False)
    md += "\n\nCriterios de validación (E2, tabla 8):\n\n| Criterio | Umbral | Valor | ¿Cumple? |\n| --- | --- | --- | --- |\n"
    md += "\n".join(f"| {r.criterio} | {r.umbral} | {'—' if r.valor is None or pd.isna(r.valor) else f'{r.valor:.3f}'} "
                    f"| {r.cumple} |" for r in crit.itertuples())
    (results_dir / "summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"\n-> {results_dir / 'summary.csv'}, summary.md, criteria.csv")


if __name__ == "__main__":
    main()
