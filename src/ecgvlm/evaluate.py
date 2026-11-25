"""Métricas de predicción y de texto, intervalos de confianza, McNemar y revisión cualitativa.

Uso:
  python -m ecgvlm.evaluate --pred results/blip_ft_s42/pred_test.csv [--text]
  python -m ecgvlm.evaluate --compare results/vit_b16_s42/pred_test.csv results/blip_ft_s42/pred_test.csv
  python -m ecgvlm.evaluate --review results/blip_ft_s42/pred_test.csv     # crea la planilla de 40 salidas
  python -m ecgvlm.evaluate --score_review results/blip_ft_s42/review_test.csv

Formato de pred_*.csv: image_id, label, pred [, prob_<clase> ...] [, text].
'pred' puede valer 'conflict' o 'unknown' (texto inválido): cuenta como error.
"""
from __future__ import annotations

import argparse
import functools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .captions import CLASSES, PARSER_LABELS, TEMPLATES, parse_caption
from .config import add_common_args, load_config, save_json


# ------------------------------------------------------------------ predicción
def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0))


def classification_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = list(y_true), list(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=PARSER_LABELS)[: len(CLASSES)]
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1(y_true, y_pred),
        "recall_mi": float(r[CLASSES.index("mi")]),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": {"rows_true": CLASSES, "cols_pred": PARSER_LABELS, "values": cm.tolist()},
    }


def bootstrap_ci(y_true, y_pred, metric=macro_f1, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """IC percentil por remuestreo con reemplazo: O(B · n)."""
    y_true, y_pred = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = [metric(y_true[idx], y_pred[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Prueba de McNemar exacta (binomial sobre los desacuerdos), pareada por imagen."""
    from scipy.stats import binomtest

    y_true, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    ca, cb = a == y_true, b == y_true
    n01 = int(np.sum(ca & ~cb))   # acierta A, falla B
    n10 = int(np.sum(~ca & cb))   # falla A, acierta B
    n = n01 + n10
    p = 1.0 if n == 0 else float(binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue)
    return {"a_correct_b_wrong": n01, "a_wrong_b_correct": n10, "discordant": n, "p_value": p,
            "acc_a": float(ca.mean()), "acc_b": float(cb.mean())}


# ------------------------------------------------------------------ texto
_TOKEN = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(str(text).lower())


@functools.lru_cache(maxsize=1)
def _meteor_available() -> bool:
    """METEOR necesita WordNet de NLTK; se descarga una vez (en Colab funciona sin cambios)."""
    import contextlib
    import io as _io

    try:
        import nltk
        from nltk.corpus import wordnet

        try:
            wordnet.ensure_loaded()
        except LookupError:
            with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(_io.StringIO()):
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
            wordnet.ensure_loaded()
        return True
    except Exception:  # noqa: BLE001
        return False


def text_metrics(hypotheses: list[str], labels: list[str]) -> dict:
    """BLEU-4, ROUGE-L (F) y METEOR con las 4 paráfrasis de la clase verdadera como referencias."""
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    from rouge_score import rouge_scorer

    refs_tok = [[tokenize(r) for r in TEMPLATES[l]] for l in labels]
    hyps_tok = [tokenize(h) for h in hypotheses]
    bleu = corpus_bleu(refs_tok, hyps_tok, smoothing_function=SmoothingFunction().method1)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge = np.mean([max(scorer.score(r, str(h))["rougeL"].fmeasure for r in TEMPLATES[l])
                     for h, l in zip(hypotheses, labels)]) if hypotheses else 0.0
    out = {"bleu4": float(bleu), "rougeL": float(rouge), "meteor": None}
    if _meteor_available():
        from nltk.translate.meteor_score import meteor_score

        out["meteor"] = float(np.mean([meteor_score(r, h) for r, h in zip(refs_tok, hyps_tok)])) if hyps_tok else 0.0
    else:
        out["meteor_note"] = "wordnet de NLTK no disponible; METEOR no calculado"
    return out


def validity(texts: list[str]) -> dict:
    """Texto válido = afirma exactamente una clase. Inválido = 'conflict' (más de una) o 'unknown'."""
    parsed = [parse_caption(t) for t in texts]
    n = max(1, len(parsed))
    conflict = sum(p == "conflict" for p in parsed) / n
    unknown = sum(p == "unknown" for p in parsed) / n
    return {"valid_rate": 1 - conflict - unknown, "conflict_rate": conflict, "unknown_rate": unknown,
            "hallucination_rate": conflict + unknown}


# ------------------------------------------------------------------ archivos
def split_of(path: Path) -> str:
    m = re.match(r"pred_(\w+)\.csv", path.name)
    return m.group(1) if m else "test"


def evaluate_frame(df: pd.DataFrame, n_boot: int = 1000, with_text: bool | None = None) -> dict:
    if "pred" not in df.columns and "text" in df.columns:
        df = df.assign(pred=[parse_caption(t) for t in df["text"]])
    metrics = classification_metrics(df["label"], df["pred"])
    metrics["macro_f1_ci95"] = bootstrap_ci(df["label"], df["pred"], n_boot=n_boot)
    if with_text is None:
        with_text = "text" in df.columns
    if with_text and "text" in df.columns:
        texts = df["text"].fillna("").astype(str).tolist()
        metrics["text"] = {**text_metrics(texts, df["label"].tolist()), **validity(texts)}
    return metrics


def evaluate_file(path: str | Path, n_boot: int = 1000, with_text: bool | None = None) -> dict:
    path = Path(path)
    df = pd.read_csv(path)
    metrics = evaluate_frame(df, n_boot=n_boot, with_text=with_text)
    metrics["source"] = str(path)
    save_json(metrics, path.parent / f"metrics_{split_of(path)}.json")
    return metrics


def compare_files(path_a: str | Path, path_b: str | Path, out_dir: str | Path) -> dict:
    a, b = pd.read_csv(path_a), pd.read_csv(path_b)
    m = a[["image_id", "label", "pred"]].merge(b[["image_id", "pred"]], on="image_id", suffixes=("_a", "_b"))
    res = {"a": str(path_a), "b": str(path_b), "n": int(len(m)),
           **mcnemar_exact(m["label"], m["pred_a"], m["pred_b"]),
           "macro_f1_a": macro_f1(m["label"], m["pred_a"]), "macro_f1_b": macro_f1(m["label"], m["pred_b"])}
    name = f"{Path(path_a).parent.name}__vs__{Path(path_b).parent.name}.json"
    save_json(res, Path(out_dir) / "comparisons" / name)
    return res


def sample_per_class(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Hasta n filas por clase, reproducible (compatible con pandas 2 y 3)."""
    parts = [g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("label", sort=True)]
    return pd.concat(parts).reset_index(drop=True) if parts else df.iloc[0:0]


def make_review(path: str | Path, per_class: int = 10, seed: int = 0) -> Path:
    """Planilla para la revisión cualitativa: per_class salidas por clase (40 por defecto),
    con tres criterios a puntuar 0-2: clase correcta, sin alucinación, lenguaje prudente."""
    path = Path(path)
    df = pd.read_csv(path)
    sample = sample_per_class(df, per_class, seed)
    cols = [c for c in ("image_id", "label", "pred", "text") if c in sample.columns]
    review = sample[cols].copy()
    for c in ("clase_correcta_0a2", "sin_alucinacion_0a2", "lenguaje_prudente_0a2", "comentario"):
        review[c] = ""
    out = path.parent / f"review_{split_of(path)}.csv"
    review.to_csv(out, index=False)
    return out


def score_review(path: str | Path) -> dict:
    path = Path(path)
    df = pd.read_csv(path)
    crit = ["clase_correcta_0a2", "sin_alucinacion_0a2", "lenguaje_prudente_0a2"]
    vals = df[crit].apply(pd.to_numeric, errors="coerce")
    res = {"n_scored": int(vals.dropna().shape[0]),
           "mean": {c: (None if vals[c].isna().all() else float(vals[c].mean())) for c in crit}}
    res["meets_threshold_1_5"] = all(v is not None and v >= 1.5 for v in res["mean"].values())
    save_json(res, path.parent / "review_scores.json")
    return res


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--pred", nargs="+", help="Uno o más pred_*.csv a evaluar.")
    parser.add_argument("--text", action="store_true", help="Calcula métricas de texto (automático si hay columna text).")
    parser.add_argument("--compare", nargs=2, metavar=("PRED_A", "PRED_B"))
    parser.add_argument("--review", help="pred_test.csv del que se toma la muestra cualitativa.")
    parser.add_argument("--score_review", help="review_test.csv ya puntuado.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.pred:
        for p in args.pred:
            m = evaluate_file(p, n_boot=cfg["evaluate"]["bootstrap"], with_text=True if args.text else None)
            lo, hi = m["macro_f1_ci95"]
            line = f"{p}: exactitud={m['accuracy']:.3f} macro-F1={m['macro_f1']:.3f} [{lo:.3f}, {hi:.3f}] recall IM={m['recall_mi']:.3f}"
            if "text" in m:
                t = m["text"]
                met = "n/d" if t["meteor"] is None else f"{t['meteor']:.3f}"
                line += f" | BLEU-4={t['bleu4']:.3f} ROUGE-L={t['rougeL']:.3f} METEOR={met} válido={t['valid_rate']:.1%}"
            print(line)
    if args.compare:
        r = compare_files(*args.compare, cfg["paths"]["results_dir"])
        print(f"McNemar: A✓B✗={r['a_correct_b_wrong']} A✗B✓={r['a_wrong_b_correct']} p={r['p_value']:.4f} "
              f"| macro-F1 A={r['macro_f1_a']:.3f} B={r['macro_f1_b']:.3f}")
    if args.review:
        out = make_review(args.review, cfg["evaluate"]["review_per_class"])
        print(f"Planilla de revisión cualitativa: {out} (puntúe cada criterio de 0 a 2)")
    if args.score_review:
        print(score_review(args.score_review))
    if not (args.pred or args.compare or args.review or args.score_review):
        parser.print_help()


if __name__ == "__main__":
    main()
