"""Pruebas unitarias del pipeline (pytest -q). No requieren GPU ni internet."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from synthetic import FOLDERS, ecg_image, make_dataset  # noqa: E402

from ecgvlm.audit import UnionFind, hamming_matrix, run_audit, trace_dhash  # noqa: E402
from ecgvlm.captions import CLASSES, TEMPLATES, folder_to_label, parse_caption  # noqa: E402
from ecgvlm.evaluate import (bootstrap_ci, classification_metrics, mcnemar_exact,  # noqa: E402
                             text_metrics, validity)
from ecgvlm.imageops import letterbox  # noqa: E402
from ecgvlm.preprocess import grouped_stratified_split  # noqa: E402

PRE = {"crop_top": 0.10, "crop_bottom": 0.0, "crop_left": 0.0, "crop_right": 0.0, "size": 128}


# ------------------------------------------------------------------ parser texto -> clase
def test_parser_reads_every_template_as_its_class():
    for label, texts in TEMPLATES.items():
        for t in texts:
            assert parse_caption(t) == label, t


def test_parser_abnormal_is_not_read_as_normal():
    assert parse_caption("abnormal heartbeat detected") == "abnormal_hb"
    assert parse_caption("ABNORMAL rhythm") == "abnormal_hb"


def test_parser_detects_conflict_and_unknown():
    assert parse_caption("normal ecg with myocardial infarction") == "conflict"
    assert parse_caption("a picture of a graph") == "unknown"
    assert parse_caption("") == "unknown"
    assert parse_caption(None) == "unknown"


def test_parser_history_precedence_and_negation():
    assert parse_caption("history of myocardial infarction") == "history_mi"
    assert parse_caption("prior mi") == "history_mi"
    assert parse_caption("a patient with a history myocardial infarction") == "history_mi"
    assert parse_caption("normal ecg with no signs of infarction") == "normal"
    assert parse_caption("ecg without abnormal findings, normal rhythm") == "normal"


def test_folder_to_label_mendeley_names():
    for label, folder in FOLDERS.items():
        assert folder_to_label(folder) == label
    assert folder_to_label("ECG Images of COVID-19 Patients") is None
    assert folder_to_label("misc") is None


# ------------------------------------------------------------------ hash, Hamming, union-find
def test_hamming_matrix():
    bits = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 0, 0]], dtype=np.uint8)
    d = hamming_matrix(bits)
    assert d.tolist() == [[0, 1, 4], [1, 0, 3], [4, 3, 0]]


def test_union_find_groups_transitively():
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)
    assert uf.find(3) != uf.find(0)
    assert not uf.union(2, 0)


def test_dhash_is_robust_to_recompression():
    img = ecg_image("normal", np.random.default_rng(1))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=60)
    recompressed = Image.open(io.BytesIO(buf.getvalue()))
    a, b = trace_dhash(img, pre_cfg=PRE), trace_dhash(recompressed, pre_cfg=PRE)
    assert a.size == 256
    assert int((a != b).sum()) <= 18


def test_dhash_separates_different_patients():
    rng = np.random.default_rng(2)
    hashes = [trace_dhash(ecg_image("normal", rng), pre_cfg=PRE) for _ in range(6)]
    d = hamming_matrix(np.stack(hashes))
    off_diag = d[~np.eye(len(hashes), dtype=bool)]
    assert off_diag.min() > 18, "ECG distintos no deben agruparse por la cuadrícula común"


def test_audit_finds_exact_and_near_duplicates(tmp_path):
    raw = make_dataset(tmp_path / "raw", n_per_class=4, seed=3)
    cfg = {"paths": {"raw_dir": str(raw), "work_dir": str(tmp_path / "work"), "results_dir": str(tmp_path / "res")},
           "audit": {"hash_size": 16, "ink_threshold": 110, "hamming_threshold": 18, "max_group_size": 12},
           "preprocess": PRE}
    df, report = run_audit(cfg, verbose=False)
    assert report["total_files"] == 18
    assert report["exact_duplicates_removed"] == 1
    kept = df[df["duplicate_of"] == ""]
    g = kept.set_index("rel_path")["group_id"]
    normal = FOLDERS["normal"]
    assert g[f"{normal}/normal_000.jpg"] == g[f"{normal}/normal_dup_recompressed.jpg"]
    assert report["near_duplicates"]["largest_group"] == 2
    assert not report["near_duplicate_label_conflicts"]


# ------------------------------------------------------------------ imagen y partición
def test_letterbox_keeps_aspect_and_pads_white():
    img = Image.new("RGB", (200, 100), (0, 0, 0))
    out = letterbox(img, 64)
    assert out.size == (64, 64)
    a = np.asarray(out)
    assert a[0, 0].tolist() == [255, 255, 255]          # relleno arriba
    assert a[32, 32].tolist() == [0, 0, 0]               # contenido al centro
    assert (a[:, 32, 0] == 0).sum() == 32                # 100/200 * 64 = 32 filas de contenido


def test_grouped_split_has_no_leakage_and_keeps_classes():
    rows = []
    for c in CLASSES:
        for i in range(40):
            rows.append({"label": c, "group_id": f"{c}_{i // 2 if i < 10 else i}"})  # algunos grupos de 2
    df = pd.DataFrame(rows)
    split = grouped_stratified_split(df, (0.7, 0.15, 0.15), seed=0)
    df["split"] = split
    assert (df.groupby("group_id")["split"].nunique() == 1).all()
    table = pd.crosstab(df["label"], df["split"])
    assert (table > 0).all().all()
    assert table["train"].between(26, 30).all()


# ------------------------------------------------------------------ métricas
def test_classification_metrics_count_invalid_text_as_error():
    y = ["normal", "mi", "abnormal_hb", "history_mi"]
    p = ["normal", "mi", "conflict", "history_mi"]
    m = classification_metrics(y, p)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["per_class"]["abnormal_hb"]["recall"] == 0.0
    assert m["macro_f1"] == pytest.approx(0.75)
    assert m["recall_mi"] == 1.0


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    y = rng.choice(CLASSES, 200).tolist()
    p = [t if rng.random() < 0.8 else rng.choice(CLASSES) for t in y]
    from ecgvlm.evaluate import macro_f1

    lo, hi = bootstrap_ci(y, p, n_boot=300, seed=1)
    assert lo <= macro_f1(y, p) <= hi
    assert hi - lo < 0.25


def test_mcnemar_exact():
    y = ["normal"] * 10
    a = ["normal"] * 10                       # A acierta todo
    b = ["mi"] * 6 + ["normal"] * 4           # B falla 6
    r = mcnemar_exact(y, a, b)
    assert (r["a_correct_b_wrong"], r["a_wrong_b_correct"]) == (6, 0)
    assert r["p_value"] == pytest.approx(2 * 0.5 ** 6)
    assert mcnemar_exact(y, a, a)["p_value"] == 1.0


def test_text_metrics_and_validity():
    labels = ["normal", "mi"]
    perfect = [TEMPLATES["normal"][0], TEMPLATES["mi"][0]]
    m = text_metrics(perfect, labels)
    assert m["bleu4"] == pytest.approx(1.0)
    assert m["rougeL"] == pytest.approx(1.0)
    v = validity(perfect + ["normal ecg with myocardial infarction", "a graph"])
    assert v["valid_rate"] == pytest.approx(0.5)
    assert v["conflict_rate"] == pytest.approx(0.25)


def test_blip_collate_aligns_first_token_with_dec():
    from ecgvlm.data import blip_collate
    from ecgvlm.models import build_tiny_blip

    model, processor = build_tiny_blip(32)
    bos = model.config.text_config.bos_token_id
    img = Image.new("RGB", (32, 32), "white")
    batch = blip_collate(processor, bos)([(img, "normal", TEMPLATES["normal"][0]),
                                          (img, "mi", TEMPLATES["mi"][1])])
    assert (batch["input_ids"][:, 0] == bos).all()
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    loss = model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"],
                 attention_mask=batch["attention_mask"], labels=batch["labels"]).loss
    assert np.isfinite(loss.item())
