"""Descripciones de referencia por clase y parser determinista texto -> clase (ADR-02).

Las descripciones están en inglés porque BLIP fue preentrenado con vocabulario en inglés
(ADR-04); la interfaz muestra la clase en español.
"""
from __future__ import annotations

import random
import re

CLASSES = ["normal", "mi", "abnormal_hb", "history_mi"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

CLASS_ES = {
    "normal": "Normal",
    "mi": "Infarto de miocardio",
    "abnormal_hb": "Latido anormal",
    "history_mi": "Historial de infarto de miocardio",
}

# 4 paráfrasis por clase: en entrenamiento se muestrea una por época (aumento de texto);
# en evaluación se usan las 4 como referencias múltiples.
TEMPLATES = {
    "normal": [
        "normal ecg with regular sinus rhythm.",
        "this electrocardiogram shows a normal heart rhythm.",
        "ecg of a normal person with no abnormal findings.",
        "normal cardiac electrical activity is observed.",
    ],
    "mi": [
        "ecg showing signs of myocardial infarction.",
        "this electrocardiogram is consistent with an acute myocardial infarction.",
        "findings suggest a heart attack due to myocardial infarction.",
        "ecg of a patient with myocardial infarction.",
    ],
    "abnormal_hb": [
        "ecg showing an abnormal heartbeat.",
        "this electrocardiogram shows an irregular heart rhythm.",
        "abnormal heartbeat pattern consistent with arrhythmia.",
        "ecg of a patient with an abnormal heartbeat.",
    ],
    "history_mi": [
        "ecg of a patient with a history of myocardial infarction.",
        "this electrocardiogram suggests a previous myocardial infarction.",
        "findings consistent with an old myocardial infarction.",
        "ecg showing evidence of prior myocardial infarction.",
    ],
}

REFERENCE_CAPTION = {c: t[0] for c, t in TEMPLATES.items()}


def sample_caption(label: str, rng: random.Random | None = None) -> str:
    rng = rng or random
    return rng.choice(TEMPLATES[label])


def references(label: str) -> list[str]:
    return list(TEMPLATES[label])


def folder_to_label(folder_name: str) -> str | None:
    """Mapea el nombre de una carpeta de Mendeley Data a una clase.

    Precedencia: 'history' antes que 'myocardial'; 'abnormal' antes que 'normal'.
    Las carpetas COVID-19 de la versión 1 del dataset se ignoran.
    """
    name = folder_name.lower()
    if "covid" in name:
        return None
    if "history" in name:
        return "history_mi"
    if "myocardial" in name or re.search(r"\bmi\b", name):
        return "mi"
    if "abnormal" in name:
        return "abnormal_hb"
    if "normal" in name:
        return "normal"
    return None


# ---------------------------------------------------------------- parser texto -> clase
_NEGATION = re.compile(r"\b(?:no|not|without|absence of|negative for|free of)\b[^.,;]*")
_HISTORY = re.compile(
    r"\b(?:history(?:\s+of)?|previous|prior|old|past|healed)\s+(?:an?\s+)?"
    r"(?:acute\s+)?(?:myocardial infarction|mi\b|heart attack|infarct(?:ion)?)"
)
_MI = re.compile(r"myocardial infarction|\bmi\b|heart attack|\binfarct")
_ABNORMAL = re.compile(r"\babnormal|\birregular|\barrhythmi")
_NORMAL = re.compile(r"\bnormal\b")

PARSER_LABELS = CLASSES + ["conflict", "unknown"]


def parse_caption(text: str) -> str:
    """Convierte un texto generado en una clase, 'conflict' (afirma >1 clase) o 'unknown'.

    1. Pasa a minúsculas y elimina cláusulas negadas ("no abnormal findings").
    2. Detecta 'historial de IM' y lo retira antes de buscar IM (precedencia).
    3. Busca IM, latido anormal y normal ('abnormal' no se lee como 'normal' por \\b).
    Complejidad O(L) en el largo del texto.
    """
    if not isinstance(text, str):
        return "unknown"
    t = text.lower()
    t = _NEGATION.sub(" ", t)
    found: set[str] = set()
    if _HISTORY.search(t):
        found.add("history_mi")
        t = _HISTORY.sub(" ", t)
    if _MI.search(t):
        found.add("mi")
    if _ABNORMAL.search(t):
        found.add("abnormal_hb")
    if _NORMAL.search(t):
        found.add("normal")
    if len(found) == 1:
        return found.pop()
    return "conflict" if found else "unknown"


def is_valid(parsed: str) -> bool:
    return parsed in CLASS_TO_IDX
