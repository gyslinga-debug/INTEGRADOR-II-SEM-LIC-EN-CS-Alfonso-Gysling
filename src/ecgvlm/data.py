"""Datasets y aumentos de datos.

Aumento (ADR-05): recorte aleatorio 85-100 %, rotación ±3°, brillo y contraste ±20 %.
Nunca se voltea la imagen: un volteo invierte el tiempo o la polaridad del ECG.
"""
from __future__ import annotations

import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .captions import REFERENCE_CAPTION, TEMPLATES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def geometric_augment(img_size: int, aug: dict) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=tuple(aug["crop_scale"]), ratio=(0.95, 1.05)),
        T.RandomRotation(aug["rotation"], fill=(255, 255, 255)),
        T.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"]),
    ])


def classifier_transform(img_size: int, aug: dict | None, train: bool) -> T.Compose:
    if train and aug:
        return T.Compose([geometric_augment(img_size, aug), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class ImageDataset(Dataset):
    """Imágenes procesadas + índice de clase para los clasificadores."""

    def __init__(self, df: pd.DataFrame, transform, images: list[Image.Image] | None = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.images = images  # imágenes ya cargadas (p. ej., perturbadas para robustez)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        img = self.images[i] if self.images is not None else load_rgb(self.df.at[i, "proc_path"])
        return self.transform(img), int(self.df.at[i, "label_idx"])


class BlipDataset(Dataset):
    """Devuelve (imagen PIL, clase, descripción). En entrenamiento la descripción es una
    paráfrasis al azar (aumento de texto); en evaluación, la referencia fija."""

    def __init__(self, df: pd.DataFrame, train: bool, img_size: int, aug: dict | None = None,
                 images: list[Image.Image] | None = None):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.images = images
        self.aug = geometric_augment(img_size, aug) if (train and aug) else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        label = self.df.at[i, "label"]
        img = self.images[i] if self.images is not None else load_rgb(self.df.at[i, "proc_path"])
        if self.aug is not None:
            img = self.aug(img)
        caption = random.choice(TEMPLATES[label]) if self.train else REFERENCE_CAPTION[label]
        return img, label, caption


def blip_collate(processor, bos_token_id: int):
    """Tokeniza y alinea el primer token con [DEC], como hace generate() de BLIP.

    Sin este ajuste el entrenamiento empezaría con [CLS] y la inferencia con [DEC]
    (desalineación silenciosa detectada en la E2).
    """
    def collate(batch):
        images, labels, captions = zip(*batch)
        enc = processor(images=list(images), text=list(captions), padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].clone()
        input_ids[:, 0] = bos_token_id
        targets = input_ids.clone()
        targets[enc["attention_mask"] == 0] = -100
        return {"pixel_values": enc["pixel_values"], "input_ids": input_ids,
                "attention_mask": enc["attention_mask"], "labels": targets, "class": list(labels)}
    return collate


def pil_collate(batch):
    images, labels, captions = zip(*batch)
    return list(images), list(labels), list(captions)


def class_weights(labels: list[int], n_classes: int) -> torch.Tensor:
    """Pesos inversos a la frecuencia: n / (K · n_c)."""
    counts = torch.bincount(torch.tensor(labels), minlength=n_classes).float().clamp(min=1)
    return counts.sum() / (n_classes * counts)
