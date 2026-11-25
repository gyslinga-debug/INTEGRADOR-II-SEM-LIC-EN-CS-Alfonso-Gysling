"""Construcción, guardado y carga de modelos.

Clasificadores (torchvision): resnet50 (B1) y vit_b16 (B2), preentrenados en ImageNet, con
nueva cabeza de 4 clases y ajuste completo. resnet18 y vit_tiny existen solo para la prueba
de integración en CPU.
VLM: BLIP-base de Hugging Face (ADR-01). model_name='tiny' construye un BLIP reducido
localmente, sin descargas, para la prueba de integración.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import nn
from torchvision import models as tvm
from torchvision.models.vision_transformer import VisionTransformer

CLASSIFIERS = ["resnet50", "vit_b16", "resnet18", "vit_tiny"]


def build_classifier(name: str, num_classes: int = 4, pretrained: bool = True, img_size: int = 224) -> nn.Module:
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "vit_b16":
        if pretrained:
            if img_size != 224:
                raise ValueError("ViT-B/16 preentrenado requiere img_size = 224")
            m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            m = tvm.vit_b_16(image_size=img_size)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    elif name == "vit_tiny":
        m = VisionTransformer(image_size=img_size, patch_size=16, num_layers=2, num_heads=2,
                              hidden_dim=64, mlp_dim=128, num_classes=num_classes)
    else:
        raise ValueError(f"Modelo desconocido: {name}. Opciones: {CLASSIFIERS}")
    return m


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def estimate_gflops(model: nn.Module, img_size: int, device) -> float | None:
    """GFLOPs por imagen (multiplicación-suma = 2 FLOPs) con el contador de PyTorch."""
    try:
        from torch.utils.flop_counter import FlopCounterMode

        model.eval()
        x = torch.zeros(1, 3, img_size, img_size, device=device)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        return round(fc.get_total_flops() / 1e9, 2)
    except Exception:  # noqa: BLE001
        return None


def gradcam_layer(model: nn.Module, name: str) -> nn.Module:
    if name.startswith("resnet"):
        return model.layer4
    if name.startswith("vit"):
        return model.encoder.layers[-1]
    raise ValueError(name)


def save_classifier(path: str | Path, model: nn.Module, name: str, img_size: int, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_name": name, "img_size": img_size,
                **(extra or {})}, path)


def load_classifier(path: str | Path, device=None) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model = build_classifier(ckpt["model_name"], pretrained=False, img_size=ckpt["img_size"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device or "cpu").eval()
    return model, ckpt


# ------------------------------------------------------------------ BLIP
def build_blip(model_name: str, img_size: int):
    """Devuelve (modelo, processor). 'tiny' = BLIP reducido local para pruebas."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    if model_name == "tiny":
        return build_tiny_blip(img_size)
    processor = BlipProcessor.from_pretrained(model_name)
    processor.image_processor.size = {"height": img_size, "width": img_size}
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    return model, processor


def load_blip(path: str | Path, device=None):
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(str(path))
    model = BlipForConditionalGeneration.from_pretrained(str(path)).to(device or "cpu").eval()
    return model, processor


def build_tiny_blip(img_size: int = 64):
    """BLIP de ~0,2 M de parámetros con un tokenizador BERT construido desde las plantillas."""
    import re

    from transformers import (BertTokenizerFast, BlipConfig, BlipForConditionalGeneration,
                              BlipImageProcessor, BlipProcessor)

    from .captions import TEMPLATES

    words = {w for ts in TEMPLATES.values() for t in ts for w in re.findall(r"\w+|[^\w\s]", t.lower())}
    words = sorted(words | {"a", "picture", "of", "graph"})  # sin duplicados: cada palabra, un id
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    with tempfile.TemporaryDirectory() as tmp:
        vocab = Path(tmp) / "vocab.txt"
        vocab.write_text("\n".join(specials + words) + "\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    image_processor = BlipImageProcessor(size={"height": img_size, "width": img_size})
    processor = BlipProcessor(image_processor=image_processor, tokenizer=tokenizer)
    config = BlipConfig(
        text_config=dict(vocab_size=len(tokenizer), hidden_size=64, encoder_hidden_size=64,
                         num_hidden_layers=2, num_attention_heads=2, intermediate_size=128,
                         max_position_embeddings=64, bos_token_id=tokenizer.bos_token_id,
                         pad_token_id=tokenizer.pad_token_id, sep_token_id=tokenizer.sep_token_id,
                         eos_token_id=tokenizer.sep_token_id),
        vision_config=dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=128, image_size=img_size, patch_size=16),
        projection_dim=64,
    )
    model = BlipForConditionalGeneration(config)
    return model, processor


def blip_param_counts(model) -> dict:
    vision = count_params(model.vision_model)
    total = count_params(model)
    return {"total": total, "vision": vision, "text": total - vision}
