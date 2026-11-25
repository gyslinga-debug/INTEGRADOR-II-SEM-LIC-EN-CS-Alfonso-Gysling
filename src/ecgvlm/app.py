"""Demo web (Gradio) con verificación cruzada entre clasificador y VLM (ADR-03).

Muestra la probabilidad por clase, la descripción generada, el mapa Grad-CAM y una alerta
"revisar por especialista" cuando las dos salidas no coinciden o el texto es inválido.
Procesa las imágenes en memoria y no las guarda. Prototipo académico: no es un dispositivo
médico ni entrega diagnósticos.

Uso: python -m ecgvlm.app --classifier models/vit_b16_s42.pt --blip models/blip_ft_s42 [--share]
"""
from __future__ import annotations

import argparse

from .captions import CLASS_ES, CLASSES, parse_caption
from .config import add_common_args, get_device, load_config
from .data import classifier_transform
from .imageops import prepare

DISCLAIMER = ("**Prototipo académico (CINF200).** No es un dispositivo médico ni entrega diagnósticos. "
              "Toda salida debe ser revisada por un profesional de la salud.")


class Predictor:
    def __init__(self, cfg: dict, classifier_path: str | None, blip_path: str | None):
        from .models import load_blip, load_classifier

        self.cfg, self.device = cfg, get_device()
        self.clf, self.clf_meta, self.cam = None, None, None
        if classifier_path:
            from .gradcam import GradCAM

            self.clf, self.clf_meta = load_classifier(classifier_path, self.device)
            self.tf = classifier_transform(self.clf_meta["img_size"], None, False)
            self.cam = GradCAM(self.clf, self.clf_meta["model_name"])
        self.blip = load_blip(blip_path, self.device) if blip_path else None

    def __call__(self, image):
        from .gradcam import overlay
        from .train_blip import generate_texts

        if image is None:
            return {}, "", "", None, "Suba una imagen de ECG."
        img = prepare(image, self.cfg["preprocess"])  # mismo preprocesamiento que en el entrenamiento
        probs_es, clf_class, cam_img = {}, None, None
        if self.clf is not None:
            cam, idx, probs = self.cam(self.tf(img).unsqueeze(0).to(self.device))
            probs_es = {CLASS_ES[c]: float(p) for c, p in zip(CLASSES, probs)}
            clf_class, cam_img = CLASSES[idx], overlay(img, cam)
        text, parsed = "", None
        if self.blip is not None:
            model, processor = self.blip
            b = self.cfg["blip"]
            text = generate_texts(model, processor, [img], self.device, b["num_beams"], b["max_new_tokens"])[0]
            parsed = parse_caption(text)
        parsed_es = CLASS_ES.get(parsed, {"conflict": "Texto contradictorio", "unknown": "Sin clase reconocible"}
                                 .get(parsed, "—")) if parsed else "—"
        if parsed in ("conflict", "unknown"):
            alert = "⚠️ **Revisar por especialista:** la descripción generada no es válida (contradictoria o sin clase)."
        elif clf_class and parsed and parsed != clf_class:
            alert = (f"⚠️ **Revisar por especialista:** el clasificador indica *{CLASS_ES[clf_class]}* "
                     f"y la descripción indica *{CLASS_ES[parsed]}*.")
        elif clf_class or parsed:
            alert = "✅ Clasificador y descripción coinciden. Igualmente requiere revisión profesional."
        else:
            alert = "No hay modelos cargados."
        return probs_es, text, parsed_es, cam_img, alert


def build_app(predictor: Predictor):
    import gradio as gr

    with gr.Blocks(title="ECG-VLM · prototipo académico") as demo:
        gr.Markdown("# Interpretación de imágenes de ECG con modelos de visión-lenguaje")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Imagen del ECG (foto o escaneo)")
                btn = gr.Button("Analizar", variant="primary")
            with gr.Column():
                alert = gr.Markdown()
                probs = gr.Label(num_top_classes=4, label="Clasificador: probabilidad por clase")
                text = gr.Textbox(label="Descripción generada (BLIP, en inglés)")
                parsed = gr.Textbox(label="Clase según la descripción")
                cam = gr.Image(label="Grad-CAM: zonas que pesaron en la predicción")
        btn.click(predictor, inputs=inp, outputs=[probs, text, parsed, cam, alert])
        gr.Markdown(DISCLAIMER)
    return demo


def main(argv=None):
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--classifier", default=None)
    parser.add_argument("--blip", default=None)
    parser.add_argument("--share", action="store_true", help="Enlace público temporal (solo para la presentación)")
    args = parser.parse_args(argv)
    predictor = Predictor(load_config(args.config), args.classifier, args.blip)
    build_app(predictor).launch(share=args.share)


if __name__ == "__main__":
    main()
