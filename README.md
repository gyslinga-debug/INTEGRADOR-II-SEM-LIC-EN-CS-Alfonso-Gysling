# ecg-vlm · Predicción e interpretación de imágenes de ECG con modelos de visión-lenguaje

Proyecto Integrador II (CINF200, NRC 8342) · Alfonso Gysling

A partir de una imagen de electrocardiograma (foto o escaneo), el sistema predice una de cuatro
condiciones (Normal, Infarto de miocardio, Latido anormal, Historial de infarto) y genera una
descripción en lenguaje natural con un modelo de visión-lenguaje (BLIP) ajustado al dominio.
Un clasificador ViT verifica la salida del VLM: si no coinciden, la demo pide revisión por especialista.

> **Prototipo académico.** No es un dispositivo médico ni entrega diagnósticos.

## Ejecución rápida (Google Colab)

Abre [`notebooks/E3_Colab.ipynb`](notebooks/E3_Colab.ipynb) en Colab, activa la GPU T4 y ejecuta las
celdas en orden. El cuaderno clona este repositorio, prepara los datos desde tu Google Drive, corre
todos los experimentos (guardando resultados y modelos en Drive para no perderlos si Colab se
desconecta), genera las tablas y sube los resultados a este repositorio.

## Datos

Dataset público *ECG Images dataset of Cardiac Patients*, versión 2 (Khan & Hussain, Mendeley Data,
2021, DOI [10.17632/gwbz3fsgp8.2](https://doi.org/10.17632/gwbz3fsgp8.2), CC BY 4.0). Descarga el ZIP,
súbelo a tu Drive y apunta `DATA_DIR` del cuaderno a él (o a la carpeta descomprimida). Las imágenes
no se versionan en Git.

## Estructura

```
configs/default.yaml        hiperparámetros, rutas, semillas
configs/smoke.yaml          configuración de la prueba de integración
src/ecgvlm/
  audit.py                  inventario, MD5, dHash del trazado, casi-duplicados (union-find)
  preprocess.py             recorte de cabecera, letterbox 512, partición agrupada 70/15/15
  captions.py               plantillas por clase y parser texto -> clase
  data.py, models.py        datasets, aumentos sin volteos, construcción de modelos
  train_classifier.py       B1 ResNet50, B2 ViT-B/16, B3 ViT + plantilla, validación cruzada
  train_blip.py             B4 BLIP sin ajustar, P BLIP ajustado
  evaluate.py               métricas, IC bootstrap, McNemar, BLEU/ROUGE/METEOR, validez, revisión cualitativa
  robustness.py             8 perturbaciones de captura
  gradcam.py                mapas Grad-CAM + indicador de energía sobre el trazado
  summarize.py              tabla resumen y verificación de criterios
  app.py                    demo Gradio con verificación cruzada
tests/                      pruebas unitarias y generador de ECG sintéticos
scripts/smoke_test.sh       prueba de integración de punta a punta (CPU, sin internet, ~1 min)
notebooks/E3_Colab.ipynb    ejecución completa en Colab
```

## Comandos

```bash
pip install -r requirements.txt && pip install -e .
pytest -q                                              # pruebas unitarias
bash scripts/smoke_test.sh                             # prueba de integración

python -m ecgvlm.preprocess                            # auditoría + partición
python -m ecgvlm.train_classifier --model vit_b16 --seed 42
python -m ecgvlm.train_classifier --model resnet50 --seed 42
python -m ecgvlm.train_classifier --model vit_b16 --cv 5
python -m ecgvlm.train_blip --zero_shot                # B4
python -m ecgvlm.train_blip --seed 42                  # P (BLIP-FT)
python -m ecgvlm.evaluate --compare results/vit_b16_s42/pred_test.csv results/blip_ft_s42/pred_test.csv
python -m ecgvlm.evaluate --review results/blip_ft_s42/pred_test.csv
python -m ecgvlm.robustness --classifier models/vit_b16_s42.pt --blip models/blip_ft_s42
python -m ecgvlm.gradcam --classifier models/vit_b16_s42.pt
python -m ecgvlm.summarize
python -m ecgvlm.app --classifier models/vit_b16_s42.pt --blip models/blip_ft_s42 --share
```

Todas las rutas se configuran en el YAML (`--config` o variable `ECGVLM_CONFIG`). Con
`--skip_existing` los entrenamientos omiten las corridas ya terminadas.

## Salidas principales (`results/`)

| Archivo | Contenido |
| --- | --- |
| `audit_report.json`, `audit_hamming_hist.png` | Calidad de datos, duplicados, conflictos de etiqueta |
| `preprocess_summary.json` | Imágenes por clase y partición |
| `<modelo>_s<semilla>/pred_*.csv`, `metrics_*.json` | Predicciones y métricas (IC 95 %, matriz de confusión, texto) |
| `<modelo>_s<semilla>/train_log.csv`, `meta.json` | Curva de entrenamiento, tiempos, parámetros, GFLOPs |
| `vit_b16_cv5/cv_summary.json` | Validación cruzada agrupada |
| `comparisons/*.json` | Prueba de McNemar |
| `robustness.csv`, `robustness_examples/` | Δ macro-F1 por perturbación |
| `gradcam_<modelo>/` | Mapas y planilla de revisión |
| `summary.csv`, `summary.md`, `criteria.csv` | Tabla final y cumplimiento de criterios |

## Uso de IA

Este código se desarrolló con asistencia de Claude (Anthropic). Ver la declaración de uso de IA del informe final.
