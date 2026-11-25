#!/usr/bin/env bash
# Prueba de integración de punta a punta: datos sintéticos, CPU, sin internet.
# Comprueba que todas las etapas se conectan, NO la calidad de los modelos.
set -euo pipefail
cd "$(dirname "$0")/.."
export ECGVLM_CONFIG=configs/smoke.yaml
rm -rf smoke
START=$(date +%s)

echo "== 1. Datos sintéticos";        python tests/synthetic.py smoke/raw 20
echo "== 2. Auditoría + partición";   python -m ecgvlm.preprocess
echo "== 3. Líneas base";             python -m ecgvlm.train_classifier --model resnet18 --seed 42
                                      python -m ecgvlm.train_classifier --model vit_tiny --seed 42
echo "== 4. Validación cruzada";      python -m ecgvlm.train_classifier --model resnet18 --cv 3 --seed 42
echo "== 5. VLM (BLIP reducido)";     python -m ecgvlm.train_blip --zero_shot
                                      python -m ecgvlm.train_blip --seed 42
echo "== 6. Evaluación y McNemar";    python -m ecgvlm.evaluate --pred smoke/results/blip_ft_s42/pred_test.csv --text
                                      python -m ecgvlm.evaluate --compare smoke/results/vit_tiny_s42/pred_test.csv smoke/results/blip_ft_s42/pred_test.csv
                                      python -m ecgvlm.evaluate --review smoke/results/blip_ft_s42/pred_test.csv
echo "== 7. Robustez";                python -m ecgvlm.robustness --classifier smoke/models/vit_tiny_s42.pt --blip smoke/models/blip_ft_s42 --limit 8
echo "== 8. Grad-CAM";                python -m ecgvlm.gradcam --classifier smoke/models/resnet18_s42.pt --per_class 1
                                      python -m ecgvlm.gradcam --classifier smoke/models/vit_tiny_s42.pt --per_class 1
echo "== 9. Resumen";                 python -m ecgvlm.summarize

echo "== Prueba de integración completa en $(( $(date +%s) - START )) s"
