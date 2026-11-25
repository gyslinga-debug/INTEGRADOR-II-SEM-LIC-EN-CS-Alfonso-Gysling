| Modelo | Exactitud | Macro-F1 | IC 95 % | Recall IM | BLEU-4 | ROUGE-L | METEOR | Texto válido |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 Mayoritaria | 0.479 | 0.162 | [0.135, 0.188] | 0.000 | — | — | — | — |
| B1 ResNet50 | 0.877 ± 0.000 | 0.865 ± 0.012 | [0.751, 0.945] | 1.000 ± 0.000 | — | — | — | — |
| B2 ViT-B/16 | 0.479 ± 0.000 | 0.162 ± 0.000 | [0.135, 0.188] | 0.000 ± 0.000 | — | — | — | — |
| B3 ViT + plantilla | 0.479 ± 0.000 | 0.162 ± 0.000 | [0.135, 0.188] | 0.000 ± 0.000 | 0.493 ± 0.000 | 0.641 ± 0.000 | 0.651 ± 0.000 | 100.0 % ± 0.0 |
| B4 BLIP sin ajustar | 0.000 | 0.000 | [0.000, 0.000] | 0.000 | 0.001 | 0.097 | 0.103 | 0.0 % |
| P BLIP-FT | 0.648 ± 0.171 | 0.396 ± 0.214 | [0.332, 0.454] | 0.000 ± 0.000 | 0.734 ± 0.113 | 0.816 ± 0.080 | 0.796 ± 0.087 | 100.0 % ± 0.0 |

Validación cruzada:
- vit_b16 (5 particiones): macro-F1 = 0.161 ± 0.005

Criterios de validación (E2, tabla 8):

| Criterio | Umbral | Valor | ¿Cumple? |
| --- | --- | --- | --- |
| Reproducción de la línea base: macro-F1 ViT-B/16 | ≥ 0,90 | 0.162 | no |
| H1: macro-F1 BLIP-FT ≥ macro-F1 ViT − 0,03 | ≥ ViT − 0,03 | 0.396 | sí |
| Recall de IM (BLIP-FT) | ≥ 0,90 | 0.000 | no |
| H2: texto válido (BLIP-FT) | ≥ 95 % | 1.000 | sí |
| METEOR de BLIP-FT | ≥ 0,80 | 0.796 | no |
| Estabilidad: d.e. macro-F1 entre semillas (BLIP-FT) | ≤ 0,03 | 0.214 | no |
| H3: Δ macro-F1 BLIP-FT en desenfoque, JPEG y rotación 5° | caída ≤ 10 pp | -19.613 | no |
| Explicabilidad: mapas Grad-CAM en el trazado | ≥ 80 % | — | pendiente |
| Revisión cualitativa: puntaje medio mínimo entre criterios | ≥ 1,5 / 2 | — | no |
