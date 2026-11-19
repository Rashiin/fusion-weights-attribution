# Code and data for "Fusion weights are not feature importance"

Code and data for the manuscript "Fusion weights are not feature importance: a margin-based attribution for weighted
ensembles of feature-group experts in leaf disease recognition", submitted to Pattern Analysis and Applications
(Springer Nature). The same files are provided as Online Resource 2 of the manuscript. Authors: Rashin Gholijani
Farahani, Azam Bastanfard, Javad Mohammadzadeh. Corresponding author: Azam Bastanfard, Department of Computer
Engineering, Karaj Branch (Ka.C.), Islamic Azad University, Karaj, Iran; bastanfard@iau.ac.ir.

Everything needed to reproduce the tables and figures of the manuscript and its supplementary material.
All results were produced in one environment: Python 3.13.16, NumPy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1,
SciPy 1.18.1, OpenCV 5.0.0 (`opencv-python` 5.0.0.93), Matplotlib 3.11.2 (see `requirements.txt`), and no
deep-learning framework is needed for them. The two exceptions are the fine-tuned MobileNetV2 reference and
the latency measurements of Section 5.1, which were run in a separate, older environment; their scripts and
outputs are in `legacy/` (see `legacy/README.md`).

## Contents

```
data/features/   87 handcrafted features per image (color 57, texture 22, shape 8), fixed partition
data/splits/     the fixed partition (image paths relative to the PlantVillage folder)
src/             all scripts (listed below)
results/         per-partition outputs (CSV), results.json, and the controls in sub-folders
                 (results/subgroups: the five-subgroup control of Section 5.8)
legacy/          fine-tuned reference network and latency scripts with their outputs (separate environment)
plantdoc/        field-image control of Section 5.9 (deep experts on PlantDoc), with its own README
figures/         figures of the manuscript
```

## Reproducing the results

Steps 1 and 2 use only the files in this package, and so does step 3 after its first command, whose outputs are
included. The first command of step 3 and step 4 need the PlantVillage images (Kaggle release emmarex/plantdisease,
or the color images of the PlantVillage GitHub mirror spMohanty/PlantVillage-Dataset, which are byte-identical).

```bash
pip install -r requirements.txt
cd src && mkdir -p ../cache/cache
# 1. handcrafted experts, fixed partition and 30 re-partitions (about 3 min per partition)
python fit.py paper $(seq 0 29)
python analyze.py ../cache/cache ../results           # attributions, controls, manipulations
python summarize.py ../results                         # statistics -> results/results.json
python figures.py ../results
# 2. hyperparameter control (about 7 min per partition)
python tuned.py paper $(seq 0 9) && python analyze.py ../cache/cache_tuned ../results/tuned --tuned
# 3. image-level controls (needs the images once; the derived features are included in results/image_controls)
python image_controls.py --image-root /path/to/PlantVillage
python fit_image.py ../results/image_controls group $(seq 0 9) && python analyze.py ../cache/cache_group ../results/group
python fit_image.py ../results/image_controls background paper $(seq 0 9) && python analyze.py ../cache/cache_leaf ../results/leaf
python fit_image.py ../results/image_controls bgonly paper $(seq 0 9)
python image_summary.py
# 4. deep experts: frozen MobileNetV2 embeddings of three cue-isolating streams (NumPy implementation)
python map_images.py /path/to/PlantVillage-Dataset/raw/color      # writes ../cache/image_map.csv
python embed_streams.py ../cache/image_map.csv /path/to/mobilenetv2_1.0-0c6065bc.pth ../cache/embeddings
python assemble_embeddings.py
python export_deep_posteriors.py --cache ../cache/deep_embeddings --splits 30
python analyze_deep.py ../cache/deep_posteriors ../results/deep && python deep_summary.py
python extras.py
# 5. five-subgroup control (Section 5.8; about 3.5 min per partition, 36 classifier fits each)
mkdir -p ../cache/cache_subgroups
python fit_subgroups.py . ../cache/cache_subgroups $(seq 0 29)
python analyze_subgroups.py . ../cache/cache_subgroups ../results/subgroups
python summarize_subgroups.py ../results/subgroups      # -> results/subgroups/summary.json
# 6. field-image control (Section 5.9): see plantdoc/README.md
```

The MobileNetV2 weights are the publicly released ImageNet weights `pretrained/mobilenetv2_1.0-0c6065bc.pth`
of the GitHub repository d-li14/mobilenetv2.pytorch; `mobilenetv2_numpy.py` reads them without PyTorch.

## Protocol in one paragraph

Each partition is a stratified 70/15/15 split (seeds 0-29 for the re-partitions). The validation split is divided
once, stratified, into V1 (fusion weights and, in the tuning control, hyperparameters) and V2 (temperatures in the
calibration control and reject thresholds). Platt scaling of every SVM uses internal five-fold cross-validation on
the training split. The primary fusion weights are the best of the same 4,000 Dirichlet(1,1,1) candidates (seed 42)
on V1 Macro-F1. All metrics, attributions and reference criteria are computed on the test split.
