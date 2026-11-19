# Field-image control (Section 5.9, Table 7)

Deep experts of Section 4.5 on PlantDoc (GitHub repository pratikkayal/PlantDoc-Dataset, training and test
folders pooled). One class with two images is removed; 2,570 images of 27 classes remain (`index.csv`, paths
relative to the repository root). The streams and the NumPy MobileNetV2 are imported unchanged from `../src`.

```bash
cd plantdoc/src
# 1. embeddings (about 75 s per quarter on a laptop; the four slices can run in parallel)
for r in "0 643" "643 1286" "1286 1929" "1929 2570"; do
  python embed.py ../../src ../index.csv /path/to/mobilenetv2_1.0-0c6065bc.pth ../cache/emb $r /path/to/PlantDoc-Dataset
done
# 2. logistic heads for each candidate regularization, 30 partitions
for c in 1 0.1 0.01 0.003 0.001; do HEAD_C=$c python fit_heads.py ../index.csv ../cache/emb ../cache/post_C$c 30; done
# 3. regularization per stream and partition on V1 log-likelihood
python select_heads.py ../index.csv ../cache/post_sel 1=../cache/post_C1 0.1=../cache/post_C0.1 \
  0.01=../cache/post_C0.01 0.003=../cache/post_C0.003 0.001=../cache/post_C0.001
# 4. attributions (selected heads; repeat with ../cache/post_C1 for the default-head columns of Table 7)
python analyze.py ../../src ../index.csv ../cache/post_sel ../results/selected
```

Handcrafted experts, last paragraph of Section 5.9. The 87 features are included in `features/`, so step 1
can be skipped; it needs scikit-image and imports the extractors of `../legacy/run_timing.py` unchanged.

```bash
cd plantdoc/src
# 1. optional: recompute the features from the images
python extract_handcrafted.py ../.. ../index.csv ../features 0 2570 /path/to/PlantDoc-Dataset
# 2. SVM experts of Section 4.2 on the same 30 partitions, then the same analysis
python fit_handcrafted.py ../.. ../index.csv ../features ../cache/post_handcrafted 30
python analyze.py ../../src ../index.csv ../cache/post_handcrafted ../results/handcrafted
```

`build_index.py` rebuilds `index.csv` from the image folders. `results/` holds the per-partition outputs:
`selected` (main result), `C1` ... `C0.001` (fixed regularization, Table 7 PlantVillage heads = `C1`) and
`handcrafted` (SVM experts on the 87 features).
In `main.csv`, `objective` is `macroF1` (weights searched on V1 Macro-F1), `calibrated` (temperatures on V2)
or `flat_<stream>` (that stream flattened to lambda = 0.1); `ok` marks agreement with the single-stream order.
