# Legacy scripts: fine-tuned reference network and latency (Section 5.1, Online Resource 1 Section S7)

These two scripts and their outputs were produced for an earlier draft, in a separate environment
(Python 3.9.12, PyTorch 2.1.0, torchvision, OpenCV 4.8.1, scikit-image 0.19.2, scikit-learn 1.6.1), and are
included unchanged so that the numbers quoted in Section 5.1 can be traced. Section and table numbers in
their comments refer to that earlier draft, not to the present manuscript.

- `run_deep_baseline.py`: MobileNetV2 (torchvision ImageNet weights) fine-tuned for ten epochs on the fixed
  partition, no augmentation; paired exact McNemar test against the all-feature baseline; CPU latency of the
  forward pass alone on random inputs (50 repetitions, image decoding and preprocessing excluded).
- `run_timing.py`: end-to-end latency of the handcrafted pipeline (decoding, feature extraction, classifier)
  over 200 test images, three passes. It contains the feature extractors used for `data/features`.
- `results/`: `table7_deep_baseline.csv`, `deep_baseline_history.csv`, `deep_vs_svm_mcnemar.csv`,
  `timing_handcrafted.csv`.

Both scripts import `common.py` from `../src` and need the PlantVillage images:

```bash
pip install torch torchvision pillow scikit-image
PYTHONPATH=../src python run_deep_baseline.py --data-root /path/to/PlantVillage
PYTHONPATH=../src python run_timing.py --data-root /path/to/PlantVillage --check
```
