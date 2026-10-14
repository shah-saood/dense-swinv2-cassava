# Hybrid Dense-SwinV2 for Cassava Leaf Disease Classification

Dual-branch architecture combining DenseNet-201 (local features) and
Swin Transformer V2 (global context), fused via a Multi-Scale Attention
Module with bidirectional feature reprojection.

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 98.0% |
| F1-score | 97.36% |
| ROC-AUC | 0.9893 |
| Inference time | 21.5 ms |
| Dataset | ~31,000 images, 5 classes |
| Benchmark | Compared to 22 SOTA models |

## Architecture

- **Branch 1:** DenseNet-201 → local texture features
- **Branch 2:** Swin Transformer V2 → global context
- **Fusion:** Multi-Scale Attention Module + bidirectional feature reprojection

## Citation
Shah Saood, S. H. Khan, R. Iqbal. "Hybrid Dense-SwinV2: A Novel
Dual-Branch Architecture for Cassava Disease Classification."
Spectrum of Engineering Sciences, Vol. 4, No. 3, pp. 1–15, 2026.

## Contact

Shah Saood — shahsaood814@gmail.com
GitHub: [github.com/shah-saood] (https://github.com/shah-saood)
