# Multimodal Click-Through Rate Prediction (MM-CTR)
**CSC 5376** · Manal Mehdaoui · Ghita Sbai · Younes Barich

## Overview
Implementation of a Multimodal CTR prediction system for the [WWW 2025 EReL@MIR MM-CTR Challenge](https://erel-mir.github.io/challenge/mmctr-track2/) using the MicroLens-1M dataset.

**Leaderboard AUC: 0.9276** (Task 1&2 Combined) · Evaluated on 379,142 test samples

## Architecture
- **CrossModalGate Fusion** — InfoNCE contrastive loss on 5M co-click pairs, BERT (768-dim) + CLIP (512-dim) → 128-dim L2-normalized embeddings
- **SeqTransformerDCN** — Last-K (k=16) sequence modeling + max pooling + DCNv2 (3 cross layers) + DNN 1024→512→256→64→32→σ

## Our Results

| Task | AUC | Description |
|------|-----|-------------|
| Task 1 | 0.9208 | Our trained CrossModalGate embeddings · cold-start |
| Task 2 | 0.9276 | Official PCA embeddings `item_emb_d128_v3` · warm eval |
| **Task 1&2** | **0.9276** | **Combined · primary leaderboard metric** |

- Inference: **0.072 ms/sample** on NVIDIA A100 · full 379K test set in 27s
- Local evaluation using official Codabench scoring script matched leaderboard exactly

## WWW 2025 Leaderboard Context (Task 1&2 Combined)

| Rank | Team | AUC |
|------|------|-----|
| #1 | tolive666 | 0.9892 |
| #2 | jiyuqian | 0.9887 |
| #3 | QVQ | 0.9885 |
| — | **Ours** | **0.9276** |
| — | Challenge Average | ~0.89 |

> Our submission was made after the official challenge deadline for class/research purposes. Top positions reflect teams competing during the live phase with full compute resources and backbone fine-tuning.

## Files
- `MMCTR_final.ipynb` — Full pipeline with training, evaluation, inference timing, and all output results

## Dataset
- **MicroLens-1M** · 1M+ interactions · 247K items · 15.3K users · 379K test samples
- Item features: BERT text embeddings (768-dim) + CLIP image embeddings (512-dim) + tags

## Challenge
- Platform: [Codabench](https://www.codabench.org/competitions/5372/)
- Workshop: [WWW 2025 EReL@MIR](https://erel-mir.github.io/challenge/mmctr-track2/)
