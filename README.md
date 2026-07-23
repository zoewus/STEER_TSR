# Repository Overview

This repository contains four folders, each with its own README documenting the original codebase and how replica exchange is implemented:

- **Toy** — a mixture of overlapping Gaussians, with options for 1 to 3 Gaussians
- **Image** — runs SD3
- **Pointmaze**
- **Robomimic**

There is also an `acceptance.py` file, shared across all datasets.

## Downloading Model Checkpoints

### Toy
Checkpoints are located in `Toy/model_checkpoints/`, labeled `single`, `barrier`, and `composed`. No changes are needed — running the notebook will pull directly from this directory.

### Image
Stable Diffusion 3 loads from a pretrained model via `StableDiffusion3Pipeline.from_pretrained`. Set `CACHE_DIR` (specified in the notebooks) to download it to netscratch.

### Pointmaze
Download and extract the pretrained models from [this link](https://drive.google.com/file/d/1ZMhoOkLLMozUdADKph3oed_OwzAYtiD1/view?usp=sharing), then place them in the `logs/` directory.

### Robomimic
Pretrained model will be uploaded to Drive.

## Algorithms

### Replica Exchange Sampling

**Input:** Temperature schedule λ, number of replicas `n_replicas`

1. Sample $x_T \sim \mathcal{N}(0, \lambda)$
2. Initialize `temp_idx` for each replica, `temp_idx = 0, ..., n_replicas - 1`
3. For $t = T, \dots, 0$:
   1. Sample $z \sim \mathcal{N}(0, 1)$
   2. If `lambda_start != 1` and `lambda_end != 1`:
      1. Compute $\epsilon(x_t, t)$
      2. `temp_idx = swap(epsilon, temp_idx)`
      3. `epsilon = scale(epsilon, temp_idx)`
   3. Perform DDPM step
4. **Return** sample corresponding to each `temp_idx`

---

### Swap Algorithm

**Input:** Scores at adjacent temperatures/timesteps ($s$, $\tau$)

1. Compute temperature-scaling ratio difference:

   $$\text{tsrDiff} = \text{scoreConstant}(\lambda_{\tau}) - \text{scoreConstant}(\lambda_{s})$$

2. Compute integral approximation:

   $$\text{integral} = -0.5 \cdot (\text{score}_{s} + \text{score}_{\tau}) \cdot (x_{\tau} - x_{s})$$

3. Compute acceptance ratio:

   $$a = \exp(\text{integral} \cdot \text{tsrDiff})$$

4. Accept swap if:

   $$\text{Unif}(0, 1) < a$$
---

> **Note:** Any replica exchange code is marked with `# replica exchange` to distinguish it from the baseline repo.