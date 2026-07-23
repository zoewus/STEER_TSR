## PointMaze Planning

The code is based on [diffuser](https://github.com/jannerm/diffuser) and the maze environment is based on [ogbench](https://github.com/seohongpark/ogbench). 
### Install
First install the dependencies in `requirements.txt`, then install ogbench with our added PointMaze Ultra environment. 
```bash
conda create -n maze python=3.10
pip install -r requirements.txt
cd ogbench
pip install -e .
cd ..
pip install -e .
```
Then, download and extract the pretrained models from [here](https://drive.google.com/file/d/1ZMhoOkLLMozUdADKph3oed_OwzAYtiD1/view?usp=sharing) and put them in the `logs/` directory. For the collected trajectory dataset for the Ultra Maze, download and extract the files from [here](https://drive.google.com/file/d/1doAvARCm04axeXFUn8ETRgWfcoMt84m3/view?usp=sharing), and put the files `pointmaze-ultra-navigate-v0.npz` and `pointmaze-ultra-navigate-v0-val.npz` in the directory `~/.ogbench/data`. 

### Inference
For inference scaling, run 
```bash
python run.py --dataset pointmaze-giant-navigate-v0 --method dfs --device cuda
```
You can change the method in `dfs`, `bfs-resampling`, `bfs-pruning` and `bon`, and change the dataset in `pointmaze-giant-navigate-v0` and `pointmaze-ultra-navigate-v0`. 


---

## Replica Exchange / Swap Algorithm — Notes

**Code layout** (a bit spread out, so worth noting where things live):
- `search_policy.py` — high-level code that iterates through timesteps
- `methods/tfg.py` — runs sampling at a single timestep; this is where the replica exchange logic lives
- `methods/base_guidance.py` — subfunctions used during sampling, including the DDPM step

### Swap Parameters

| Parameter | Description |
|---|---|
| `replica_exchange` | Enables replica exchange when set to `True` |
| `lam_start` | Starting lambda value for temporal score rescaling |
| `lam_end` | Ending lambda value for temporal score rescaling |
| `n_particles` | Number of samples/replicas generated (e.g., in best-of-N sampling, this is N) |

**Sampling modes:**
- **Untempered sampling** — set `lam_start = lam_end = 1.0`
- **Temporal score rescaling** — set `lam_start` and `lam_end` to any values ≠ 1.0
- **Replica exchange** — set `replica_exchange = True` (typically combined with rescaling)

### Example

```bash
srun python run.py \
  --dataset pointmaze-giant-navigate-v0 \
  --lam_start 0.9 \
  --lam_end 1.0 \
  --replica_exchange \
  --n_particles 8 \
  --device cuda
```

This runs replica exchange with 8 particles.
