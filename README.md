# A hybrid deep reinforcement learning approach for MPPT control with variable shading and temperature

This repository is the official implementation of the paper "Combining PPO and incremental conductance for MPPT under dynamic shading and temperature"

## Requirements

To install requirements, run:

```setup
pip install -r requirements.txt
```

## Evaluation


To train and evaluate the hybrid PPO model, run:

```eval
python evaluate_mppt_ppo_hybrid.py
```

## Baseline Models

To evaluate the hill climbing baseline, run this command:

```
python evaluate_mppt_hill_climbing.py
```

To evaluate the adapative incremental conductance baseline, run this command:

```
python evaluate_mppt_incr_cond_Li_Chendi_et_al.py
```

## RL environment

The "mppt_grid_rl" environment is compatible with a gym interface and can be imported directly.


