# Gomoku RL

A Deep Q-Network (DQN) agent that learns to play 9×9 Gomoku (Five-in-a-Row) through a three-stage curriculum: random opponents, a hand-crafted heuristic, and finally self-play against a pool of its own frozen checkpoints.

[![Gomoku RL video](https://img.youtube.com/vi/lpM9TbTGkxs/maxresdefault.jpg)](https://youtu.be/lpM9TbTGkxs)

▶ **[Watch the project video](https://youtu.be/lpM9TbTGkxs)**

## Highlights

- **Residual CNN Q-network** — a convolutional stem plus 4 residual blocks (64 channels) over a 3-plane board encoding (own stones, opponent stones, last move), outputting a Q-value for each of the 81 squares.
- **Curriculum training** — 1,000 episodes vs. a random opponent, 1,000 vs. a threat-aware heuristic, then 4,000 episodes of self-play.
- **Self-play opponent pool** — every 500 self-play episodes the agent is frozen into a checkpoint. New games are played against the newest checkpoint 80% of the time and a random older one 20% of the time, which prevents catastrophic forgetting.
- **Reward shaping** — small intermediate rewards for building open twos/threes/fours and for blocking opponent threats, on top of the terminal win/loss/draw signal (see `src/rewards.py` and the weights in `src/config.py`).
- **8-fold symmetry augmentation** — every transition is stored under all rotations and reflections of the board, multiplying the effective experience per game.

## Setup

Requires [conda](https://docs.conda.io/). Pick the environment that matches your hardware:

```bash
# CPU
conda env create -f gomoku-cpu.yml -y
conda activate gomoku-cpu

# NVIDIA GPU (CUDA)
conda env create -f gomoku-gpu.yml -y
conda activate gomoku-gpu
```

## Play against the AI

A pygame GUI lets you play against any saved checkpoint:

```bash
python play.py 9                 # play as black vs. checkpoint_9.pth (strongest)
python play.py 0 --color white   # play as white vs. checkpoint_0.pth (weakest)
```

Checkpoints live in `models/` and are numbered by training progress — `checkpoint_0` has only seen random play, `checkpoint_1` finished the heuristic stage, and `checkpoint_2`–`checkpoint_9` come from successive rounds of self-play. Playing against increasing indices is a nice way to feel the agent getting stronger.

## Train

```bash
python train.py
```

Training runs the full curriculum (~6,000 episodes), logs progress every 100 episodes, saves checkpoints to `models/`, and writes CSV logs (per-episode game stats and training metrics) via `src/logger.py`. All hyperparameters — episode counts, epsilon schedule, learning rate, reward-shaping weights — are in `src/config.py`.

## Evaluate

```bash
python evaluate.py
```

Plays 1,000 greedy (ε = 0) games against a random opponent, alternating colors, and reports the win rate. Edit the checkpoint path at the bottom of `evaluate.py` to evaluate a different model.

## How training works

1. **Stage 1 — Random (episodes 0–999):** the agent learns basic mechanics (legal moves, that five-in-a-row wins) against uniformly random play.
2. **Stage 2 — Heuristic (episodes 1,000–1,999):** the opponent (`src/heuristic.py`) scans for threats and plays wins/blocks, forcing the agent to learn defense.
3. **Stage 3 — Self-play (episodes 2,000+):** the agent plays against frozen copies of itself from the checkpoint pool. Pool opponents act with a small forced exploration rate (ε = 0.02) so games don't become deterministic repeats.

Each episode the agent's color is randomized, transitions are augmented with all 8 board symmetries, and the network takes 4 gradient steps on batches sampled from a 50k-transition replay buffer. A target network (updated every 2,000 steps) stabilizes the Q-learning targets, and ε decays from 1.0 to 0.01 with a 0.995 per-episode decay.

## Project structure

```
├── train.py            # Curriculum training loop
├── play.py             # Pygame GUI to play vs. a checkpoint
├── evaluate.py         # Win-rate evaluation vs. a random opponent
├── test_random.py      # Sanity check: environment with random play
├── src/
│   ├── game.py         # Gomoku rules, board state, network encoding
│   ├── network.py      # Residual Q-network + DQNAgent (replay, target net, ε-greedy)
│   ├── config.py       # All hyperparameters and reward weights
│   ├── rewards.py      # Shaped rewards for threats and blocks
│   ├── heuristic.py    # Threat-scanning opponent for stage 2
│   ├── symmetry.py     # 8-fold rotation/reflection augmentation
│   ├── buffer.py       # Replay buffer
│   ├── renderer.py     # Pygame board renderer
│   └── logger.py       # CSV logging
├── models/             # Trained checkpoints (checkpoint_0 … checkpoint_9)
└── history/            # Configs, logs, and checkpoints from earlier training runs
```

## Development

Code is formatted with [black](https://github.com/psf/black) (line length 100):

```bash
pip install black
black .
```
