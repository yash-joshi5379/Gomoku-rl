# Deep Reinforcement Learning for 9×9 Gomoku: Design Document

**Goal:** Train a strong 9×9 Gomoku (five-in-a-row) agent with **MCTS** for search, but **lighter than full AlphaZero** (fewer sims, smaller net, optional mixed opponents).

---

## 1. Problem Summary

- **Board:** 9×9.
- **Rules:** Two players alternate; first to get 5 (or more) in a row (horizontal, vertical, diagonal) wins. Draw if board full.
- **State:** 9×9 grid with three values per cell: empty, player 1, player 2.
- **Action:** One empty cell index (0–80) per move.

---

## 2. Network Architecture

### 2.1 Why Convolutional N×N Kernels?

- Gomoku is **spatially local**: threats and wins are short lines (3, 4, 5). Convolutions capture local patterns (lines, forks, blocks) and reuse weights across the board.
- **N×N kernels:** 3×3 is standard and cheap; 5×5 can capture slightly longer lines in one layer. For 9×9, **3×3 is a good default**; stack several layers to get larger receptive fields.

### 2.2 Recommended Architecture (Lightweight)

- **Input:** 9×9 × C channels. Use **C = 3**: channel 0 = own stones, channel 1 = opponent stones, channel 2 = current player (all 1s for “me to move”, or separate for symmetry). Alternatively C = 2 (own / opponent) and pass current player as a global scalar.
- **Body (shared):**
  - Conv2d(3, 64, kernel_size=3, padding=1) → ReLU → BatchNorm  
  - 2–4 more blocks: Conv 3×3 (e.g. 64→128→128) + ReLU + BN. No residual needed at this scale.
- **Heads:**
  - **Policy head:** Conv 1×1 → 81 logits (flatten 9×9), then mask invalid (occupied) and softmax. Outputs probability over actions.
  - **Value head:** Conv 1×1 → 64 → FC → 1, tanh. Outputs scalar in [-1, 1] (expected outcome for current player).

**Justification:** A single shared torso plus policy and value heads keeps the model small (on the order of 100k–500k parameters), trains quickly, and still gives both “where to play” (policy) and “how good is this state” (value) for RL and for optional planning.

---

## 3. Value Function, Policy Function, or Both?

**Recommendation: Use both (value + policy), with a shared torso.**

| Option            | Pros                          | Cons                           |
|-------------------|-------------------------------|---------------------------------|
| **Policy only**   | Simpler, fewer params         | No bootstrap; high variance     |
| **Value only**    | Can do 1-step greedy with value | No direct action distribution; need search or enumeration |
| **Both (recommended)** | Value reduces variance (TD target); policy gives direct control | Slightly more parameters and loss terms |

- **Policy:** Necessary for selecting moves and for policy-gradient updates.
- **Value:** Provides a baseline (reduces variance) and allows TD-style learning (e.g. value loss toward reward or n-step return). You can train with **A2C-style** or **PPO-style** updates without any MCTS.

So: **shared CNN torso → policy head + value head**, used both for direct action probabilities and inside MCTS (see §6).

---

## 4. What to Train Against

Avoid **pure AlphaZero-style self-play** (no MCTS, no massive self-play league) to save compute.

### 4.1 Opponent Mix (Recommended)

Train against a **mixture of opponents**:

1. **Random agent (e.g. 30–40% of games)**  
   - Samples uniformly over legal moves.  
   - **Pros:** Fast, diverse states, avoids collapse to a single strategy.  
   - **Cons:** Weak; alone it may not teach advanced play.

2. **Rule-based / heuristic agent (e.g. 30–40%)**  
   - Simple rules: if can win in one, play it; else if opponent can win in one, block; else if can make open 4, play it; else if can make 3, play it; else random or center-preferring.  
   - **Pros:** Gives meaningful gradient (beating a “block/win” player teaches basic tactics).  
   - **Cons:** Still limited; may have blind spots.

3. **Minimax / negamax with small depth (e.g. 20–30%)**  
   - Depth 2–4, simple evaluation (e.g. count 2s, 3s, 4s, 5s).  
   - **Pros:** Stronger signal, teaches medium-term planning.  
   - **Cons:** Slower; keep depth low to stay cheap.

4. **Past checkpoint of the current policy (optional, 0–20%)**  
   - Opponent = policy network from N updates ago.  
   - **Pros:** Some self-improvement without full self-play.  
   - **Cons:** Can be unstable; use a small fraction and not from the very latest policy.

**With MCTS:** You can either (a) use **self-play with MCTS** for both sides (AlphaZero-style, strongest but costliest), or (b) keep a **mix**: e.g. our agent uses MCTS for its own moves, and the opponent is random / rule-based / past checkpoint for some fraction of games to reduce compute and diversify training.

---

## 5. Reward Design: Shaping vs Terminal Only

### 5.1 Terminal-Only Rewards (Recommended Baseline)

- **Win:** +1 for the winner (current player at terminal state).  
- **Lose:** −1 for the loser.  
- **Draw:** 0 (or small constant, e.g. 0).

**Pros:** Simple, no bias, aligns with true objective.  
**Cons:** Sparse; many moves get 0 before the end, so credit assignment is harder.

### 5.2 Optional Reward Shaping

If learning is too slow, add **small intermediate rewards** (keep them much smaller than ±1):

- **Subgoals (example):**  
  - Small positive bonus for creating an open 4 (e.g. +0.1).  
  - Small positive for creating a double threat (e.g. +0.05).  
  - Small negative for allowing opponent open 4 (e.g. −0.1).

**Pros:** Denser signal, can speed up learning.  
**Cons:** Risk of reward hacking (e.g. agent chases bonuses instead of winning). Prefer **terminal-only first**; add shaping only if needed and tune magnitudes carefully.

**Recommendation:** Start with **terminal rewards only** (win/loss/draw). Introduce shaping only if progress is too slow and then keep shaping terms small and sparse.

---

## 6. MCTS (Monte Carlo Tree Search)

MCTS is used to **select moves** during play and to **generate training targets** for the policy and value heads. The tree is **neural-network-driven**: priors and leaf evaluation come from the CNN.

### 6.1 Tree Structure

- **Node** = board state (after some sequence of moves). Root = current position.
- **Edge** from state \(s\) = taking action \(a\). Store:
  - \(N(s,a)\) = visit count
  - \(W(s,a)\) = total backpropagated value (for current player at \(s\))
  - \(Q(s,a) = W(s,a)/N(s,a)\)
  - \(P(s,a)\) = prior from **policy head** (masked for legal moves, renormalized).

### 6.2 One Simulation (per move)

1. **Selection:** From root, follow the edge that maximizes **PUCT** until a leaf (unexpanded) node:
   \[
   a^* = \arg\max_a \left[ Q(s,a) + c_{\mathrm{puct}}\, P(s,a)\, \frac{\sqrt{N(s)}}{1 + N(s,a)} \right],
   \]
   where \(N(s) = \sum_a N(s,a)\). Typical \(c_{\mathrm{puct}} \in [1, 2]\) (e.g. 1.5). Ties broken arbitrarily.

2. **Expansion:** If the leaf is non-terminal, expand it: run the **network** on the leaf state to get \((p, v)\). For each legal action \(a\), set \(P(s,a)\) from \(p\) (mask and renormalize), and create a new child node. Store \(v\) for the leaf (used in backup).

3. **Backup:** If the leaf is **terminal** (win/loss/draw), set \(v = +1/-1/0\) for the player who *moved into* that state. Otherwise use the network value \(v\). Backpropagate \(v\) along the path to the root: for each edge \((s,a)\) on the path, add the value (flip sign when alternating turns) to \(W(s,a)\) and increment \(N(s,a)\).

4. **Repeat** for a fixed number of **simulations per move** (e.g. 50–200 for training; 100–400 for evaluation).

### 6.3 Action Selection After MCTS

- **Training (self-play):** Sample move from **visit-count distribution** (softmax over \(N(s,a)\) with temperature \(\tau\)): \(\pi_{\mathrm{MCTS}}(a) \propto N(s,a)^{1/\tau}\). Use \(\tau \approx 1\) early in the game (more exploration), \(\tau \to 0\) near the end (greedy).
- **Evaluation / inference:** Pick \(a^* = \arg\max_a N(s,a)\) (no temperature).

### 6.4 Keeping It Lighter Than AlphaZero

| Levers | AlphaZero-style | Lighter option |
|--------|------------------|----------------|
| **Simulations per move** | 800–1600+ | **50–200** (training), 100–400 (eval) |
| **Network size** | Large residual stack | Small CNN (see §2) |
| **Self-play only?** | Yes | Optional: mix MCTS agent vs random/rule-based (e.g. 50% self-play, 50% vs others) |
| **Reanalyze / replay buffer** | Often used | Omit or small buffer to save compute |

Use **virtual loss** (add a constant to \(W\) for in-progress simulations) so parallel simulations don’t all take the same path.

### 6.5 Integration With the Network

- **Policy head** → \(P(s,a)\) for PUCT and for defining the visit prior at expansion.
- **Value head** → \(v\) at leaf nodes (non-terminal). Terminal nodes use game outcome (+1/−1/0).
- State representation for the network must be **from the perspective of the player to move** (so value is “expected outcome for current player”). When backing up, flip sign for the opponent’s nodes.

---

## 7. Learning Algorithm (With MCTS)

1. **Data generation:** For each game (or batch of games):
   - Start from empty board. For each move:
     - Run **MCTS** from the current state (e.g. 50–200 sims).
     - Store **(state, \(\pi_{\mathrm{MCTS}}\), z)** where \(\pi_{\mathrm{MCTS}}\) = visit-count distribution (target for policy), \(z\) = game outcome (+1/−1/0) for the player who *made that move* (or use MCTS root value as target; outcome is simpler and stable).
     - Play the move by sampling from \(\pi_{\mathrm{MCTS}}\) (with temperature).
   - Opponent can be: same MCTS agent (self-play), or random/rule-based/past checkpoint for a fraction of games.

2. **Training:** Sample a minibatch of \((s, \pi_{\mathrm{MCTS}}, z)\). Loss:
   - **Policy loss:** Cross-entropy between network policy \(p_\theta(\cdot|s)\) and \(\pi_{\mathrm{MCTS}}\) (only over legal actions).
   - **Value loss:** MSE between value head \(v_\theta(s)\) and \(z\).
   - **Total:** \(\mathcal{L} = \mathcal{L}_{\mathrm{policy}} + c_v \mathcal{L}_{\mathrm{value}}\) (e.g. \(c_v = 1\)).

3. **Optimizer:** Adam, LR 1e-4 to 3e-4; train on a replay buffer of recent games (e.g. last 50k–200k positions) with multiple passes or one pass per game.

4. **Evaluation:** Periodically play the current network (with MCTS, e.g. 100–200 sims) vs random / rule-based / previous checkpoint; track win rate.

This is **AlphaZero-style training but scaled down**: MCTS provides the policy target and game outcome the value target; no separate RL value bootstrap needed.

---

## 8. Summary Table

| Dimension              | Choice                                      | Rationale                                      |
|------------------------|---------------------------------------------|------------------------------------------------|
| **Kernel size**        | 3×3 (N×N)                                  | Captures local lines; cheap; stack for reach   |
| **Value / Policy**     | Both, shared torso                          | Policy for PUCT prior & expansion; value for leaf eval |
| **MCTS**               | PUCT, 50–200 sims/move (train), 100–400 (eval) | Stronger play; policy target = visit counts; value target = game outcome |
| **Training opponents** | Self-play with MCTS, or mix vs random/rule-based/past self | Self-play for strength; mix to reduce compute & diversify |
| **Rewards**            | Terminal only (win/loss/draw); optional small shaping | Same as before; MCTS uses outcome for backup & value target |

---

## 9. Suggested Next Steps

1. Implement the 9×9 env (state, legal moves, win/draw detection).  
2. Implement the CNN (input encoding, torso, policy + value heads).  
3. Implement **MCTS**: PUCT selection, expansion with network \((p,v)\), backup (terminal = outcome, else \(v\)), virtual loss for parallelism.  
4. Generate games: MCTS for our agent; opponent = same MCTS (self-play) or random/rule-based for a fraction. Store \((s, \pi_{\mathrm{MCTS}}, z)\).  
5. Train on collected data: policy cross-entropy to \(\pi_{\mathrm{MCTS}}\), value MSE to \(z\).  
6. Evaluate periodically (MCTS agent vs random/rule-based); tune sim count and opponent mix for compute vs strength.
