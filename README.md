# safety-critic

Reference implementation of the context-conditioned safety critic for
diffusion-based visual navigation: a diffusion generator proposes `K` candidate
trajectories from RGB-D observations, and a learnable critic with a
trajectory-dependent clearance budget selects among them. ESDF is used only
offline to supervise the teacher; the deployed selector runs from RGB-D alone,
with no map building.

## Status

The components are unit- and integration-tested on synthetic scenes (see
`tests/`), and both training stages run end to end. Training on real collected
data requires the extra dependencies below.

## Method

The critic scores a trajectory `tau` with waypoints `p_0 ... p_T`:

```
V_phi(tau) = V_safe(tau) + V_efficient(tau) + V_balance(tau)

  d_min,j    = d_safe + softplus(q_eta(f_j))
  h(p)       = d(p) - d_safe
  r_cbf_j    = (1 - rho) * h(p_j) - h(p_{j+1})
  L_cbf      = sum_j max(r_cbf_j, 0)
  V_safe     = -sum_j 1[d_j < d_min,j] - lambda_cbf * L_cbf

  L_detour   = max(L_path / (D_chord + eps) - 1, 0)
  w_j        = sigmoid(kappa * (d_j - d_min,j))
  V_efficient= -beta * sum_j ||Delta^2 pi(p_j)|| - mu * mean_j(w_j) * L_detour

  V_balance  = -psi * sum_j (d_j - d_min,j)^2

  w = [beta, lambda_cbf, mu, psi] = softplus(w_tilde)     # nonnegative by construction
  C_phi(tau) = -V_phi(tau) >= 0
  D_phi(tau) = sigmoid(-a * C_phi(tau) + b),  a = softplus(a_hat) > 0
```

| Term | Implementation |
|---|---|
| Margin head `q_eta` | `models/scorer.py` &rarr; `MarginHead` |
| CBF residual, `rho = 0.1` | `models/scorer.py` &rarr; `_cbf_loss` |
| Safety-gated detour ratio | `models/scorer.py` &rarr; `_gated_detour` |
| Softplus-reparameterized weights | `models/scorer.py` &rarr; `weights` |
| Discriminator `D_phi` | `models/scorer.py` &rarr; `discriminator_logit` |
| Adversarial teacher loss `L_scr` | `engine/train_teacher.py` |
| Selector loss `L_sel` | `engine/train_student.py` |
| A\* non-expert proposal `q(tau | p_R, p_G, ESDF)` | `data/negatives.py` |

The margin head is fed a geometry-only context feature — local clearance,
finite-difference ESDF gradient magnitude, and normalized step index — so the
budget adapts to the scene rather than to the time step alone.

### Two implementation notes

- **Unsafe count.** `V_safe` contains `sum_j 1[d_j < d_min,j]`, whose gradient
  w.r.t. `d_min` is zero almost everywhere. Training uses a sigmoid surrogate
  (`soft_unsafe=True`, active in `train()` mode only) so `q_eta` receives
  gradient; `eval()` scores with the exact hard count.
- **Margin initialization.** With a default-initialized output layer,
  `softplus(0) ~= 0.69` m — larger than the clearance available in ordinary
  indoor corridors, which would make every waypoint unsafe from step 0. The
  output bias is instead set so the budget starts at `d_safe + init_margin`
  (default 0.25 m total).

## Install

```bash
pip install -r requirements.txt
```

Two dependencies are not on PyPI and must be installed separately:

- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) —
  importable as `depth_anything.depth_anything_v2.dpt`. Needed by
  `models/backbone.py` only; the critic itself imports without it.
- [habitat-sim](https://github.com/facebookresearch/habitat-sim) — needed by
  `scripts/build_esdf.py` only, for navmesh queries.

## Data

Training data is collected with a companion Habitat-based tool — **coming
soon**. It samples random start–goal pairs with automatic per-floor detection,
plans over the habitat-sim navmesh, smooths the path with a cubic spline and an
obstacle-clearance pass, and records RGB-D along it with randomized camera
height and pitch.

`NavDPDataset` expects one folder per scene, holding:

| File | Contents |
|---|---|
| `rgb.pkl` | `dict[step_key] -> uint8 [H, W, 3]` |
| `depth.pkl` | `dict[step_key] -> float32 [H, W]`, metres |
| `traj.pkl` | `dict[step_key] -> float32 [7]`: `[x, y, z, yaw, pitch, roll, camera_height]` |
| `episode_info.pkl` | list of per-episode dicts |

`step_key` is `ep{episode:03d}_step{step:02d}`; columns 0, 2, 3 of `traj.pkl`
are read as the planar pose `(x, z, yaw)`. Any collector producing this layout
works — point `--data_root` at the directory holding the scene folders.

## Usage

```bash
# 1. Build the scene ESDF (offline, teacher supervision only)
python scripts/build_esdf.py --scene <scene.glb> --out scene_esdf.npz

# 2. Train: diffusion generator -> teacher critic -> student selector
python scripts/train.py --esdf_npz scene_esdf.npz --data_root <dir> --num_scenes 5

# Stages are selectable; the student stage requires a trained teacher.
python scripts/train.py --esdf_npz scene_esdf.npz --stages teacher
python scripts/train.py --esdf_npz scene_esdf.npz --stages student
```

Key options: `--d_safe` (physical floor, default 0.1 m), `--rho` (CBF
conservativeness, 0.1), `--kappa` (safety-gate sharpness, 10.0),
`--k_candidates` (candidates per observation in `D_mix`, 16).

## Tests

```bash
python -m pytest tests/ -v
```

Covers the ESDF query, each critic term against hand-computed values, gradient
flow to `q_eta` and `w_tilde`, the A\* proposal's start/goal matching and
`d_safe` rejection, the RGB-D tensor-layout dispatch, and both
training stages end to end.

## Layout

```
navdp_safety/
  data/     dataset, ESDF build/query, A* non-expert proposals
  models/   RGB-D backbone, diffusion policy, safety critic
  engine/   train_diffusion / train_teacher / train_student, evaluation
scripts/    build_esdf.py, train.py
tests/      synthetic-scene unit and integration tests
```

## License

[Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)
(CC BY-NC 4.0) — see [LICENSE](LICENSE). Free to share and adapt with
attribution, for non-commercial purposes.
