<h1 align="center">
QNN From Scratch
</h1>
<h3 align="center">
A NumPy-only implementation of a data-reuploading Quantum Neural Network (QNN) for binary classification.
</h3>

---

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Test Linux](https://github.com/neuralsorcerer/qnn-from-scratch/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/neuralsorcerer/qnn-from-scratch/actions/workflows/ubuntu.yml?query=branch%3Amain)
[![Test Windows](https://github.com/neuralsorcerer/qnn-from-scratch/actions/workflows/windows.yml/badge.svg)](https://github.com/neuralsorcerer/qnn-from-scratch/actions/workflows/windows.yml?query=branch%3Amain)
[![Test MacOS](https://github.com/neuralsorcerer/qnn-from-scratch/actions/workflows/macos.yml/badge.svg)](https://github.com/neuralsorcerer/qnn-from-scratch/actions/workflows/macos.yml?query=branch%3Amain)
[![Lints](https://github.com/neuralsorcerer/qnn-from-scratch/actions/workflows/lint.yml/badge.svg)](https://github.com/neuralsorcerer/qnn-from-scratch/actions/workflows/lint.yml?query=branch%3Amain)
[![License](https://img.shields.io/badge/License-MIT-3c60b1.svg?logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

> [!NOTE]
> This repository is for learning and experimentation only. It is not intended for production decisions or claims of quantum advantage.

---

## Why this project exists

Most QNN tutorials hide critical internals behind framework abstractions. This repository is built to show every moving part and the reason for each design choice:

1. **Statevector simulation** in $\mathbb{C}^{2^n}$ so you can inspect exact amplitudes (no sampling noise).
2. **Explicit unitaries** so gate-level math maps directly to code.
3. **Variational ansatz** so trainable parameters are physically interpretable as rotation angles.
4. **Expectation measurement** so model outputs come from quantum observables.
5. **Cross-entropy loss** so training aligns with probabilistic binary classification.
6. **Parameter-shift gradients** because conventional backprop through quantum gates is not directly available in this setup.
7. **Adam optimization** for stable first-order updates on noisy/non-convex loss surfaces.

---

## What we use and why

- **NumPy**: deterministic linear algebra and transparent tensor operations.
- **Statevector model**: exact expectations $\langle Z \rangle$ without shot variance, ideal for pedagogy.
- **$R_X, R_Y, R_Z$ gates**: minimal universal-style rotation primitives for expressive single-qubit transformations.
- **CNOT ring entanglement**: low-complexity pattern that still couples qubit subspaces.
- **Data re-uploading**: repeatedly injects classical features, increasing representational capacity with shallow qubit counts.
- **Binary cross-entropy (BCE)**: principled objective for Bernoulli likelihood.
- **Parameter-shift rule**: exact gradient estimator for Pauli-generated rotations.
- **Adam**: adaptive per-parameter step sizes to improve convergence behavior.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Run tests:

```bash
python -m pytest -q
```

Train:

```bash
qnn train --config configs/default.json
```

Predict:

```bash
qnn predict --params outputs/default_run/trained_params.npy --x0 0.2 --x1 0.8 --num-qubits 2 --num-layers 2
```

---

## CLI usage

### `qnn train`

```bash
qnn train \
  --config configs/default.json \
  --output-dir outputs/experiment_a \
  --epochs 50 \
  --learning-rate 0.08 \
  --no-plots
```

- `--config`: base experiment configuration.
- `--output-dir`: artifact destination override.
- `--epochs`: optimization horizon override.
- `--learning-rate`: Adam step size override.
- `--no-plots`: skips plotting for faster runs.

### `qnn predict`

```bash
qnn predict \
  --params outputs/default_run/trained_params.npy \
  --x0 0.2 \
  --x1 0.8 \
  --num-qubits 2 \
  --num-layers 2
```

`--num-qubits` and `--num-layers` must match the training architecture used to produce the parameter file.

---

## Configuration reference

| Key | Type | Why it matters |
|---|---|---|
| `seed` | `int` | Controls deterministic RNG across dataset/model/training shuffle. |
| `num_samples` | `int` | Trades off generalization estimate quality vs runtime. |
| `test_size` | `float` | Sets holdout ratio, requires $0<test\_{size}<1$. |
| `dataset` | `str` | Selects decision-boundary family (`linear`, `vertical`, `horizontal`, `circle`, `sine`). |
| `noise` | `float` | Injects label ambiguity to control task difficulty. |
| `num_qubits` | `int` | Sets Hilbert size $2^n$ (cost grows exponentially in $n$). |
| `num_layers` | `int` | Increases capacity and parameter count $(L,n,3)$. |
| `epochs` | `int` | More optimization steps; higher runtime. |
| `learning_rate` | `float` | Directly scales Adam update magnitude. |
| `batch_size` | `int` | `0` means full-batch; mini-batch may reduce per-step cost. |
| `output_dir` | `str` | Governs reproducible artifact placement. |
| `log_every` | `int` | Controls monitoring cadence. |

---

## Artifacts and outputs

```text
outputs/default_run/
├── config.json
├── trained_params.npy
├── metrics.csv
├── test_predictions.csv
├── summary.json
├── loss_curve.png
├── accuracy_curve.png
└── decision_boundary.png
```

- `config.json`: resolved training config for experiment provenance.
- `trained_params.npy`: learned parameter tensor $\theta\in\mathbb{R}^{L\times n\times 3}$.
- `metrics.csv`: epoch snapshots of train/test loss + accuracy + gradient norm.
- `test_predictions.csv`: per-sample labels, predictions, probabilities.
- `summary.json`: final metrics + confusion matrix.

---

## Implementations

### 1) State model

For $n$ qubits, the model state is:

$|\psi\rangle = \sum_{k=0}^{2^n-1} a_k |k\rangle, \quad a_k\in\mathbb{C}, \quad \sum_k |a_k|^2=1$.

### Why this choice

Statevectors provide exact amplitudes and exact expectation values, ideal for understanding algorithm behavior without shot noise confounds.

### 2) Rotation gates

$$ R_X(\theta) = e^{-i\theta X/2} = \begin{bmatrix} \cos\left(\frac{\theta}{2}\right) & -i\sin\left(\frac{\theta}{2}\right) \\ -i\sin\left(\frac{\theta}{2}\right) & \cos\left(\frac{\theta}{2}\right) \end{bmatrix} $$

$$ R_Y(\theta) = e^{-i\theta Y/2} = \begin{bmatrix} \cos\left(\frac{\theta}{2}\right) & -\sin\left(\frac{\theta}{2}\right) \\ \sin\left(\frac{\theta}{2}\right) & \cos\left(\frac{\theta}{2}\right) \end{bmatrix} $$

$$ R_Z(\theta) = e^{-i\theta Z/2} = \begin{bmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{bmatrix} $$

### Why these gates

They are smooth, differentiable, physically meaningful, and sufficient for expressive parameterized single-qubit transformations.

### 3) Ansatz

For each layer $\ell=1,\dots,L$ and qubit $q=0,\dots,n-1$:

1. Data encoding: $R_X(x_{q\bmod d})$, then $R_Y(x_{(q+1)\bmod d})$.
2. Trainable block: $R_Y(\theta_{\ell,q,0})$, then $R_Z(\theta_{\ell,q,1})$.
3. Entanglement ring: CNOT chain wrapped cyclically.
4. Post-entanglement: $R_Y(\theta_{\ell,q,2})$.

### Why this structure

Data re-uploading increases nonlinear feature interaction depth; entanglement enables cross-qubit correlation; repeated layers expand hypothesis space.

### 4) Measurement and probability

On observable wire $w$:

$z(x;\theta)=\langle\psi(x;\theta)|Z_w|\psi(x;\theta)\rangle\in[-1,1]$

$p(x;\theta)=\frac{1-z(x;\theta)}{2}\in[0,1]$

$\hat y=\mathbf{1}[p\ge0.5]$

### Why this mapping

$\langle Z\rangle$ naturally lies in $[-1,1]$, so affine rescaling yields a valid Bernoulli probability for BCE training.

### 5) Loss

For samples $(x_i,y_i)$, $y_i\in\{0,1\}$:

$\mathcal{L}(\theta)=-\frac{1}{m}\sum_{i=1}^m \left[y_i\log p_i + (1-y_i)\log(1-p_i)\right]$

where $p_i=p(x_i;\theta)$.

### Why BCE

BCE corresponds to negative log-likelihood of Bernoulli outcomes and provides calibrated probabilistic supervision.

### 6) Parameter-shift gradient

For any trainable scalar parameter $\theta_k$:

$\frac{\partial z_i}{\partial\theta_k}=\frac{1}{2}\left[z_i\left(\theta_k+\frac{\pi}{2}\right)-z_i\left(\theta_k-\frac{\pi}{2}\right)\right]$

$\frac{\partial p_i}{\partial\theta_k}=-\frac{1}{2}\frac{\partial z_i}{\partial\theta_k}$

$\frac{\partial\mathcal{L}}{\partial p_i}= -\frac{1}{m}\left(\frac{y_i}{p_i}-\frac{1-y_i}{1-p_i}\right)$

$\frac{\partial\mathcal{L}}{\partial\theta_k}=\sum_{i=1}^{m}\frac{\partial\mathcal{L}}{\partial p_i}\frac{\partial p_i}{\partial\theta_k}$

### Why parameter-shift

It gives an exact analytic gradient for this gate family without finite-difference step-size bias.

### 7) Optimizer

With gradient $g_t=\nabla_\theta \mathcal{L}_t$:

$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$

$v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$

$\hat m_t=\frac{m_t}{1-\beta_1^t},\quad \hat v_t=\frac{v_t}{1-\beta_2^t}$

$\theta_t=\theta_{t-1}-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$

### Why Adam

Adaptive moments reduce sensitivity to raw gradient scale and usually stabilize optimization in non-convex variational landscapes.

---

## Reproducibility

Determinism comes from seeded RNG in dataset generation, split shuffling, parameter initialization, and epoch-level batching order.

---

## Troubleshooting

- **Parameter shape error on `predict`**: architecture flags must match trained tensor shape.
- **Training is slow**: expected with $O(2^n)$ state simulation and repeated parameter-shift passes.
- **Weak accuracy**: reduce dataset noise, increase epochs, tune learning rate, or adjust depth $L$.

---

## License

MIT License. See [LICENSE](./LICENSE).
