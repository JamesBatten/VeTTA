# VeTTA: Vessel Tree Transformer Autoencoder

[![Paper](https://img.shields.io/badge/paper-arXiv:2506.11163-b31b1b.svg)](https://arxiv.org/abs/2506.11163)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **VeTTA** (Vector **T**ree **A**utoencoder) is a two‑stage Transformer framework for learning compact, expressive representations of **3‑D vascular trees** such as coronary arteries. It encodes the *continuous* geometry of each vessel **segment** and the *discrete* **topology** of the full tree into a single latent vector, and it can recursively decode that vector back into a valid tree.

The code in this repository is the **official PyTorch implementation** accompanying our paper presented at **MIDL 2025**:

> **Vector Representations of Vessel Trees**
> James Batten, Michiel Schaap, Matthew Sinclair, Ying Bai, Ben Glocker
> Medical Imaging with Deep Learning (MIDL), 2025 — **Oral**

---

## Repository Structure

All core code lives in the **`vetta/`** package. The high‑level organisation is shown below.

```text
vetta/
├── model/
│   ├── vetta.py                # Vessel Tree Autoencoder
│   ├── vessel_edges_encoder.py # Transformer encoder for edges
│   ├── mlp2.py                 # Simple 2‑layer MLP
│   ├── norm.py                 # Normalisation factory
│   ├── nonlinearity.py         # Activation factory
│   └── weight_init.py          # Weight‑initialisation helpers
├── common/
│   └── utils.py                # Fourier features, I/O helpers, …
└── …
```

### Main Model — `Vetta`

**File:** `vetta/model/vetta.py`

The `Vetta` class implements the **Vessel Tree Autoencoder** described in the paper (Fig. 1, §3).

* **Encoder branch** — encodes the whole tree into a latent vector *z* using a `VesselEdgesEncoder`.
  *Optionally* runs in **VAE** mode to produce $\mu\_z,\;\log\sigma^2\_z$.
* **Decoder branch** — **recursively** reconstructs a tree conditioned on *partial* reconstructions:

  1. Encode the partial tree with **another** `VesselEdgesEncoder`.
  2. Concatenate global latent $z$ to each partial‑edge feature → **memory** for a Transformer **decoder**.
  3. Attend with a small set of learnable **slot queries** → slot embeddings.
  4. Feed each slot through lightweight **MLP heads** to predict child *position*, *radius*, *topology*, … then cluster.

### Core Encoder — `VesselEdgesEncoder`

**File:** `vetta/model/vessel_edges_encoder.py`

* Assemble edge features from the two incident nodes (position, radius, topology).
* Lift 3‑D coordinates with **high‑frequency Fourier features** (cf. §3.1).
* Process the sequence with a `nn.TransformerEncoder`.
* Includes a learnable **start token** so the model gracefully handles empty / initial trees.

### Utilities & Building Blocks

| File                                                 | Purpose                                                                |
| ---------------------------------------------------- | ---------------------------------------------------------------------- |
| `vetta/common/utils.py`                              | Fourier feature lifting (`add_octaves`, `lift`), tensor/array helpers. |
| `vetta/model/mlp2.py`                                | Reusable 2‑layer MLP (`MLP2`).                                         |
| `vetta/model/norm.py`, `vetta/model/nonlinearity.py` | Factory fns for norms & activations (e.g. GroupNorm, GELU).            |
| `vetta/model/weight_init.py`                         | Consistent weight initialisation (e.g. Xavier) across modules.         |

---

## Citation

If you find **VeTTA** useful, please cite:

```bibtex
@inproceedings{batten2025_vector,
  author    = {James Batten and Michiel Schaap and Matthew Sinclair
               and Ying Bai and Ben Glocker},
  title     = {{Vector Representations of Vessel Trees}},
  booktitle = {Proceedings of the 8th Medical Imaging with Deep Learning (MIDL)},
  year      = {2025},
  note      = {Oral presentation},
  url       = {https://openreview.net/forum?id=ESzOwfBhRv}
}
```

---

## License

This project is distributed under the terms of the **MIT License**. See the [LICENSE](LICENSE) file for details.
