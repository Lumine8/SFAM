
![SFAM Logo](https://raw.githubusercontent.com/Lumine8/SFAM/main/SFAM.png)

# 🔐 SFAM-ADR: Secure Feature Abstraction Model

[![PyPI version](https://badge.fury.io/py/sfam-ADR.svg)](https://badge.fury.io/py/sfam-ADR)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

**sfam-ADR** is a **neuro-symbolic biometric engine** that secures user identity using **privacy-preserving feature abstraction** instead of raw biometric storage.

It transforms human interaction data (such as mouse gestures, touch patterns, or behavioral dynamics) into **irreversible, cancellable biometric hashes**, enabling secure authentication without exposing sensitive user data.

SFAM-ADR combines:

- **GhostNet** for efficient spatial feature abstraction
- **Differential physics–based modeling** for behavioral dynamics
- **Seed-based secure projection** for cancellable biometric identity

---

## 🚀 Why SFAM-ADR?

Traditional authentication systems rely on:

- passwords that can be stolen,
- biometrics that cannot be revoked,
- or embeddings that can be reconstructed.

SFAM-ADR introduces a new paradigm where **identity is verified through abstracted behavior**, not stored secrets.

> If the database is compromised, **nothing meaningful can be reconstructed** — and identities can be safely reissued.

---

## 🧠 Core Concepts

### 🔐 Secure Feature Abstraction

Raw input data is never stored. Instead, SFAM-ADR produces **non-invertible abstract representations** that preserve discriminative power without revealing the original signal.

### 🔄 Cancellable Biometrics

Each user identity is generated using a **seed-based projection**. Rotating the seed instantly invalidates old biometric hashes, bringing password-like revocability to biometrics.

### 🧩 Neuro-Symbolic Design

The system combines:

- **Neural networks** (for perception and representation learning)
- **Symbolic control** (via deterministic seed-driven transformations)

This hybrid approach improves robustness, explainability, and security.

---

## 📦 Installation

Install directly from PyPI:

```bash
pip install sfam-ADR
```

> **Note:** Requires **PyTorch 2.0+**

---

## 🛠 Usage

### 1️⃣ Initialize the Engine

Initialize the SFAM engine by specifying the behavioral feature dimension and secure embedding size.

```
import torch from sfam.models.sfam_net import SFAM

# Automatically selects GPU if available
device = "cuda"  if torch.cuda.is_available() else  "cpu"

model = SFAM(
    behavioral_dim=6, # velocity, acceleration, jerk, etc. secure_dim=256 ).to(device).eval()

print(f"🚀 SFAM Engine loaded on {device}")`
```

---

### 2️⃣ Generate a Secure Hash

Pass spatial and behavioral features along with a user-specific seed.

```
# Dummy example input
# Spatial: [Batch, 128]
# Behavioral: [Batch, 64]
spatial_features = torch.randn(1, 128).to(device)
behavioral_features = torch.randn(1, 64).to(device)

# Cancellable user seed
user_seed = 12345  with torch.no_grad():
    secure_hash = model(
        spatial_features,
        behavioral_features,
        user_seed
    )

print(f"🔒 Secure Hash Generated: {secure_hash.shape}") # Output: torch.Size([1, 256])`
```

Only the **secure hash** is stored — never the raw input.

---

## 🌟 Key Features

- **Cancellable Biometrics**  
  Rotate the `user_seed` to revoke compromised identities instantly.
- **Privacy-Preserving by Design**  
  No raw biometric data or reconstructable embeddings are stored.
- **Modality Agnostic**  
  Adaptable to gestures, touch, keystrokes, voice, or other behavioral signals.
- **Lightweight & Edge-Friendly**  
  Built on GhostNet for low-latency, resource-efficient deployment.
- **Unsupervised-Ready**  
  Does not require labeled biometric datasets.

---

## 🧪 What SFAM-ADR Is (and Is Not)

| Aspect          | Description                                   |
| --------------- | --------------------------------------------- |
| Learning Type   | Feature abstraction / representation learning |
| Classification  | ❌ Not a classifier                           |
| Reconstruction  | ❌ Not possible                               |
| Labels Required | ❌ No                                         |
| Output          | Secure, irreversible biometric hash           |
| Revocability    | ✅ Yes (seed rotation)                        |

---

## 🔐 Security Model Overview

SFAM-ADR is designed to resist:

- template inversion attacks
- replay and imitation attacks
- database compromise scenarios
- model-weight leakage attacks

Even with full access to:

- stored hashes
- model architecture

**Original biometric inputs cannot be reconstructed.**

---

## 🌍 Use Cases

- Gesture-based authentication systems
- Passwordless mobile login
- Behavioral biometrics for fintech
- Privacy-first identity systems
- Edge-device authentication (IoT, embedded systems)

---

## 👥 Contributors

- **Lumine8** — Core architecture, modeling, implementation
- **miss_anonymous** — Conceptual support, validation, documentation

---

## 📄 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

---

## 🔗 Links

- **Source Code:** [https://github.com/Lumine8/SFAM](https://github.com/Lumine8/SFAM)
- **Bug Reports:** [https://github.com/Lumine8/SFAM/issues](https://github.com/Lumine8/SFAM/issues)

## ⭐ Citation

If you use SFAM-ADR in academic or research work, please cite the project or link back to this repository.
