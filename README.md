# 🛡️ SecuADR: Secure Adaptive Dynamic Recognition
> **A Neuro-Symbolic Behavioral Biometric Engine powered by SFAM.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

**SecuADR** is a next-generation authentication framework that replaces static passwords with **dynamic behavioral biometrics**. Unlike traditional pattern locks, SecuADR analyzes the *physics* of your movement—velocity, acceleration, and tortuosity—to create a unique, unforgeable digital signature.

Under the hood, it uses **SFAM (Secure Feature Abstraction Model)** to convert these biometric inputs into irreversible, privacy-preserving hashes.

---

## 🚀 Key Features

### 🧠 Neuro-Symbolic Architecture
* **Spatial Path:** Uses **GhostNet (1.0x)** to analyze the geometric shape of the drawing.
* **Behavioral Path:** Uses a custom **Physics MLP** to analyze temporal dynamics (speed, jitter, rhythm).

### 🔒 Privacy-First (Cancellable Biometrics)
* **Irreversible Abstraction:** Your raw biometric data is never stored.
* **Revocability:** If your hash is stolen, we simply rotate the security seed. The old hash becomes useless, and a new one is generated from the *same* hand gesture.

### 🛡️ Anti-Spoofing
* **Log-Space Velocity Analysis:** Distinguishes between human micro-movements and robotic/replay attacks.
* **Tortuosity Checks:** Ensures the complexity of the path matches human efficiency.

---

## 📦 Installation

Clone the repository and install the package in editable mode:

```bash
git clone [https://github.com/Lumine8/SFAM.git](https://github.com/Lumine8/SFAM.git)
cd SFAM
pip install -e .
