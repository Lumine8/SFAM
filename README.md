# 🔐 SFAM: Secure Feature Abstraction Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

**SFAM** is a modality-agnostic, privacy-preserving feature abstraction framework. It is designed to transform raw multimodal data (images, time-series, audio) into **irreversible, cancellable biometric hashes**.

This repository contains the core SFAM architecture and a reference implementation applied to **Behavioral Biometrics** (codenamed *SecuADR*), which secures user identity using dynamic mouse/touch physics.

---

## 🏗️ The SFAM Architecture

SFAM is not just a classifier; it is a security layer that sits between raw data and authentication logic.

### 1. Modality-Agnostic Encoders
SFAM uses swappable backbones depending on the input data:
* **Spatial Path:** Uses **GhostNet** (Lightweight CNN) for visual patterns.
* **Temporal/Physics Path:** Uses **Differential MLP** for time-series dynamics (velocity, acceleration, jitter).

### 2. Irreversible Abstraction (BioHashing)
Instead of storing user feature vectors (which can be stolen and reversed), SFAM projects features into a high-dimensional orthogonal space using a user-specific **Seed Key**.
* **Result:** A binary hash (e.g., `10110...`) that is mathematically impossible to reverse into the original face/gesture.

### 3. Cancellable Biometrics
If a database is breached, the user's raw biometrics are safe. The system simply:
1.  **Revokes** the old Seed Key.
2.  **Issues** a new Seed Key.
3.  **Regenerates** a completely new hash from the *same* physical biometric.

---

## 📦 Installation

Clone the repository and install the package:

```bash
git clone [https://github.com/Lumine8/SFAM.git](https://github.com/Lumine8/SFAM.git)
cd SFAM
pip install -e .
