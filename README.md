# 🔐 SFAM-ADR: Secure Feature Abstraction Model

![SFAM Logo](https://raw.githubusercontent.com/Lumine8/SFAM/main/SFAM.png)

[![PyPI version](https://img.shields.io/pypi/v/sfam-ADR.svg)](https://pypi.org/project/sfam-ADR/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

**sfam-ADR** is a **neuro-symbolic biometric engine** that secures user identity using **privacy-preserving feature abstraction** instead of raw biometric storage.

It transforms human interaction data (images, mouse gestures, touch patterns) into **irreversible, cancellable biometric hashes**, enabling secure authentication without exposing sensitive user data.

SFAM-ADR combines:

- **GhostNet** for efficient spatial feature abstraction
- **Differential Physics** (Velocity, Acceleration, Jerk) for behavioral dynamics
- **Seed-based secure projection** for cancellable biometric identity

---

## 🚀 What's New in v1.1.0?

We have introduced **Feature Managers** to handle raw data inputs directly:

- **`image_fm`**: Automatically converts images to spatial tensors.
- **`gesture_fm`**: Converts raw coordinate lists (x, y, time) into physics-based behavioral tensors.
- **Simplified Imports**: Access everything directly from the `sfam` namespace.

---

## 📦 Installation

Install directly from PyPI:

```bash
pip install sfam-ADR

```

> **Note:** Requires **PyTorch 2.0+**

---

## 🛠 Usage

### 1️⃣ Import & Initialize

You can now import the engine and feature managers directly.

Python

```
import torch
from sfam import SFAM, image_fm, gesture_fm

# Initialize the Engine
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SFAM(
    behavioral_dim=64,  # Matches the output of gesture_fm
    secure_dim=256      # Size of the final hash
).to(device).eval()

print(f"🚀 SFAM Engine loaded on {device}")

```

### 2️⃣ Process Raw Inputs (New in v1.1.0)

Instead of manually creating tensors, use the built-in processors.

#### **A. Process Images (Spatial Features)**

Python

```
# Automatically resizes, normalizes, and batches the image
spatial_features = image_fm.processor.process("user_face.jpg").to(device)

```

#### **B. Process Gestures (Behavioral Features)**

Pass a list of dictionaries containing `x`, `y`, and `t` (timestamp).

Python

```
# Example raw mouse/touch data
raw_data = [
    {'x': 100, 'y': 200, 't': 0.0},
    {'x': 105, 'y': 202, 't': 0.02},
    {'x': 112, 'y': 208, 't': 0.04},
    # ... more points ...
]

# Calculates Velocity, Acceleration, and Jerk automatically
behavioral_features = gesture_fm.processor.process(raw_data).to(device)

```

### 3️⃣ Generate the Secure Hash

Pass the processed features and a user-specific seed to generate the cancellable identity.

Python

```
# User-specific secret seed (can be rotated if compromised)
user_seed = 12345

with torch.no_grad():
    secure_hash = model(
        spatial_features,
        behavioral_features,
        user_seed
    )

print(f"🔒 Secure Hash Generated: {secure_hash.shape}")
# Output: torch.Size([1, 256])

```

**Only the `secure_hash` is stored. The raw image and gesture data are discarded.**

---

## 🧠 Core Concepts

### 🔐 Secure Feature Abstraction

Raw input data is never stored. Instead, SFAM-ADR produces **non-invertible abstract representations** that preserve discriminative power without revealing the original signal.

### 🔄 Cancellable Biometrics

Each user identity is generated using a **seed-based projection**. Rotating the seed instantly invalidates old biometric hashes, bringing password-like revocability to biometrics.

### ⚡ Differential Physics

For behavioral data, `sfam-ADR` uses differential physics to analyze **how** a user moves, not just **where**.

- **Velocity:** Speed of movement.
- **Acceleration:** Force and intent.
- **Jerk:** Smoothness and neuromuscular control.

---

## 🧪 What SFAM-ADR Is (and Is Not)

| Aspect              | Description                                   |
| :------------------ | :-------------------------------------------- |
| **Learning Type**   | Feature abstraction / representation learning |
| **Classification**  | ❌ Not a classifier (No Softmax)              |
| **Reconstruction**  | ❌ Not possible (Irreversible)                |
| **Labels Required** | ❌ No (Unsupervised-ready)                    |
| **Output**          | Secure, irreversible biometric hash           |
| **Revocability**    | ✅ Yes (Seed rotation)                        |

---

## 🌍 Use Cases

- **Gesture-based Authentication:** Verify users by how they move their mouse or swipe their phone.
- **Privacy-First Identity:** Authenticate without storing faces or fingerprints.
- **Continuous Authentication:** Verifying user identity dynamically during a session.
- **Edge AI:** Lightweight architecture (GhostNet) runs efficiently on IoT devices.

---

## 📄 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

---

## 🔗 Links

- **Source Code:** [https://github.com/Lumine8/SFAM](https://github.com/Lumine8/SFAM)
- **Bug Reports:** [https://github.com/Lumine8/SFAM/issues](https://github.com/Lumine8/SFAM/issues)
