# 🔐 SFAM-ADR: Secure Feature Abstraction Model

![SFAM Logo](https://raw.githubusercontent.com/Lumine8/SFAM/main/SFAM.png)

[![PyPI version](https://img.shields.io/pypi/v/sfam-ADR.svg)](https://pypi.org/project/sfam-ADR/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

**sfam-ADR** is a **neuro-symbolic biometric engine** that secures user identity using **privacy-preserving feature abstraction** instead of raw biometric storage.

It transforms human interaction data (images, mouse gestures, touch patterns) into **irreversible, cancellable biometric hashes**, enabling secure authentication without exposing sensitive user data.

SFAM-ADR combines:

- **GhostNet & Attention** for adaptive spatial feature abstraction.
- **Differential Physics** (Velocity, Acceleration, Jerk) for behavioral dynamics.
- **Cancelable Biometrics** via seed-based orthogonal projection.

---

## 🚀 What's New in v1.2.2?

- **Adaptive Fusion:** New `SFAM_Adaptive` model that dynamically weights Face vs. Behavior inputs based on signal quality (Attention Gating).
- **Modular Architecture:** Separated `ImageEncoder` and `TemporalEncoder` for cleaner integration.
- **Simplified Imports:** Access everything directly from the `sfam` namespace.

---

## 📦 Installation

Install directly from PyPI:

```bash
pip install sfam-ADR

```

> **Note:** Requires **PyTorch 2.0+** and **timm**

---

## 🛠 Usage

### 1️⃣ Import & Initialize

You can now import the Adaptive Engine and feature managers directly.

```
import torch
from sfam import SFAM_Adaptive, image_fm, gesture_fm

# Initialize the Adaptive Engine
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SFAM_Adaptive(
    behavioral_dim=64,  # Matches the output of gesture_fm
    secure_dim=256      # Size of the final hash
).to(device).eval()

print(f"🚀 SFAM Adaptive Engine loaded on {device}")

```

### 2️⃣ Process Raw Inputs

Instead of manually creating tensors, use the built-in processors.

#### **A. Process Images (Spatial Features)**

Python

```
# Automatically resizes, normalizes, and batches the image
# Returns tensor shape: (1, 3, 224, 224)
spatial_input = image_fm.processor.process("user_face.jpg").to(device)

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
# Returns tensor shape: (1, Seq_Len, 6)
behavioral_input = gesture_fm.processor.process(raw_data).to(device)

```

### 3️⃣ Generate the Secure Hash

Pass the processed inputs and a user-specific key matrix to generate the cancellable identity.

```
from sfam.models.sfam_net import generate_user_key

# 1. Generate a Revocable Key (Salt + User ID)
# If this key is lost/stolen, you just change the salt!
user_key = generate_user_key("user_123", "salt_v1", dim=256).to(device)

# 2. Run Inference
with torch.no_grad():
    # The model fuses the inputs and projects them onto the user key
    secure_hash = model(
        spatial_input,
        behavioral_input,
        user_key,       # The secret projection matrix
        binarize=True   # Returns +1/-1 for Hamming Distance
    )

print(f"🔒 Secure Hash Generated: {secure_hash.shape}")
# Output: torch.Size([1, 256])

```

**Only the `secure_hash` is stored. The raw image and gesture data are discarded immediately.**

---

## 🧠 Core Concepts

### 🔐 Secure Feature Abstraction

Raw input data is never stored. Instead, SFAM-ADR produces **non-invertible abstract representations** that preserve discriminative power without revealing the original signal.

### 🔄 Cancellable Biometrics

Each user identity is generated using a **seed-based projection** (BioHashing). Rotating the seed instantly invalidates old biometric hashes, bringing password-like revocability to biometrics.

### ⚡ Adaptive Fusion

Unlike static fusion models, `SFAM_Adaptive` uses an **Attention Mechanism** to decide which modality is more trustworthy in real-time.

- _Camera blocked?_ The model relies more on **Motion**.
- _Hand jittery?_ The model relies more on **Face**.

---

## 🧪 What SFAM-ADR Is (and Is Not)

| Aspect             | Description                                   |
| :----------------- | :-------------------------------------------- |
| **Learning Type**  | Feature abstraction / representation learning |
| **Classification** | ❌ Not a classifier (No Softmax / Classes)    |
| **Reconstruction** | ❌ Not possible (Irreversible)                |
| **Fusion Type**    | ✅ Adaptive (Attention-based weighting)       |
| **Output**         | Secure, irreversible biometric hash           |
| **Revocability**   | ✅ Yes (Key/Salt rotation)                    |

---

## 🌍 Use Cases

- **Gesture-based Authentication:** Verify users by how they move their mouse or swipe their phone.
- **Privacy-First Identity:** Authenticate without storing faces or fingerprints.
- **Continuous Authentication:** Verifying user identity dynamically during a session.
- **Edge AI:** Lightweight architecture (GhostNet) runs efficiently on IoT devices.

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
