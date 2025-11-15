# Animal Face Identification POC

## Overview
- Proof-of-concept pipeline for animal face identification with closed-set baseline, open-set rejection, and future enrollment workflows.
- Designed for local experimentation on a Windows workstation with NVIDIA RTX 5080; code targets PyTorch 2.x.
- Builds on cropped animal face datasets before adding detection/tracking.

## Repository Layout
- `docs/` – context (`context-raw.md`), project plan, plus `notes/` and `research/` placeholders for experiments and literature.
- `src/config/` – YAML-driven config schema (see `configs/train_closed_set.yaml` for defaults).
- `src/datasets/` – Animal face dataset wrapper, transform builder, and dataloader registry.
- `src/models/` – ResNet18 embedding backbone scaffold.
- `src/training/` – Closed-set training loop skeleton (`train.py`) and evaluation placeholder.
- `src/inference/` – k-NN gallery helper plus `predict.py` CLI scaffold.
- `data/` – (gitignored raw/processed, but annotations tracked) location for datasets, embeddings, and unknown pools.
- `configs/` – Run configurations; currently only `train_closed_set.yaml`.
- `validate_dataset.py` – Dataset structure and integrity validation script.
- `DATASET_AUDIT_REPORT.md` – Comprehensive dataset validation report with statistics and recommendations.

## Current Status (Phase 1 Skeleton)
- ResNet18 embedding wrapper implemented with L2-normalized outputs; classifier head defined inside training loop.
- Animal face dataset loader expects `data/processed/animal_faces/<split>/<individual_id>/*.jpg|png` layout.
- Training loop and validation pass are stubbed with TODOs; checkpoints saved to `artifacts/closed_set_skeleton.pt`.
- k-NN inference pipeline loads gallery index (`artifacts/gallery_index.pkl`) built outside this scaffold; prediction CLI prints nearest IDs.
- Config loader now produces a `TrainingConfig` dataclass for reuse across training/inference scripts.

## Getting Started

### Platform-Specific Setup

<details>
<summary><b>🐧 Windows with WSL2 + Ubuntu (Recommended for NVIDIA GPU Training)</b></summary>

If you're on Windows with an NVIDIA GPU (e.g., RTX 5080), using **WSL2 with Ubuntu** provides the best PyTorch + CUDA performance and compatibility.

#### Prerequisites
1. **Install WSL2** (if not already installed):
   ```powershell
   # In PowerShell (Administrator)
   wsl --install
   # Restart your computer after installation
   ```

2. **Verify NVIDIA Driver** (in Windows PowerShell):
   ```powershell
   nvidia-smi
   ```
   You should see your GPU listed (e.g., RTX 5080).

3. **Verify GPU Access in WSL2** (in Ubuntu terminal):
   ```bash
   nvidia-smi
   ```
   If you see the same GPU table, WSL2 can access your GPU ✅

#### Environment Setup in WSL2 Ubuntu

1. **Navigate to Project Directory**:
   ```bash
   # If your repo is at C:\Users\jones\Downloads\animal-face-id
   cd /mnt/c/Users/jones/Downloads/animal-face-id
   ```

2. **Install Python and venv** (if needed):
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip
   python3 --version  # Should be 3.10+
   ```

3. **Create Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   You should see `(.venv)` prefix in your prompt.

4. **Upgrade pip**:
   ```bash
   pip install --upgrade pip
   ```

5. **Install PyTorch with CUDA Support**:
   ```bash
   # For CUDA 12.1 (check https://pytorch.org/get-started/locally/ for latest)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

6. **Install Project Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

7. **Verify GPU Detection**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
   ```
   Expected output:
   ```
   PyTorch: 2.x.x
   CUDA available: True
   GPU: NVIDIA GeForce RTX 5080
   ```

#### Daily Workflow
```bash
# 1. Open WSL2 Ubuntu terminal (in Cursor or Windows Terminal)
# 2. Navigate to project
cd /mnt/c/Users/jones/Downloads/animal-face-id

# 3. Activate venv
source .venv/bin/activate

# 4. Run training
python -m src.training.train --config configs/train_closed_set.yaml
```

</details>

<details>
<summary><b>🪟 Windows (Native, without WSL)</b></summary>

1. **Python Version**: Python 3.11 (recommended) or 3.10+

2. **Create Virtual Environment**:
   ```powershell
   # In PowerShell or CMD
   python -m venv venv
   
   # Activate (PowerShell)
   .\venv\Scripts\Activate.ps1
   # Or CMD
   .\venv\Scripts\activate.bat
   ```

3. **Install PyTorch with CUDA**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify GPU**:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

</details>

<details>
<summary><b>🍎 macOS (Apple Silicon)</b></summary>

1. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install PyTorch with MPS Support**:
   ```bash
   pip install torch torchvision torchaudio
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify MPS**:
   ```bash
   python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
   ```

</details>

<details>
<summary><b>🍎 macOS (Intel) or Linux CPU-only</b></summary>

1. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install PyTorch (CPU)**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

</details>

---

## 📦 Dataset Setup

### Dataset Source

This project uses the **Chimpanzee Faces Dataset** from:
👉 **https://github.com/cvjena/chimpanzee_faces**

The dataset contains cropped face images of individual chimpanzees collected from two field sites:

- **data_CTai** — Taï National Park, Ivory Coast (5,078 images)
- **data_CZoo** — Leipzig Zoo, Germany (2,109 images)

Each individual has a unique ID label, with metadata including annotations, identity mapping, age group, gender, and keypoint information.

> **Note:** This repository does not bundle the full dataset due to size and licensing considerations.
> Please follow the steps below to prepare the dataset locally.

### 📁 Required Folder Structure

After downloading the dataset from the [chimpanzee_faces repository](https://github.com/cvjena/chimpanzee_faces), organize it as follows:

```
animal-face-id/
├── data/
│   ├── chimpanzee_faces/
│   │   ├── raw/
│   │   │   └── datasets_cropped_chimpanzee_faces/
│   │   │       ├── data_CTai/
│   │   │       │   └── face_images/          # 5,078 PNG images
│   │   │       └── data_CZoo/
│   │   │           └── face_images/          # 2,109 PNG images
│   │   ├── annotations/
│   │   │   ├── annotations_merged_all.txt    # All 102 individuals (7,187 images)
│   │   │   ├── annotations_merged_min10.txt  # 87 individuals with ≥10 images (7,150 images)
│   │   │   └── kept_ids_min10.txt            # List of valid IDs for training
│   │   └── processed/
│   │       └── (empty - ready for train/val/test splits)
```

**Key annotation files:**
- `annotations_merged_all.txt` — Complete dataset with all individuals (102 IDs, 7,187 images)
- `annotations_merged_min10.txt` — Filtered dataset with individuals having ≥10 images (87 IDs, 7,150 images) — **recommended for training**
- `kept_ids_min10.txt` — List of 87 individual IDs meeting the 10-image threshold

### 🔍 Dataset Validation

Before training, validate that your dataset is properly organized:

```bash
python validate_dataset.py
```

**This script checks:**
- ✅ All annotation paths reference existing images
- ✅ Folder structure matches the expected layout
- ✅ All individuals in min10 subset have ≥10 images
- ✅ ID consistency between annotation files and kept_ids list
- ✅ No missing or corrupted files

**Expected output:**
```
✓✓✓ Dataset structure verified — ready for model training.
```

If validation passes, you're ready to proceed. Detailed validation results are saved to:
- `validation_results.json` — Machine-readable validation metrics
- `DATASET_AUDIT_REPORT.md` — Comprehensive human-readable audit report

### 📊 Dataset Statistics

| Dataset | Images | Individuals | CTai | CZoo |
|---------|--------|-------------|------|------|
| **All** | 7,187 | 102 | 5,078 (70.6%) | 2,109 (29.4%) |
| **Min10** ⭐ | 7,150 | 87 | 5,041 (70.5%) | 2,109 (29.5%) |

**Recommendation:** Use `annotations_merged_min10.txt` for training to ensure each class has sufficient samples (≥10 images per individual).

### 📖 Dataset License

Please refer to the original dataset's license:
👉 **https://github.com/cvjena/chimpanzee_faces**

All training in this project is based on that dataset. We do not redistribute the original images in this repository.

---

## Configuration

- Copy `configs/train_closed_set.yaml` and adjust paths, batch sizes, embedding dimensions, etc., as needed.
- Update `data.num_classes` to `87` when using the `annotations_merged_min10.txt` dataset (recommended).

## Usage
1. **Training Skeleton**
   ```bash
   python -m src.training.train --config configs/train_closed_set.yaml
   ```
   - Script currently iterates over loaders without real forward/backward logic. Fill in TODO sections before running actual training.
2. **Building the Gallery Index**
   - After training, write a small helper (future task) that exports embeddings for gallery splits and saves `EmbeddingIndex` via `src/inference/knn.py`.
   - Expected output: `artifacts/gallery_index.pkl`.
3. **Prediction CLI**
   ```bash
    python -m src.inference.predict --image path/to/cropped_face.jpg --config configs/train_closed_set.yaml
   ```
   - Requires existing gallery index; raises if missing.

## GroVE Integration Outlook
- **What it is**: cvjena/GroVE converts deterministic vision-language embeddings into probabilistic embeddings (mean + covariance), providing uncertainty estimates for each sample.
- **Why it matters**: uncertainty-aware distances (e.g., KL divergence) dramatically improve open-set rejection and prevent low-quality crops from spawning new IDs during enrollment.
- **When to use**:
  1. *Phase 1–2*: optional; baseline can stay deterministic.
  2. *Phase 3*: highly valuable—filter unknown pools by uncertainty before clustering.
  3. *Future wildlife deployments*: near-essential for noisy side-profile footage.
- **Integration plan**: keep the existing ResNet/ViT backbone, add a GroVE head to output μ/Σ, and bolt on KL-divergence scoring + uncertainty gating without discarding current weights.

## Recommended Next Steps
1. Flesh out `train_closed_set` forward/backward pass, metric logging, and checkpointing.
2. Add an embedding export script to populate `artifacts/gallery_index.pkl`.
3. Extend evaluation to compute Top-1/Top-5 accuracy and confusion matrices.
4. Once closed-set baseline stabilizes, start Phase 2 work (open-set threshold calibration, unknown buffering).
5. Prototype GroVE probabilistic head as a drop-in module to compare deterministic vs probabilistic open-set metrics.
