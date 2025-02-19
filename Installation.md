# PARSpeech Installation Guide

## 📌 Prerequisites

### Install Miniconda
Before installing or activating any new environment, **always deactivate the base Conda environment** to avoid inheriting unnecessary environment variables.

**Fairseq Commit Reference:** `ecbf110e1eb43861214b05fa001eff584954f65a`

---

## 🚀 Step 1: Create and Activate Conda Environment
```bash
conda create --prefix ./env_fair -c anaconda python=3.8 nvidia/label/cuda-12.1.1::cuda-toolkit cudnn=9.1
conda activate ./env_fair
```

---

## ⚙️ Step 2: Install Fairseq
```bash
git clone https://github.com/facebookresearch/fairseq
cd fairseq
pip install --editable ./
```
---

## 📦 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### `requirements.txt`:
```txt
packaging
soundfile
swifter
joblib==1.4.2
indic-nlp-library
tqdm==4.67.1
numpy==1.24.4
pandas==1.2.2
progressbar2==3.53.1
python_Levenshtein==0.12.2
editdistance==0.3.1
omegaconf==2.0.6  # Do not change this version
tensorboard
tensorboardX
wandb
jiwer
jupyterlab
```

---

## 🔥 Step 4: Install Flashlight for KenLM Decoding

### **Flashlight Text**
```bash
pip install flashlight-text
pip install git+https://github.com/kpu/kenlm.git
```

### **Flashlight Sequence**
For **CUDA support**, set `USE_CUDA=1` when installing. Disable OpenMP by setting `USE_OPENMP=0`.

```bash
git clone https://github.com/flashlight/sequence
cd sequence
pip install .
```

---

### 🔹 Important Note for Inference  

1️⃣ While running inference, ensure that `infer.py` can access Fairseq by adding the following line at the beginning of `infer.py`:  

```python
import sys
sys.path.append("path/to/your/fairseq")  # Replace with the actual path to your Fairseq directory
```

This is required to correctly import Fairseq modules while performing inference.

✅ **Installation Complete!** 🚀
