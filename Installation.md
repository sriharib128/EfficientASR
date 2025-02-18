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
pip install --upgrade pip==24.0  # Important for omegaconf==2.0.6
```

---

## 🔧 Step 2: Setup CUDA in Conda Environment
Create activation and deactivation scripts to set CUDA and environment variables:

```bash
#!/bin/bash

# Create directories for activation and deactivation scripts
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# Activation script content
activate_script_content='#!/bin/sh
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export WANDB_API_KEY=$YOUR_API_KEY'

# Deactivation script content
deactivate_script_content='#!/bin/sh
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed -e "s|$CONDA_PREFIX/lib:||g")
unset CUDA_HOME
unset WANDB_API_KEY'

# Save and make the scripts executable
echo "$activate_script_content" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "$deactivate_script_content" > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

echo "CUDA environment setup completed!"
```

---

## ⚙️ Step 3: Install Fairseq
```bash
git clone https://github.com/facebookresearch/fairseq
cd fairseq
pip install --editable ./
pip install protobuf==3.20
```

### 🔧 Modifications in Fairseq
Modify the following lines in Fairseq files:

**1. `fairseq/fairseq/criterions/wav2vec_criterion.py`**  
📌 **At line 231**, modify:
```python
return False  # Modify line 231
```

**2. `fairseq/fairseq/distributed/utils.py`**  
📌 **At line 97** (inside `_infer_slurm_init`), add:
```python
return False  # Modify line 97
```

---

## 📦 Step 4: Install Dependencies

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

## 🔥 Step 5: Install Flashlight for KenLM Decoding

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
