# PARSpeech 

## 📌 Overview

Automatic Speech Recognition (ASR) for Perso-Arabic languages is challenging due to limited labeled data and complex orthography. While existing state-of-the-art(SOTA) models achieve impressive results, they are computationally intensive and require extensive labeled datasets, limiting their applicability to low-resource languages. To address these challenges, we present PARSpeech, an efficient approach that uses a scalable pipeline to collect unlabeled data, creating a 3,000-hour multilingual corpus. Our methodology combines continuous pretraining with SentencePiece-based tokenization tailored for Perso-Arabic scripts. Despite using only 300M parameters—a mere 20\% of the size of current SOTA models (1.5B+ parameters)—PARSpeech achieves competitive performance while requiring significantly less labeled data for fine-tuning. These results demonstrate the effectiveness of targeted pretraining for developing efficient, high-performance ASR for low-resource languages.

## 🏗 Folder Structure
```
📂 PARSpeech
 ├── run.sh                 # Main script to run the pipeline
 ├── utils
 │   ├── config
 │   │   └── config.yaml      # Configuration file for training
 │   ├── data_generation
 │   │   ├── data_prep.sh        # Script for data preparation
 │   │   ├── generate_audio_report.py
 │   │   ├── generate_dict_analysis.py
 │   │   ├── manifest.py
 │   │   ├── sp_dict_gen.py
 │   │   ├── sp_labels.py
 │   ├── inference
 │   │   ├── components.py
 │   │   ├── infer.py
 │   │   ├── save_predicted_output.py
 │   │   ├── wer_wav2vec.py
 ├── start_finetuning.sh    # Script to start fine-tuning
 ├── infer.sh               # Script to run inference
```

## 🚀 How to Use

### 1️⃣ Run the Script
The entire pipeline is executed using `run.sh`. You need to specify the **stage number** and **language code**:
```bash
bash run.sh <stage> <language_code>
```
Example:
```bash
bash run.sh 1 ur  # Prepare training data for Urdu
bash run.sh 2 ar  # Fine-tune model for Arabic
bash run.sh 3 fa  # Prepare test data for Farsi
bash run.sh 4 ur  # Run inference for Urdu
```

### 2️⃣ Stages in the Pipeline
| Stage | Description |
|-------|------------|
| 1 | **Prepare Training Data**: Generates manifests, labels, and dictionaries for training |
| 2 | **Fine-tune the Model**: Starts fine-tuning using wav2vec pretrained models |
| 3 | **Prepare Testing Data**: Creates manifests and dictionaries for test data |
| 4 | **Run Inference**: Performs ASR inference and calculates WER (Word Error Rate) |

## 🛠 Dependencies
Refer to the [Installation Guide](Installation.md)

## 🔧 Configuration
Modify `config.yaml` inside `utils/config` to change training parameters.

## 📝 Script Descriptions
### **run.sh** (Main script)
- Calls appropriate stage functions
- Manages dataset and model paths

### **data_prep.sh** (Data preparation script)
- Generates dataset manifests and labels
- Splits validation data
- Creates dictionaries

### **start_finetuning.sh** (Fine-tuning script)
- Loads pretrained wav2vec model
- Logs training process
- Saves fine-tuned checkpoints

### **infer.sh** (Inference script)
- Loads fine-tuned model
- Runs inference on test dataset
- Computes WER (Word Error Rate)

## 📂 Output Structure
```
📂 output
 ├── checkpoints/        # Model checkpoints
 ├── results_ar/         # Arabic ASR results
 ├── results_ur/         # Urdu ASR results
 ├── results_fa/         # Farsi ASR results
 ├── tensorboard/        # Logs for visualization
 ├── logs/               # Training and inference logs
```

## 📊 Evaluation
After inference, WER results are stored in:
```
output/results_<lang>/sentence_wise_wer.csv
```


## 📥 Download Models
Download the models from Google Drive:

🔗 [Model Checkpoints (Google Drive)](https://drive.google.com/file/d/1448YTjUV_adVcM8O0cFCb3pypj_SpXpA/view?usp=drive_link)

## ✨ Upcoming Add-ons

1) Pretraining Config
2) Unlabelled Pipeline

## 👥 Authors

The development of PARSpeech was led by:

1) Srihari Bandarupalli - IIITH -  📧 Email: srihari.bandarupalli@research.iiit.ac.in 

2) Bhavana Akkiraju - IIITH - 📧 Email: bhavana.akkiraju@research.iiit.ac.in


## ⚖️ License

PARSpeech is licensed under the MIT License - see the LICENSE file for details. The pretrained and fine-tuned models in this repository are licensed under the same terms as the code.


