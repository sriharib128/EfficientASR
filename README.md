# PARSpeech - ASR Training and Inference Pipeline

## 📌 Overview
PARSPeech, an open-source model that leverages unlabeled data to enhance ASR performance for Persian, Arabic, and Urdu. We develop a scalable pipeline to collect, process, and filter unlabelled speech, resulting in a 3,000-hour multilingual corpus. Using this dataset, we pre-train a multilingual acoustic model following a continuous pretraining approach to leverage its existing knowledge. Given the complex orthography of Perso-Arabic languages, we further propose the use of sentencepiece based tokenization for vocabulary construction. Fine-tuning this pre-trained model even on limited labelled data yields performance comparable to state-of-the-art (SOTA) large models(>1B) while using only a small model(300M). This highlights the efficiency of our approach in achieving SOTA with significantly lower computational requirements

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

## 👥 Authors

The development of PARSpeech was led by:

1) Srihari Bandarupalli - IIITH -  📧 Email: srihari.bandarupalli@research.iiit.ac.in 

2) Bhavana Akkiraju - IIITH - 📧 Email: bhavana.akkiraju@research.iiit.ac.in


## ⚖️ License

PARSpeech is licensed under the MIT License - see the LICENSE file for details. The pretrained and fine-tuned models in this repository are licensed under the same terms as the code.


