#!/bin/bash

set -e  # Exit script on any error

LOG_FILE="error.log"

# Function to handle errors
error_exit() {
    echo "❌ Error: $1" | tee -a "$LOG_FILE"
    exit 1
}
export CUDA_VISIBLE_DEVICES=0,1
# -----------------------------------------------
# How to Run This Script:
# -----------------------------------------------
# Run the script by passing a stage number and a language code:
# Example:
#   bash run.sh 1 ur  # Run data preparation for Urdu
#   bash run.sh 2 ar  # Start fine-tuning for Arabic
#   bash run.sh 3 fa  # Prepare testing data for Farsi
#   bash run.sh 4 ur  # Run inference for Urdu
# -----------------------------------------------

# Validate input arguments
if [ "$#" -ne 2 ]; then
    error_exit "Usage: $0 <stage> <language_code>\nValid language codes: ar, ur, fa"
fi

stage=$1  # Pass the stage as a command-line argument
lang=$2   # Pass the language as a command-line argument

# Validate language selection
if [[ "$lang" != "ar" && "$lang" != "ur" && "$lang" != "fa" ]]; then
    error_exit "Invalid language code! Choose from: ar, ur, fa"
fi

# -----------------------------------------------
# Define Folder Paths
# -----------------------------------------------
DATA_DIR=$(pwd)
echo "Current Directory: $current_dir"

model_name="cp1"
gpus=2
INPUT_FOLDER="$DATA_DIR/input"
OUTPUT_FOLDER="$DATA_DIR/output"
AUDIO_FOLDER="$INPUT_FOLDER/audio"

DATA_PATH="$OUTPUT_FOLDER/${lang}_data"
SPM_MODEL="$INPUT_FOLDER/spm_models"

SPM_PATH="$INPUT_FOLDER/spm_models/$spm_$lang"

# Training Data Paths
TRAIN_WAV_PATH="$AUDIO_FOLDER/train"

TRAIN_DEST_PATH="$OUTPPUT_FOLDER/finetuning_$lang"

# Fine-tuning Paths
FINETUNING_DICT="$TRAIN_DEST_PATH/dict.ltr.txt"

# Testing Data Paths
TEST_WAV_FOLDER="$AUDIO_FOLDER/test"
TEST_OUT_PATH="$OUTPUT_FOLDER/infer_valid_$lang"

# Inference Paths
INFER_DEST_FOLDER="$OUTPUT_FOLDER/infer_$lang"
PRETRAINED_MODEL_PATH="$INPUT_FOLDER/checkpoints/xlsr2_300m.pt"

# Model checkpoint for fine-tuned model
CHECKPOINT_PATH="$OUTPUT_FOLDER/checkpoints/${lang}/checkpoint_best.pt"

# Output path for inference results
RESULT_PATH="$OUTPUT_FOLDER/results_${lang}/test"

# -----------------------------------------------
# Check Required Folders and Files
# -----------------------------------------------

# Check if input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Creating output folder: $INPUT_FOLDER"
    mkdir -p "$INPUT_FOLDER" || error_exit "Failed to create output folder!"
fi

if [ ! -d "$SPM_MODEL" ]; then
    echo "Creating output folder: $SPM_MODEL"
    mkdir -p "$SPM_MODEL" || error_exit "Failed to create output folder!"
fi

# Check if audio folder exists
[ ! -d "$AUDIO_FOLDER" ] && error_exit "Audio folder '$AUDIO_FOLDER' does not exist!"

# Check if train & test audio folders exist
[ ! -d "$TRAIN_WAV_PATH" ] && error_exit "Training audio folder '$TRAIN_WAV_PATH' does not exist!"
[ ! -d "$TEST_WAV_FOLDER" ] && error_exit "Testing audio folder '$TEST_WAV_FOLDER' does not exist!"


# Create output folder if it doesn't exist
if [ ! -d "$OUTPUT_FOLDER" ]; then
    echo "Creating output folder: $OUTPUT_FOLDER"
    mkdir -p "$OUTPUT_FOLDER" || error_exit "Failed to create output folder!"
fi

# -----------------------------------------------
# Functions for Different Stages
# -----------------------------------------------

# Stage 1: Prepare Training Data
prepare_train_data() {
    echo "📌 Stage 1: Preparing Training Data for $lang..."

    local valid_wav_path=""
    local make_valid_from_train=1
    local valid_percentage=0.15  # 15% validation data

    # Check if the script exists before execution
    [ ! -f "./utils/data_generation/data_prep.sh" ] && error_exit "Missing script: data_prep.sh"

    # Calls a script to prepare training data
    bash ./utils/data_generation/data_prep.sh \
        "train" "$TRAIN_WAV_PATH" "$valid_wav_path" "$make_valid_from_train" \
        "$valid_percentage" "$TRAIN_DEST_PATH" "$SPM_PATH" || error_exit "Data preparation failed!"
}

# Stage 2: Fine-tune the Model
start_finetuning() {
    echo "📌 Stage 2: Starting Fine-tuning for $lang..."

    local config_name="PARSpeech_config.yaml"
    local gpus=2
    local wandb_name="finetuning_${lang}_${model_name}_test"
    local config_path="${DATA_DIR}/utils/config"
    local checkpoints_path="$OUTPUT_FOLDER/checkpoints_${wandb_name}"
    local log_path="$OUTPUT_FOLDER/logs/${wandb_name}"
    local tensorboard_path="$OUTPUT_FOLDER/tensorboard"
    local data="$TRAIN_DEST_PATH"

    # Check if the script exists before execution
    [ ! -f "utils/start_finetuning.sh" ] && error_exit "Missing script: start_finetuning.sh"

    # Calls fine-tuning script
    bash utils/start_finetuning.sh \
        "$CUDA_VISIBLE_DEVICES" "$PRETRAINED_MODEL_PATH" "$config_name" \
        "$config_path" "$wandb_name" "$data" "$checkpoints_path" \
        "$tensorboard_path" "$log_path" "$gpus"|| error_exit "Fine-tuning failed!"
}

# Stage 3: Prepare Test Data
prepare_test_data() {
    echo "📌 Stage 3: Preparing Testing Data for $lang..."

    # Check if the script exists before execution
    [ ! -f "./utils/data_generation/data_prep.sh" ] && error_exit "Missing script: data_prep.sh"

    # Calls a script to prepare testing data
    bash utils/data_generation/data_prep.sh \
        "test" "$TEST_WAV_FOLDER" "$TEST_OUT_PATH" "$SPM_PATH" "$FINETUNING_DICT" || error_exit "Test data preparation failed!"
}

# Stage 4: Run Inference on Test Data
run_inference() {
    echo "📌 Stage 4: Running Inference for $lang..."
    SUBSET="test"

    # Check if the script exists before execution
    [ ! -f "utils/infer.sh" ] && error_exit "Missing script: infer.sh"

    # Calls the inference script
    bash utils/infer.sh \
        "$DATA_PATH" "$INFER_DEST_FOLDER" \
        "$PRETRAINED_MODEL_PATH" "$CHECKPOINT_PATH" "$RESULT_PATH" "$TEST_OUT_PATH" "$SUBSET" "$SPM_PATH.model" || error_exit "Inference failed!"
}

# -----------------------------------------------
# Run the Requested Stage
# -----------------------------------------------
case "$stage" in
    1) prepare_train_data ;;  # Data Preparation
    2) start_finetuning ;;    # Model Fine-tuning
    3) prepare_test_data ;;   # Test Data Preparation
    4) run_inference ;;       # Inference
    *)
        error_exit "Invalid stage! Please provide a valid stage number:\n  1 - Data Preparation\n  2 - Fine-tuning\n  3 - Testing Data Preparation\n  4 - Inference"
        ;;
esac
