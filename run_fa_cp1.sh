#!/bin/bash

set -e  # Exit script if any command fails

stage=$1  # Pass the stage as a command-line argument
lang="fa"

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=2,3

# --- Data Paths ---
DATA_PATH="/data/bhavana/Interspeech/data_fa_cp1_sp"
SPM_PATH="$DATA_PATH/spm_models/$lang/spm_$lang"

# --- Training Paths ---
TRAIN_WAV_PATH="/data/bhavana/labelled_data/Persian/train_16khz"
TRAIN_DEST_PATH="$DATA_PATH/finetuning"

# --- Fine-tuning Paths ---
FINETUNING_DICT="$DATA_PATH/dict.ltr.txt"

# --- Testing Paths ---
TEST_WAV_FOLDER="/data/bhavana/labelled_data/Persian/test_16khz"
TEST_OUT_PATH="$DATA_PATH/infer_valid"

# --- Inference Paths ---
INFER_DEST_FOLDER="${DATA_PATH}/infer/"
PRETRAINED_MODEL_PATH="/data/bhavana/Interspeech/checkpoints/pretrained/xlsr_300_epoch2.pt"

# --- Decoding Parameters ---
W2L_DECODER_VITERBI=1  # 1 for Viterbi, 0 for KenLM
SUBSET="test"

CHECKPOINT_PATH="/data/bhavana/Interspeech/checkpoints/test/checkpoint_best.pt"
RESULT_PATH="${DATA_PATH}/results/${SUBSET}"

# --- Functions ---

prepare_train_data() {
    echo "Stage 1: Preparing Training Data..."
    
    local valid_wav_path=""
    local make_valid_from_train=1
    local valid_percentage=0.15

    bash ./utils/data_generation/data_prep.sh \
        "train" "$TRAIN_WAV_PATH" "$valid_wav_path" "$make_valid_from_train" \
        "$valid_percentage" "$TRAIN_DEST_PATH" "$SPM_PATH"
}

start_finetuning() {
    echo "Stage 2: Starting Fine-tuning..."

    local config_name="PARSpeech_config.yaml"
    local gpus=2
    local wandb_name="finetuning_fa_cp1_sp_dict"
    local config_path="utils/config"
    local checkpoints_path="$DATA_PATH/checkpoints_${wandb_name}"
    local log_path="$DATA_PATH/logs/${wandb_name}"
    local tensorboard_path="${log_path}/tensorboard"
    local data="$DATA_PATH/finetuning"

    bash utils/start_finetuning.sh \
        "$CUDA_VISIBLE_DEVICES" "$PRETRAINED_MODEL_PATH" "$config_name" \
        "$config_path" "$wandb_name" "$data" "$checkpoints_path" \
        "$tensorboard_path" "$log_path"
}

prepare_test_data() {
    echo "Stage 3: Preparing Testing Data..."

    bash utils/data_generation/data_prep.sh \
        "test" "$TEST_WAV_FOLDER" "$TEST_OUT_PATH" "$SPM_PATH" "$FINETUNING_DICT"
}

run_inference() {
    echo "Stage 4: Running Inference..."

    bash utils/infer.sh \
        "$W2L_DECODER_VITERBI" "$DATA_PATH" "$INFER_DEST_FOLDER" \
        "$PRETRAINED_MODEL_PATH" "$CHECKPOINT_PATH" "$RESULT_PATH" "$TEST_OUT_PATH" "$SUBSET"
}

# --- Main Script Execution ---

case "$stage" in
    1) prepare_train_data ;;
    2) start_finetuning ;;
    3) prepare_test_data ;;
    4) run_inference ;;
    *)
        echo "Invalid stage! Please provide:"
        echo "  1 - Data Preparation"
        echo "  2 - Fine-tuning"
        echo "  3 - Testing Data Preparation"
        echo "  4 - Inference"
        exit 1
        ;;
esac
