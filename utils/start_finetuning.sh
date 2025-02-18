#!/bin/bash

# Assign input parameters
CUDA_VISIBLE_DEVICES="$1"
pretrained_model_path="$2"
config_name="$3"
config_path="$4"
wandb_name="$5"
data_path="$6"
checkpoints_path="$7"
tensorboard_path="$8"
log_path="$9"
gpus="${10}"

# Logging script parameters
echo "🎯 GPUs Assigned: '$gpus'"
echo "🚀 Fine-tuning started with:"
echo "🖥️  CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "📁 Pretrained Model Path: $pretrained_model_path"
echo "⚙️  Config Name: $config_name"
echo "📂 Config Path: $config_path"
echo "📝 Data Path: $data_path"
echo "💾 Checkpoints Path: $checkpoints_path"
echo "📊 TensorBoard Path: $tensorboard_path"
echo "📜 Log Path: $log_path"

# Ensure necessary directories exist
mkdir -p "$checkpoints_path" "$tensorboard_path" "$log_path"

# Function to generate a timestamp
timestamp() {
  date +"%Y-%m-%d_%H-%M-%S"
}

# Store timestamp for logging
local_timestamp=$(timestamp)

# Print important paths
printf "\n🔍 ** Config path is: %s" "$config_path"
printf "\n🔍 ** Data path is: %s" "$data_path"
printf "\n💾 ** Checkpoint will be saved at: %s" "$checkpoints_path"
printf "\n📜 ** Logs will be saved at: %s\n" "$log_path"

# Run the fine-tuning process
fairseq-hydra-train \
  task.data="$data_path" \
  common.wandb_project="$wandb_name" \
  common.log_interval=50 \
  common.log_format=tqdm \
  model.w2v_path="$pretrained_model_path" \
  distributed_training.distributed_world_size="$gpus" \
  +common.tensorboard_logdir="$tensorboard_path" \
  checkpoint.save_dir="$checkpoints_path" \
  checkpoint.restore_file="$checkpoints_path/checkpoint_last.pt" \
  --config-dir "$config_path" \
  --config-name "$config_name"

echo "✅ Fine-tuning completed!"