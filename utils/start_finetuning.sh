#!/bin/bash

CUDA_VISIBLE_DEVICES=$1
pretrained_model_path=$2
config_name=$3
config_path=$4
wandb_name=$5
data_path=$6
checkpoints_path=$7
tensorboard_path=$8
log_path=$9

echo "Fine-tuning started with:"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "Pretrained Model Path: $pretrained_model_path"
echo "Config Name: $config_name"
echo "Config Path: $config_path"
echo "Data Path: $data_path"
echo "Checkpoints Path: $checkpoints_path"
echo "TensorBoard Path: $tensorboard_path"

#cp2_modelpath
# pretrained_model_path='/data/bhavana/Interspeech/checkpoints/pretrained/xlsr_300_epoch2.pt'

printf "\n** Config path is: $config_path"
printf "\n** Data path is: $data_path"
printf "\n** Checkpoint will be saved at: $checkpoints_path"
printf "\n** Logs will be saved at: ${log_path}"

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}


fairseq-hydra-train \
  task.data=${data_path} \
  common.wandb_project=${wandb_name}\
  common.log_interval=50 common.log_format=tqdm\
  model.w2v_path=${pretrained_model_path} \
  +common.tensorboard_logdir=${tensorboard_path} \
  checkpoint.save_dir=${checkpoints_path} \
  checkpoint.restore_file=${checkpoints_path}/checkpoint_last.pt \
  --config-dir ${config_path} --config-name ${config_name} 