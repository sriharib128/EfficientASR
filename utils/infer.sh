#!/bin/bash

# Input arguments
parentdir="$1"
subset="$7"
spm_path="$8"

# Save predicted text files flag (1 = Save, 0 = Default)
dest_folder="$2"
save_predicted=1 

# Paths for model and results
pretrained_model_path="$3"
checkpoint_path="$4"
result_path="$5"
data_path="$6"

# Create result directory if not exists
mkdir -p "$result_path"

echo "🔍 Running inference with the following parameters:"
echo "📁 Data Path: $data_path"
echo "📌 Checkpoint Path: $checkpoint_path"
echo "📝 Result Path: $result_path"
echo "🔠 SPM Path: $spm_path"
echo "🎯 Subset: $subset"

# Run inference
python ./utils/inference/infer.py "$data_path" \
    --task audio_finetuning \
    --nbest 1 \
    --path "$checkpoint_path" \
    --gen-subset "$subset" \
    --results-path "$result_path" \
    --w2l-decoder viterbi \
    --lm-weight 2 \
    --word-score -1 \
    --sil-weight 0 \
    --criterion ctc \
    --labels ltr \
    --max-tokens 6000000 \
    --post-process "$spm_path" \
    --model-overrides "{'w2v_path':'${pretrained_model_path}'}"

echo "✅ Inference completed."

# Run WER evaluation
python ./utils/inference/wer_wav2vec.py \
    -o "${result_path}/ref.word-ur_cp2.pt-${subset}.txt" \
    -p "${result_path}/hypo.word-ur_cp2.pt-${subset}.txt" \
    -t "${data_path}/${subset}.tsv" \
    -s ${save_predicted} \
    -n "${result_path}/sentence_wise_wer.csv" \
    -e true

echo "📊 WER evaluation completed."

# Save predicted outputs if enabled
if [ "$save_predicted" -eq 1 ]; then
    echo "💾 Saving predicted outputs..."
    python ./utils/inference/save_predicted_output.py \
        -f "${result_path}/sentence_wise_wer.csv" \
        -d "$dest_folder"
    echo "✅ Predicted outputs saved to: $dest_folder"
fi

echo "🚀 Script execution finished!"
