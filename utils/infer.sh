#!/bin/bash

w2l_decoder_viterbi=$1 # 1 for viterbi, 0 for kenlm
parentdir=$2
beam=1024 # 128 or 1024
subset='test'

# FOR LM MODEL
lm_name='lm_fa_cp2'
lm_model_path="${parentdir}/lm/${lm_name}/lm.binary"
lexicon_lst_path="${parentdir}/lm/${lm_name}/lexicon.lst"
echo "$lexicon_lst_path"

# SAVE PREDICTED TEXT FILES
dest_folder=$3
save_predicted=1 # 1 to save, 0 default

# FOR pretrained model
pretrained_model_path=$4
checkpoint_path=$5
result_path=$6
data_path=$7
echo "================"
echo "${result_path}"
echo "${data_path}"

if [ "$w2l_decoder_viterbi" -eq 1 ]; then
    mkdir -p "$result_path"
    python ./utils/inference/infer.py "$data_path" --task audio_finetuning \
    --nbest 1 --path "$checkpoint_path" --gen-subset "$subset" --results-path "$result_path" --w2l-decoder viterbi \
    --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 6000000 \
    --post-process sentencepiece --model-overrides "{'w2v_path':'${pretrained_model_path}'}"

    python ./utils/inference/wer_wav2vec.py -o "${result_path}/ref.word-checkpoint_best.pt-${subset}.txt" \
    -p "${result_path}/hypo.word-checkpoint_best.pt-${subset}.txt" -t "${data_path}/${subset}.tsv" -s save \
    -n "${result_path}/sentence_wise_wer.csv" -e true

    if [ "$save_predicted" -eq 1 ]; then
        python ./utils/inference/save_predicted_output.py -f "${result_path}/sentence_wise_wer.csv" -d "$dest_folder"
    fi

else
    kenlm_result_path="${result_path}_${lm_name}_${beam}"
    mkdir -p "$kenlm_result_path"

    python ./utils/inference/infer.py "$data_path" --task audio_finetuning \
    --nbest 1 --path "$checkpoint_path" --gen-subset "$subset" --results-path "$kenlm_result_path" \
    --w2l-decoder kenlm --lm-model "$lm_model_path" --lm-weight 2 --word-score -1 --sil-weight 0 \
    --criterion ctc --labels ltr --max-tokens 6000000 --lexicon "$lexicon_lst_path" \
    --post-process letter --beam "$beam" --model-overrides "{'w2v_path':'${pretrained_model_path}'}"

    python ./utils/inference/wer_wav2vec.py -o "${kenlm_result_path}/ref.word-checkpoint_best.pt-${subset}.txt" \
    -p "${kenlm_result_path}/hypo.word-checkpoint_best.pt-${subset}.txt" -t "${data_path}/${subset}.tsv" -s save \
    -n "${kenlm_result_path}/sentence_wise_wer.csv" -e true

    if [ "$save_predicted" -eq 1 ]; then
        python ./utils/inference/save_predicted_output.py -f "${kenlm_result_path}/sentence_wise_wer.csv" -d "$dest_folder"
    fi
fi
