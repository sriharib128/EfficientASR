#!/bin/bash

# Function to generate dataset manifest and labels
generate_manifest_and_labels() {
    local name=$1
    local wav_path=$2
    local dest_path=$3
    local spm_model=$4

    python utils/data_generation/manifest.py ${wav_path} --dest ${dest_path} --ext wav --train-name ${name} --valid-percent 0 --jobs -1
    echo "Manifest for ${name} created."

    python utils/data_generation/sp_labels.py --jobs 64 --tsv ${dest_path}/${name}.tsv --output-dir ${dest_path} --output-name ${name} --txt-dir ${wav_path} --spm_path ${spm_model}
    echo "Word file for ${name} generated."
}

# Function to generate dictionary and perform analysis
generate_dictionary_and_analysis() {
    local name=$1
    local dest_path=$2
    local spm_model=$3

    python utils/data_generation/sp_dict_gen.py --wrd ${dest_path}/${name}.wrd --lexicon ${dest_path}/lexicon.lst --dict ${dest_path}/dict.ltr.txt --vocab_size 512 --spm_path ${spm_model}
    echo "Dictionary for ${name} generated."

    echo "Analyzing dictionary for punctuation marks..."
    python utils/data_generation/generate_dict_analysis.py --dict ${dest_path}/dict.ltr.txt

    echo "Analyzing ${name} TSV file..."
    python utils/data_generation/generate_wav_report_from_tsv.py --tsv ${dest_path}/${name}.tsv
}

# Main logic
caller=$1

if [ "$caller" = "train" ]; then
    train_name="train"
    valid_name="valid"
    train_wav_path=$2
    valid_wav_path=$3
    make_valid_from_train=$4
    valid_percentage=$5
    destination_path=$6
    spm_path=$7

    # Create training and validation dataset
    if [ "$make_valid_from_train" = "1" ]; then
        python utils/data_generation/manifest.py ${train_wav_path} --dest ${destination_path} --ext wav --train-name ${train_name} --valid-percent ${valid_percentage} --jobs -1
        echo "Manifest creation done. Validation data extracted from train set."
    else
        valid_percentage=0.0
        generate_manifest_and_labels "${train_name}" "${train_wav_path}" "${destination_path}" "${spm_path}"
        generate_manifest_and_labels "${valid_name}" "${valid_wav_path}" "${destination_path}" "${spm_path}"
    fi

    # Generate dictionary and perform analysis
    generate_dictionary_and_analysis "${train_name}" "${destination_path}" "${spm_path}"
    generate_dictionary_and_analysis "${valid_name}" "${destination_path}" "${spm_path}"

    echo "Training data preparation completed."

elif [ "$caller" = "test" ]; then
    test_name="test"
    test_folder=$2
    test_data_path=$3
    spm_model=$4
    finetuning_dict=$5

    mkdir -p ${test_data_path}

    # Generate test manifest and labels
    generate_manifest_and_labels "${test_name}" "${test_folder}" "${test_data_path}" "${spm_model}"

    # Generate dictionary and perform analysis
    generate_dictionary_and_analysis "${test_name}" "${test_data_path}" "${spm_model}"

    # Copy fine-tuning dictionary
    echo "Copying fine-tuning dictionary from ${finetuning_dict} to ${test_data_path}"
    cp ${finetuning_dict} ${test_data_path}

    echo "Test data preparation completed."
fi
