
 # Usage: python labels.py --jobs 64 --tsv <path to train.tsv>train.tsv --output-dir <destination dir> --output-name test --txt-dir

import argparse
import os
import re
from tqdm import tqdm
from joblib import Parallel, delayed
import sentencepiece as spm


def get_text(line,root):
    txt_path = line.split("\t")[0].replace(".wav",".txt").strip() ## implies that the text filename and wav filename should be same

    txt_path = os.path.join( root , txt_path )

    text = ''
    with open(txt_path , mode = "r", encoding="utf-8") as file_local:
        text = file_local.readline().strip()

    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type = str, help = "TSV file for which labels need to be generated")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--txt-dir")
    parser.add_argument("--jobs", default=-1, type=int, help="Number of jobs to run in parallel")
    parser.add_argument("--spm_path",type=str,help="sentence piece path",required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tsv_file=args.tsv
    output_dir=args.output_dir
    output_name=args.output_name

    with open(tsv_file) as tsv, open(
            os.path.join(output_dir, output_name + ".ltr"), "w",encoding="utf-8"
        ) as ltr_out, open(
            os.path.join(output_dir, output_name + ".wrd"), "w",encoding="utf-8"
        ) as wrd_out:

        root = next(tsv).strip()

        if not args.txt_dir:
            args.txt_dir = root
        
        local_arr = []

        local_arr.extend(Parallel(n_jobs = args.jobs)( delayed(get_text)(line , args.txt_dir) for line in tqdm(tsv)))
        wrd_out.writelines("\n".join(local_arr))

        print(".wrd file Generated ######################")

        with open(os.path.join(output_dir, output_name + ".wrd"), mode='r', encoding='utf-8') as file_local:
            text_lines = file_local.readlines()
        
        with open('temp.txt', 'w', encoding='utf-8') as temp_file:
            temp_file.write("".join(text_lines))
        

        os.makedirs(args.spm_path,exist_ok=True)
        if(args.output_name =="train"):
            spm.SentencePieceTrainer.train(input='temp.txt', model_prefix=args.spm_path, vocab_size=512, character_coverage=0.9995,model_type='bpe')

        # Load the SentencePiece model
        sp = spm.SentencePieceProcessor(model_file=args.spm_path+".model")

        formatted_text = []
        for text in local_arr:
            tokenized_words = text.strip().split()  # Word-level split
            word_level_tokenised = [" ".join(sp.encode_as_pieces(word)) for word in tokenized_words]
            final_text = " | ".join(word_level_tokenised) + " |"
            formatted_text.append(final_text)
        
        ltr_out.writelines("\n".join(formatted_text))

        print(".ltr file Generated ######################")
        

if __name__ == "__main__":
    main()
