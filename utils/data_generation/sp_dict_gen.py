import argparse
import sentencepiece as spm
import numpy as np

def load_sentencepiece_model(spm_path):
    """Loads the SentencePiece model."""
    return spm.SentencePieceProcessor(model_file=f'{spm_path}.model')

def read_word_file(wrd_file):
    """Reads the word file and extracts unique words."""
    with open(wrd_file, mode='r', encoding='utf-8') as file:
        text_lines = file.readlines()
    total_words = " ".join(text_lines).split(" ")
    total_words = [word.strip() for word in total_words]
    return np.unique(total_words)  # Using numpy for unique elements

def create_lexicon_file(unique_words, sp, lexicon_path):
    """Generates a lexicon file with word-to-subword mappings."""
    unique_characters_dict = []

    with open(lexicon_path, mode='w+', encoding='utf-8') as lexicon_file:
        for word in unique_words:
            if word:  # Ignore empty words
                subwords = sp.encode_as_pieces(word)
                unique_characters_dict.extend(subwords)
                lexicon_file.write(f"{word}\t{' '.join(subwords)} |\n")

    return unique_characters_dict  # Return list of all subwords for dictionary creation

def create_dictionary_file(unique_characters_dict, dict_path):
    """Generates a dictionary file from unique characters."""
    unique_character_set = ['|'] + list(np.unique(unique_characters_dict))

    with open(dict_path, mode='w+', encoding='utf-8') as dict_file:
        for char in unique_character_set:
            dict_file.write(f"{char} {unique_characters_dict.count(char)}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate lexicon and dictionary using SentencePiece.")

    parser.add_argument("--wrd", type=str, required=True, help="Path to the word file (.wrd)")
    parser.add_argument("--lexicon", type=str, default='lexicon.lst', help="Path to save the lexicon file")
    parser.add_argument("--dict", type=str, default='dict.ltr.txt', help="Path to save the dictionary file")
    parser.add_argument("--spm_path", type=str, required=True, help="Path to SentencePiece model (without extension)")

    args = parser.parse_args()

    print("Loading SentencePiece model...")
    sp = load_sentencepiece_model(args.spm_path)
    
    print("Reading and processing word file...")
    unique_words = read_word_file(args.wrd)
    
    print("Generating lexicon file...")
    unique_characters_dict = create_lexicon_file(unique_words, sp, args.lexicon)

    print("Generating dictionary file...")
    create_dictionary_file(unique_characters_dict, args.dict)

    print("✅ Lexicon and dictionary files successfully created!")

if __name__ == "__main__":
    main()
