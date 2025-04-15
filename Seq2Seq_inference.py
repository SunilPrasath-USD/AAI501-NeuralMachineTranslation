import torch
import argparse
import sentencepiece as spm
from tqdm import tqdm
import sys
import os

# Import model components
# Make sure these imports match your main implementation
from improved_implementation import (
    Encoder, Decoder, Seq2SeqTransformer, 
    translate_sentence, interactive_translation,
    load_model_and_vocabularies
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='German-English Neural Machine Translation Inference')
    parser.add_argument('--model-path', type=str, default='models/best-model.pt',
                        help='Path to the trained model file')
    parser.add_argument('--spm-model-path', type=str, default='data/spm/spm.model',
                        help='Path to the SentencePiece model file')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Path to input file with German sentences (one per line)')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Path to output file for English translations')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    return parser.parse_args()

def translate_file(input_file, output_file, model, sp_model, device):
    # Read German sentences from input file
    with open(input_file, 'r', encoding='utf-8') as f:
        german_sentences = [line.strip() for line in f]
    
    # Translate sentences
    translations = []
    for sentence in tqdm(german_sentences, desc="Translating"):
        translation = translate_sentence(sentence, sp_model, model, device)
        translations.append(translation)
    
    # Write translations to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for translation in translations:
            f.write(translation + '\n')
    
    print(f"Translated {len(translations)} sentences from {input_file} to {output_file}")

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if model and spm files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.spm_model_path):
        print(f"Error: SentencePiece model file not found at {args.spm_model_path}")
        sys.exit(1)
    
    # Load model and SentencePiece model
    try:
        model, sp_model = load_model_and_vocabularies(
            Encoder, Decoder, Seq2SeqTransformer,
            model_path=args.model_path,
            sp_model_path=args.spm_model_path,
            device=device
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Set model to evaluation mode
    model.eval()
    
    # Run in file mode or interactive mode
    if args.input_file and args.output_file:
        translate_file(args.input_file, args.output_file, model, sp_model, device)
    elif args.interactive:
        interactive_translation(model, sp_model, device)
    else:
        print("Error: Either specify input/output files or use interactive mode.")
        parser.print_help()

if __name__ == "__main__":
    main()