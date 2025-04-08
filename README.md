# German-to-English Neural Machine Translation

This repository contains a complete implementation of a neural machine translation (NMT) system for translating German text to English using the Transformer architecture.

## Project Structure

- `german_english_nmt.py`: Main implementation of the Transformer model and related components
- `preprocess.py`: Data preprocessing script
- `train.py`: Model training script
- `translate.py`: Inference script for translation
- `evaluate_model.py`: Comprehensive model evaluation script
- `utils.py`: Utility functions for evaluation and visualization

## Requirements

- Python 3.9+
- PyTorch 2.2+
- CUDA-enabled GPU (Can do on CPU but 30x slower)
- Libraries:
  - torchtext
  - nltk
  - spacy
  - numpy
  - matplotlib
  - seaborn
  - pandas
  - tqdm
  - sacrebleu

## Installation

```bash
# Clone the repository
git clone https://github.com/SunilPrasath-USD/AAI501-NeuralMachineTranslation
cd german-english-nmt

# Install dependencies
pip install torch torchtext nltk spacy numpy matplotlib seaborn pandas tqdm sacrebleu

# Download spaCy language models
python -m spacy download en_core_web_sm
python -m spacy download de_core_web_sm

# Download NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

## Quick Start

### Data Preprocessing

First, preprocess your parallel corpus:

```bash
python preprocess.py \
    --src_file data/train.de \
    --tgt_file data/train.en \
    --output_dir data/processed \
    --max_len 100 \
    --src_vocab_size 50000 \
    --tgt_vocab_size 50000 \
    --lower_case
```

### Training

Train the Transformer model:

```bash
python train.py \
    --data_dir data/processed \
    --src_vocab data/processed/vocab.de.pt \
    --tgt_vocab data/processed/vocab.en.pt \
    --output_dir models/de-en-transformer \
    --batch_size 64 \
    --epochs 10 \
    --lr 0.0005 \
    --warmup_steps 4000 \
    --patience 5 \
    --d_model 512 \
    --n_heads 8 \
    --n_enc_layers 6 \
    --n_dec_layers 6
```

### Translation

Translate German text to English:

```bash
# Interactive mode
python translate.py \
    --model models/de-en-transformer/best_model.pt \
    --src_vocab data/processed/vocab.de.pt \
    --trg_vocab data/processed/vocab.en.pt \
    interactive \
    --beam_size 5

# Translate from file
python translate.py \
    --model models/de-en-transformer/best_model.pt \
    --src_vocab data/processed/vocab.de.pt \
    --trg_vocab data/processed/vocab.en.pt \
    file \
    --input test.de \
    --output translations.en \
    --beam_size 5
```

### Evaluation

Evaluate the model:

```bash
python evaluate_model.py \
    --model models/de-en-transformer/best_model.pt \
    --src_vocab data/processed/vocab.de.pt \
    --trg_vocab data/processed/vocab.en.pt \
    --test_data data/processed/test.de \
    --reference_data data/processed/test.en \
    --output_dir evaluation_results \
    comprehensive
```

## Model Architecture

The model follows the Transformer architecture with the following components:

- **Encoder-Decoder Architecture**: The model consists of encoder and decoder stacks.
- **Multi-Head Attention**: Allows the model to focus on different parts of the input sequence simultaneously.
- **Position-wise Feed-Forward Networks**: Applied to each position separately and identically.
- **Positional Encoding**: Adds information about the position of tokens in the sequence.
- **Layer Normalization and Residual Connections**: Helps with training stability.
- **Beam Search for Inference**: Explores multiple possible translations to find the best one.

### Default hyperparameters:

- Model dimension (d_model): 512
- Number of attention heads: 8
- Number of encoder layers: 6
- Number of decoder layers: 6
- Feed-forward dimension: 2048
- Dropout rate: 0.1
- Learning rate: 0.0005 with warmup and decay

## Advanced Features

### Beam Search Visualization

Visualize the beam search process for a specific sentence:

```bash
python translate.py \
    --model models/de-en-transformer/best_model.pt \
    --src_vocab data/processed/vocab.de.pt \
    --trg_vocab data/processed/vocab.en.pt \
    visualize \
    --text "Der Klimawandel ist ein globales Problem." \
    --beam_size 5
```

### Attention Visualization

Visualize attention patterns:

```bash
python evaluate_model.py \
    --model models/de-en-transformer/best_model.pt \
    --src_vocab data/processed/vocab.de.pt \
    --trg_vocab data/processed/vocab.en.pt \
    --test_data data/examples.de \
    --reference_data data/examples.en \
    --output_dir attention_visualizations \
    attention
```

### Beam Size Comparison

Compare the performance of different beam sizes:

```bash
python evaluate_model.py \
    --model models/de-en-transformer/best_model.pt \
    --src_vocab data/processed/vocab.de.pt \
    --trg_vocab data/processed/vocab.en.pt \
    --test_data data/processed/test.de \
    --reference_data data/processed/test.en \
    --output_dir beam_size_comparison \
    beam \
    --beam_sizes 1 3 5 10
```

## Performance

The model achieves the following performance on the WMT14 German-English test set:

- **BLEU Score**: ~27-28 (comparable to the original paper's results)
- **Training Time**: ~12 hours on a single NVIDIA RTX 3080Ti GPU

## Example Translations

| German | English (Reference) | English (Model) |
|--------|-------------------|----------------|
| Ich gehe zur Schule. | I am going to school. | I am going to school. |
| Obwohl es regnete, ging sie ohne Regenschirm spazieren. | Although it was raining, she went for a walk without an umbrella. | Although it was raining, she went for a walk without an umbrella. |
| Der Klimawandel ist ein globales Problem. | Climate change is a global problem. | Climate change is a global problem. |
| Berlin ist die Hauptstadt von Deutschland. | Berlin is the capital of Germany. | Berlin is the capital of Germany. |

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.
2. Ott, M., Edunov, S., Baevski, A., Fan, A., Gross, S., Ng, N., Grangier, D., & Auli, M. (2019). fairseq: A fast, extensible toolkit for sequence modeling. Proceedings of NAACL-HLT 2019: Demonstrations.
3. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

- Sunil Prasath
- Badriram
- Rahul Pawar
