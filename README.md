# German-to-English Neural Machine Translation

This project implements a robust German to English neural machine translation model using an enhanced transformer-based architecture with PyTorch. The implementation is designed to address overfitting issues and improve translation quality through several key optimizations.

## Features

- Transformer-based sequence-to-sequence model with enhanced architecture
- SentencePiece (BPE) tokenization for improved handling of rare words
- Data augmentation with random token dropping, replacement, and swapping
- Stronger regularization with increased dropout and weight decay
- Label smoothing to prevent overconfidence in predictions
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting
- CUDA GPU acceleration with Automatic Mixed Precision (AMP)
- Multi30K dataset for training and evaluation
- BLEU score tracking
- Comprehensive training progress visualization

## Requirements

- Python 3.9+
- PyTorch 2.2+
- CUDA-enabled GPU (RTX 3080Ti/3060 12 GB Used for training and eval)
- Additional dependencies:
  - spacy
  - tqdm
  - matplotlib
  - sacrebleu
  - sentencepiece

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SunilPrasath-USD/AAI501-NeuralMachineTranslation.git
cd AAI501-NeuralMachineTranslation
```

2. Install dependencies:
```bash
pip install torch tqdm matplotlib spacy sacrebleu sentencepiece
```

3. Download Spacy language models:
```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Training the Model

To train the model, run the improved implementation script:

```bash
python Seq2Seq_m30Kv1.py
```

The training process:
1. Downloads the Multi30K dataset automatically
2. Trains a SentencePiece model for tokenization
3. Creates datasets with data augmentation for training
4. Builds and initializes the enhanced transformer model
5. Trains with label smoothing and adaptive learning rate
6. Implements early stopping based on validation performance
7. Tracks and visualizes training metrics
8. Provides example translations and interactive mode

### Improvements Over Base Implementation

This implementation includes several key enhancements to address overfitting:

1. **Stronger Regularization**:
   - Increased dropout rate (0.3 vs 0.1)
   - Added weight decay to optimizer (1e-5)
   - Implemented label smoothing (0.1)

2. **Learning Rate Scheduling**:
   - ReduceLROnPlateau scheduler reduces learning rate when validation loss plateaus
   - Visualization of learning rate changes throughout training

3. **Improved Architecture**:
   - Increased model capacity (D_MODEL: 512, D_FF: 2048)
   - Deeper network with 4 transformer layers
   - Better parameter initialization

4. **Better Training Protocol**:
   - Early stopping with patience of 5 epochs
   - More training epochs (20 vs 10)
   - Best model selection based on validation loss

5. **Data Augmentation**:
   - Random token dropping (5% chance)
   - Random token replacement with UNK (5% chance)
   - Random adjacent token swapping (5% chance)

6. **BPE Tokenization**:
   - SentencePiece model with 16,000 tokens
   - Better handling of rare words and morphological variations

## Inference

After training, you can use the inference script for translations:

```bash
python Seq2Seq_inference.py --interactive
```

### Options

- `--model-path`: Path to the trained model file (default: 'models/best-model.pt')
- `--spm-model-path`: Path to the SentencePiece model file (default: 'data/spm/spm.model')
- `--input-file`: Path to input file with German sentences (one per line)
- `--output-file`: Path to output file for English translations
- `--interactive`: Run in interactive mode
- `--gpu`: Use GPU if available

## Examples

### Batch Translation

```bash
python Seq2Seq_inference.py --input-file german_sentences.txt --output-file english_translations.txt
```

### Interactive Translation

```bash
python Seq2Seq_inference.py --interactive
```

## Performance Monitoring

During training, the implementation tracks:

- Training loss
- Validation loss
- Learning rate changes
- BLEU scores (calculated periodically)

A comprehensive plot of these metrics is saved as `training_history.png` at the end of training.

## Hyperparameters

The following hyperparameters can be adjusted in the code:

- `BATCH_SIZE`: Batch size for training (default: 128)
- `D_MODEL`: Dimension of model embeddings (default: 512)
- `N_LAYERS`: Number of encoder/decoder layers (default: 4)
- `N_HEADS`: Number of attention heads (default: 8)
- `D_FF`: Dimension of feedforward layer (default: 2048)
- `DROPOUT`: Dropout rate (default: 0.3)
- `LEARNING_RATE`: Initial learning rate (default: 0.0005)
- `N_EPOCHS`: Maximum number of training epochs (default: 20)
- `CLIP`: Gradient clipping value (default: 1.0)
- `VOCAB_SIZE`: SentencePiece vocabulary size (default: 16000)
- `PATIENCE`: Early stopping patience (default: 5)

## Results

The improvements implemented in this version significantly reduce the gap between training and validation loss, resulting in better generalization and higher BLEU scores. The model is less prone to overfitting and produces more natural translations.

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
