import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import numpy as np
import random
import math
import time
import os
import re
import io
import spacy
import matplotlib.pyplot as plt
import urllib.request
import gzip
import pickle
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from pathlib import Path

# Set random seeds for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Download Multi30k dataset
def download_multi30k():
    base_url = "https://github.com/multi30k/dataset/raw/master/"
    
    os.makedirs("data", exist_ok=True)
    
    for lang in ['de', 'en']:
        url = f"{base_url}data/task1/raw/train.{lang}.gz"
        path = f"data/train.{lang}"
        
        if not os.path.exists(path):
            print(f"Downloading {url}...")
            import gzip
            with urllib.request.urlopen(url) as response:
                with gzip.GzipFile(fileobj=io.BytesIO(response.read())) as uncompressed:
                    file_content = uncompressed.read()
            
            with open(path, 'wb') as f:
                f.write(file_content)
    
    for lang in ['de', 'en']:
        url = f"{base_url}data/task1/raw/val.{lang}.gz"
        path = f"data/val.{lang}"
        
        if not os.path.exists(path):
            print(f"Downloading {url}...")
            import gzip
            with urllib.request.urlopen(url) as response:
                with gzip.GzipFile(fileobj=io.BytesIO(response.read())) as uncompressed:
                    file_content = uncompressed.read()
            
            with open(path, 'wb') as f:
                f.write(file_content)
    
    for lang in ['de', 'en']:
        url = f"{base_url}data/task1/raw/test_2016_flickr.{lang}.gz"
        path = f"data/test.{lang}"
        
        if not os.path.exists(path):
            print(f"Downloading {url}...")
            import gzip
            with urllib.request.urlopen(url) as response:
                with gzip.GzipFile(fileobj=io.BytesIO(response.read())) as uncompressed:
                    file_content = uncompressed.read()
            
            with open(path, 'wb') as f:
                f.write(file_content)
    
    print("Multi30k dataset downloaded successfully.")

# Read dataset files
def read_dataset_file(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
        # Try to decode with different encodings
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text = content.decode('latin1')
            except UnicodeDecodeError:
                text = content.decode('utf-8', errors='replace')
    
    return [line.strip() for line in text.splitlines()]

# Create a single file for SPM training
def prepare_spm_training_data():
    # Create directory for SPM
    os.makedirs("data/spm", exist_ok=True)
    
    # Combine train files for SPM training
    combined_lines = []
    
    # Add German training data
    german_lines = read_dataset_file('data/train.de')
    combined_lines.extend(german_lines)
    
    # Add English training data
    english_lines = read_dataset_file('data/train.en')
    combined_lines.extend(english_lines)
    
    # Write combined file
    with open('data/spm/spm_train.txt', 'w', encoding='utf-8') as f:
        for line in combined_lines:
            f.write(line + '\n')
    
    return 'data/spm/spm_train.txt'

# Train SentencePiece model
def train_spm_model(input_file, vocab_size=16000, model_type='bpe', model_prefix='data/spm/spm'):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        normalization_rule_name='nmt_nfkc',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='<PAD>',
        unk_piece='<UNK>',
        bos_piece='<SOS>',
        eos_piece='<EOS>'
    )
    
    # Load the trained model
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    
    return sp

# Custom dataset class with BPE tokenization
class TranslationDataset(Dataset):
    def __init__(self, german_texts, english_texts, sp_model, use_augmentation=False, drop_prob=0.05, replace_prob=0.05, swap_prob=0.05):
        self.german_texts = german_texts
        self.english_texts = english_texts
        self.sp_model = sp_model
        self.use_augmentation = use_augmentation
        self.drop_prob = drop_prob
        self.replace_prob = replace_prob
        self.swap_prob = swap_prob
        
    def __len__(self):
        return len(self.german_texts)
    
    def add_noise_to_sentence(self, tokens):
        if not self.use_augmentation:
            return tokens
            
        tokens = tokens.copy()
        
        # Randomly drop tokens
        tokens = [t for t in tokens if random.random() > self.drop_prob]
        
        # Randomly replace tokens with UNK
        tokens = [t if random.random() > self.replace_prob else 1 for t in tokens]  # 1 is UNK id
        
        # Randomly swap adjacent tokens
        for i in range(len(tokens)-1):
            if random.random() < self.swap_prob:
                tokens[i], tokens[i+1] = tokens[i+1], tokens[i]
                
        return tokens
    
    def __getitem__(self, index):
        german_text = self.german_texts[index]
        english_text = self.english_texts[index]
        
        # Encode with SentencePiece
        german_tokens = self.sp_model.encode(german_text, out_type=int)
        english_tokens = self.sp_model.encode(english_text, out_type=int)
        
        # Add data augmentation to source (German) if enabled
        if self.use_augmentation:
            german_tokens = self.add_noise_to_sentence(german_tokens)
        
        # Add <SOS> and <EOS> tokens to target (English)
        english_tokens = [2] + english_tokens + [3]  # 2 is <SOS>, 3 is <EOS>
        
        # Convert to tensors
        german_tensor = torch.LongTensor(german_tokens)
        english_tensor = torch.LongTensor(english_tokens)
        
        return german_tensor, english_tensor

# Collate function for DataLoader
def collate_fn(batch):
    # Separate source and target sequences
    srcs = [item[0] for item in batch]
    trgs = [item[1] for item in batch]
    
    # Pad sequences
    src_pad = pad_sequence(srcs, batch_first=True, padding_value=0)
    trg_pad = pad_sequence(trgs, batch_first=True, padding_value=0)
    
    return src_pad, trg_pad

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (won't be trained)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

# Multi-Head Attention Layer
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout_p=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Define projection matrices
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout_p)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Apply linear transformations and split into multiple heads
        Q = self.fc_q(query)  # [batch_size, q_len, d_model]
        K = self.fc_k(key)    # [batch_size, k_len, d_model]
        V = self.fc_v(value)  # [batch_size, v_len, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, n_heads, q_len, head_dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, n_heads, k_len, head_dim]
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, n_heads, v_len, head_dim]
        
        # Calculate attention scores
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # [batch_size, n_heads, q_len, k_len]
        
        # Apply mask if provided - use a smaller negative number for half precision compatibility
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e4)
        
        # Apply softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)  # [batch_size, n_heads, q_len, k_len]
        
        # Apply dropout
        attention = self.dropout(attention)
        
        # Calculate weighted sum
        x = torch.matmul(attention, V)  # [batch_size, n_heads, q_len, head_dim]
        
        # Concatenate heads
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, q_len, n_heads, head_dim]
        x = x.view(batch_size, -1, self.d_model)  # [batch_size, q_len, d_model]
        
        # Final linear layer
        x = self.fc_out(x)  # [batch_size, q_len, d_model]
        
        return x, attention

# Position-wise Feed-Forward Network
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p=0.1):
        super(PositionwiseFeedforwardLayer, self).__init__()
        
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = self.dropout(torch.relu(self.fc_1(x)))  # [batch_size, seq_len, d_ff]
        x = self.fc_2(x)  # [batch_size, seq_len, d_model]
        return x

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_p=0.1):
        super(EncoderLayer, self).__init__()
        
        # Define layers
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttentionLayer(d_model, n_heads, dropout_p)
        self.feedforward = PositionwiseFeedforwardLayer(d_model, d_ff, dropout_p)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, src, src_mask):
        # src shape: [batch_size, src_len, d_model]
        # src_mask shape: [batch_size, 1, 1, src_len]
        
        # Self-attention block
        _src, attention = self.self_attention(src, src, src, src_mask)
        # Apply dropout and residual connection
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        # Feedforward block
        _src = self.feedforward(src)
        # Apply dropout and residual connection
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        # Return output and attention weights
        return src, attention

# Full Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, d_ff, dropout_p=0.1, max_len=100):
        super(Encoder, self).__init__()
        
        # Embedding layer
        self.tok_embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout_p)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout_p)
            for _ in range(n_layers)
        ])
        
    def forward(self, src, src_mask):
        # src shape: [batch_size, src_len]
        # src_mask shape: [batch_size, 1, 1, src_len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        # Apply token embedding and positional encoding
        src = self.tok_embedding(src)  # [batch_size, src_len, d_model]
        src = self.pos_encoding(src)   # [batch_size, src_len, d_model]
        src = self.dropout(src)
        
        # Apply encoder layers
        attentions = []
        for layer in self.layers:
            src, attention = layer(src, src_mask)
            attentions.append(attention)
        
        return src, attentions

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_p=0.1):
        super(DecoderLayer, self).__init__()
        
        # Define layers
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.enc_attn_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttentionLayer(d_model, n_heads, dropout_p)
        self.encoder_attention = MultiHeadAttentionLayer(d_model, n_heads, dropout_p)
        self.feedforward = PositionwiseFeedforwardLayer(d_model, d_ff, dropout_p)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg shape: [batch_size, trg_len, d_model]
        # enc_src shape: [batch_size, src_len, d_model]
        # trg_mask shape: [batch_size, 1, trg_len, trg_len]
        # src_mask shape: [batch_size, 1, 1, src_len]
        
        # Self-attention block
        _trg, self_attention = self.self_attention(trg, trg, trg, trg_mask)
        # Apply dropout and residual connection
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        
        # Encoder-decoder attention block
        _trg, encoder_attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        # Apply dropout and residual connection
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        
        # Feedforward block
        _trg = self.feedforward(trg)
        # Apply dropout and residual connection
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        # Return output and attention weights
        return trg, self_attention, encoder_attention

# Full Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, n_layers, n_heads, d_ff, dropout_p=0.1, max_len=100):
        super(Decoder, self).__init__()
        
        # Embedding layer
        self.tok_embedding = nn.Embedding(output_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout_p)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout_p)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.fc_out = nn.Linear(d_model, output_dim)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg shape: [batch_size, trg_len]
        # enc_src shape: [batch_size, src_len, d_model]
        # trg_mask shape: [batch_size, 1, trg_len, trg_len]
        # src_mask shape: [batch_size, 1, 1, src_len]
        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        # Apply token embedding and positional encoding
        trg = self.tok_embedding(trg)  # [batch_size, trg_len, d_model]
        trg = self.pos_encoding(trg)   # [batch_size, trg_len, d_model]
        trg = self.dropout(trg)
        
        # Apply decoder layers
        self_attentions = []
        encoder_attentions = []
        for layer in self.layers:
            trg, self_attention, encoder_attention = layer(trg, enc_src, trg_mask, src_mask)
            self_attentions.append(self_attention)
            encoder_attentions.append(encoder_attention)
        
        # Apply output layer
        output = self.fc_out(trg)  # [batch_size, trg_len, output_dim]
        
        return output, self_attentions, encoder_attentions

# Full Seq2Seq Model with Transformer
class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super(Seq2SeqTransformer, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        # src shape: [batch_size, src_len]
        
        # Create mask to hide padding tokens
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
        
        return src_mask
    
    def make_trg_mask(self, trg):
        # trg shape: [batch_size, trg_len]
        
        # Create mask to hide padding tokens
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)  # [batch_size, 1, trg_len, 1]
        
        # Create mask to prevent attention to future tokens
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()  # [trg_len, trg_len]
        trg_sub_mask = trg_sub_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, trg_len, trg_len]
        
        # Combine both masks
        trg_mask = trg_pad_mask & trg_sub_mask  # [batch_size, 1, trg_len, trg_len]
        
        return trg_mask
    
    def forward(self, src, trg):
        # src shape: [batch_size, src_len]
        # trg shape: [batch_size, trg_len]
        
        # Create masks
        src_mask = self.make_src_mask(src)  # [batch_size, 1, 1, src_len]
        trg_mask = self.make_trg_mask(trg)  # [batch_size, 1, trg_len, trg_len]
        
        # Apply encoder
        enc_src, enc_attention = self.encoder(src, src_mask)  # [batch_size, src_len, d_model]
        
        # Apply decoder
        output, self_attention, enc_dec_attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        # output shape: [batch_size, trg_len, output_dim]
        
        return output, enc_attention, self_attention, enc_dec_attention

# Function to initialize model weights
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

# Function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training function with mixed precision
def train(model, iterator, optimizer, criterion, scaler, clip):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(tqdm(iterator, desc="Training")):
        src = src.to(device)
        trg = trg.to(device)
        
        # Remove <EOS> token from target for input to decoder
        trg_input = trg[:, :-1]
        # Remove <SOS> token from target for loss calculation
        trg_output = trg[:, 1:]
        
        # Enable autocasting for mixed precision
        with amp.autocast():
            # Forward pass
            output, _, _, _ = model(src, trg_input)
            
            # Reshape output and target for loss calculation
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg_output = trg_output.contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(output, trg_output)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Clip gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update parameters with gradient scaling
        scaler.step(optimizer)
        scaler.update()
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Update epoch loss
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# Evaluation function with mixed precision
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(tqdm(iterator, desc="Evaluating")):
            src = src.to(device)
            trg = trg.to(device)
            
            # Remove <EOS> token from target for input to decoder
            trg_input = trg[:, :-1]
            # Remove <SOS> token from target for loss calculation
            trg_output = trg[:, 1:]
            
            # Enable autocasting for mixed precision
            with amp.autocast():
                # Forward pass
                output, _, _, _ = model(src, trg_input)
                
                # Reshape output and target for loss calculation
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg_output = trg_output.contiguous().view(-1)
                
                # Calculate loss
                loss = criterion(output, trg_output)
            
            # Update epoch loss
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# Translation function
def translate_sentence(sentence, sp_model, model, device, max_len=50):
    model.eval()
    
    # Tokenize sentence
    if isinstance(sentence, str):
        tokens = sp_model.encode(sentence, out_type=int)
    else:
        tokens = sentence
    
    # Add batch dimension
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    # Create source mask
    src_mask = model.make_src_mask(src_tensor)
    
    # Encode the source sentence
    with torch.no_grad():
        enc_src, _ = model.encoder(src_tensor, src_mask)
    
    # Start with <SOS> token (id=2)
    trg_indices = [2]  # <SOS> token
    
    for i in range(max_len):
        # Convert current target to tensor
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
        
        # Create target mask
        trg_mask = model.make_trg_mask(trg_tensor)
        
        # Decode the source sentence
        with torch.no_grad():
            # The decoder returns 3 values
            output, _, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        # Get the next token prediction
        pred_token = output.argmax(2)[:, -1].item()
        
        # Add token to output
        trg_indices.append(pred_token)
        
        # Break if <EOS> token is predicted (id=3)
        if pred_token == 3:  # <EOS> token
            break
    
    # Convert indices to tokens
    trg_tokens = sp_model.decode_ids(trg_indices)
    
    return trg_tokens

# Calculate BLEU score
def calculate_bleu(model, test_data, sp_model, device):
    original_text = []
    translated_text = []
    
    for src, trg in tqdm(test_data, desc="Calculating BLEU"):
        # Convert tensors to lists of integers
        src_indices = src.tolist()
        trg_indices = trg.tolist()
        
        # Translate source sentence
        translation = translate_sentence(src_indices, sp_model, model, device)
        
        # Get target sentence (remove <SOS> and <EOS> tokens)
        # In SentencePiece: 2 is <SOS>, 3 is <EOS>, 0 is <PAD>
        trg_text = sp_model.decode_ids([idx for idx in trg_indices if idx not in [0, 2, 3]])
        
        # Add to lists for BLEU calculation
        original_text.append([trg_text])
        translated_text.append(translation)
    
    # Calculate BLEU score
    bleu = BLEU()
    bleu_score = bleu.corpus_score(translated_text, original_text).score
    
    return bleu_score

# Function to plot training history
def plot_training_history(train_losses, valid_losses, learning_rates=None, bleu_scores=None):
    if learning_rates:
        fig, axes = plt.subplots(3 if bleu_scores else 2, 1, figsize=(12, 15 if bleu_scores else 10))
    else:
        fig, axes = plt.subplots(2 if bleu_scores else 1, 1, figsize=(12, 10 if bleu_scores else 6))
        axes = [axes] if not bleu_scores else axes
    
    # Plot losses
    ax_idx = 0
    axes[ax_idx].plot(train_losses, label='Train Loss')
    axes[ax_idx].plot(valid_losses, label='Validation Loss')
    axes[ax_idx].set_xlabel('Epoch')
    axes[ax_idx].set_ylabel('Loss')
    axes[ax_idx].set_title('Training and Validation Loss')
    axes[ax_idx].legend()
    axes[ax_idx].grid(True)
    
    # Plot BLEU scores if available
    if bleu_scores:
        ax_idx += 1
        epochs = list(range(0, len(train_losses), 2)) if len(bleu_scores) * 2 == len(train_losses) else list(range(len(bleu_scores)))
        axes[ax_idx].plot(epochs, bleu_scores, marker='o', linestyle='-')
        axes[ax_idx].set_xlabel('Epoch')
        axes[ax_idx].set_ylabel('BLEU Score')
        axes[ax_idx].set_title('BLEU Score during Training')
        axes[ax_idx].grid(True)
    
    # Plot learning rates if available
    if learning_rates:
        ax_idx += 1
        axes[ax_idx].plot(learning_rates, marker='o', linestyle='-')
        axes[ax_idx].set_xlabel('Epoch')
        axes[ax_idx].set_ylabel('Learning Rate')
        axes[ax_idx].set_title('Learning Rate Schedule')
        axes[ax_idx].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Save model and vocabularies
def save_model_and_vocabularies(model, sp_model, model_path="model.pt", sp_model_path="spm.model"):
    """
    Save model state and SentencePiece model
    """
    # Save model state
    torch.save(model.state_dict(), model_path)
    
    # SentencePiece model is already saved during training
    print(f"Model saved to {model_path}")
    print(f"SentencePiece model at {sp_model_path}")

# Load model and vocabularies
def load_model_and_vocabularies(encoder_class, decoder_class, model_class, model_path="model.pt", sp_model_path="data/spm/spm.model", device=None):
    """
    Load model state and SentencePiece model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    
    # Get vocabulary size (SentencePiece model size)
    vocab_size = len(sp)
    
    # Create model
    SRC_PAD_IDX = 0  # <PAD> token id
    TRG_PAD_IDX = 0  # <PAD> token id
    
    INPUT_DIM = vocab_size
    OUTPUT_DIM = vocab_size
    D_MODEL = 512
    N_LAYERS = 4
    N_HEADS = 8
    D_FF = 2048
    DROPOUT = 0.3
    
    # Create encoder and decoder
    encoder = encoder_class(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout_p=DROPOUT
    )
    
    decoder = decoder_class(
        output_dim=OUTPUT_DIM,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout_p=DROPOUT
    )
    
    model = model_class(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=SRC_PAD_IDX,
        trg_pad_idx=TRG_PAD_IDX,
        device=device
    ).to(device)
    
    # Load model state
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    print(f"Model loaded from {model_path}")
    print(f"SentencePiece model loaded from {sp_model_path}")
    
    return model, sp

# Interactive translation
def interactive_translation(model, sp_model, device):
    """
    Interactive translation mode
    """
    print("Interactive Translation Mode")
    print("Enter 'q' to quit")
    
    while True:
        # Get input sentence
        sentence = input("\nEnter German sentence: ")
        
        if sentence.lower() == 'q':
            break
        
        # Translate
        translation = translate_sentence(sentence, sp_model, model, device)
        
        # Print translation
        print(f"English translation: {translation}")

# Main function to run training
def main():
    # Hyperparameters
    BATCH_SIZE = 128
    D_MODEL = 512  # Increased from 256
    N_LAYERS = 4   # Increased from 3
    N_HEADS = 8
    D_FF = 2048    # Increased from 512
    DROPOUT = 0.3  # Increased from 0.1
    LEARNING_RATE = 0.0005
    N_EPOCHS = 200  # Increased from 10
    CLIP = 1.0
    VOCAB_SIZE = 16000  # For SentencePiece
    
    # Early stopping parameters
    PATIENCE = 20
    
    # Download Multi30k dataset
    download_multi30k()
    
    # Load Multi30k dataset from files
    train_de = read_dataset_file('data/train.de')
    train_en = read_dataset_file('data/train.en')
    valid_de = read_dataset_file('data/val.de')
    valid_en = read_dataset_file('data/val.en')
    test_de = read_dataset_file('data/test.de')
    test_en = read_dataset_file('data/test.en')
    
    # Train SentencePiece model
    spm_train_file = prepare_spm_training_data()
    sp_model = train_spm_model(spm_train_file, vocab_size=VOCAB_SIZE)
    
    # Create datasets
    train_dataset = TranslationDataset(train_de, train_en, sp_model, use_augmentation=True)
    valid_dataset = TranslationDataset(valid_de, valid_en, sp_model, use_augmentation=False)
    test_dataset = TranslationDataset(test_de, test_en, sp_model, use_augmentation=False)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Get vocabulary sizes from SentencePiece model
    INPUT_DIM = len(sp_model)
    OUTPUT_DIM = len(sp_model)
    
    print(f"Vocabulary size: {INPUT_DIM}")
    
    # Create encoder and decoder
    encoder = Encoder(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout_p=DROPOUT
    )
    
    decoder = Decoder(
        output_dim=OUTPUT_DIM,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout_p=DROPOUT
    )
    
    # Create model
    SRC_PAD_IDX = 0  # <PAD> token id in SentencePiece
    TRG_PAD_IDX = 0  # <PAD> token id in SentencePiece
    
    model = Seq2SeqTransformer(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=SRC_PAD_IDX,
        trg_pad_idx=TRG_PAD_IDX,
        device=device
    ).to(device)
    
    # Initialize weights
    model.apply(initialize_weights)
    
    # Print model information
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    # Define optimizer and criterion with label smoothing
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX, label_smoothing=0.1)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Create gradient scaler for mixed precision training
    scaler = amp.GradScaler()
    
    # Training loop
    best_valid_loss = float('inf')
    patience_counter = 0
    train_losses = []
    valid_losses = []
    bleu_scores = []
    learning_rates = []
    
    # Create directory for saving models
    if not os.path.exists('models'):
        os.makedirs('models')
    
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        # Store current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Train model
        train_loss = train(model, train_dataloader, optimizer, criterion, scaler, CLIP)
        # Evaluate model
        valid_loss = evaluate(model, valid_dataloader, criterion)
        
        # Update scheduler
        scheduler.step(valid_loss)
        
        # Calculate BLEU score (using a subset for speed)
        if epoch % 2 == 0:  # Calculate BLEU every 2 epochs
            test_subset = [(test_dataset[i][0], test_dataset[i][1]) for i in range(min(100, len(test_dataset)))]
            bleu_score = calculate_bleu(model, test_subset, sp_model, device)
            bleu_scores.append(bleu_score)
            print(f'BLEU Score: {bleu_score:.2f}')
        
        # Update loss histories
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # Early stopping and model saving
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best-model.pt')
            
            # Save model
            save_model_and_vocabularies(model, sp_model, 
                                        model_path='models/best-model.pt',
                                        sp_model_path='data/spm/spm.model')
        else:
            patience_counter += 1
        
        # Print epoch information
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.0f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
        print(f'\tLearning Rate: {current_lr:.7f}')
        
        # Check if early stopping criteria is met
        if patience_counter >= PATIENCE:
            print(f'Early stopping after {epoch+1} epochs!')
            break
    
    # Plot training history
    plot_training_history(train_losses, valid_losses, learning_rates, bleu_scores)
    
    # Load best model
    model.load_state_dict(torch.load('models/best-model.pt'))
    
    # Final evaluation
    test_loss = evaluate(model, test_dataloader, criterion)
    print(f'Test Loss: {test_loss:.3f}')
    
    # Calculate BLEU score on full test set
    test_data_sample = [(test_dataset[i][0], test_dataset[i][1]) for i in range(min(500, len(test_dataset)))]
    final_bleu = calculate_bleu(model, test_data_sample, sp_model, device)
    print(f'Final BLEU Score: {final_bleu:.2f}')
    
    # Example translations
    examples = [
        "Ich liebe Programmierung.",
        "Die Katze sitzt auf dem Tisch.",
        "Heute ist ein schöner Tag.",
        "Ich möchte Deutsch lernen."
    ]
    
    print("\nExample Translations:")
    for example in examples:
        translation = translate_sentence(example, sp_model, model, device)
        print(f'DE: {example}')
        print(f'EN: {translation}')
        print()
    
    # Interactive translation mode
    interactive_translation(model, sp_model, device)

# If running as a script
if __name__ == "__main__":
    main()