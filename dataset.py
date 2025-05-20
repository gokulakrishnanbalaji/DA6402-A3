# Importing Libraries

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
from collections import Counter
import os
import math

# Vocabulary class to handle character-to-index mapping
class Vocabulary:
    def __init__(self):
        self.char2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2char = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.size = 4

    def add_sequence(self, sequence):
        for char in sequence:
            if char not in self.char2idx:
                self.char2idx[char] = self.size
                self.idx2char[self.size] = char
                self.size += 1

    def get_indices(self, sequence):
        indices = [self.char2idx.get(char, self.char2idx['<UNK>']) for char in sequence]
        return indices

# Custom Dataset class

class DakshinaDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data.iloc[idx, 1]  # English (Latin)
        tgt = self.data.iloc[idx, 0]  # Tamil
        src_indices = [self.src_vocab.char2idx['<SOS>']] + self.src_vocab.get_indices(src) + [self.src_vocab.char2idx['<EOS>']]
        tgt_indices = [self.tgt_vocab.char2idx['<SOS>']] + self.tgt_vocab.get_indices(tgt) + [self.tgt_vocab.char2idx['<EOS>']]
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)
    

# Function to load and preprocess data
def load_dakshina_data(train_path, val_path, test_path):
    # Read TSV files without headers
    train_df = pd.read_csv(train_path, sep='\t', header=None, usecols=[0, 1])
    val_df = pd.read_csv(val_path, sep='\t', header=None, usecols=[0, 1])
    test_df = pd.read_csv(test_path, sep='\t', header=None, usecols=[0, 1])

    # Ensure strings
    train_df[0] = train_df[0].astype(str)
    train_df[1] = train_df[1].astype(str)
    val_df[0] = val_df[0].astype(str)
    val_df[1] = val_df[1].astype(str)
    test_df[0] = test_df[0].astype(str)
    test_df[1] = test_df[1].astype(str)

    # Build vocabularies
    src_vocab = Vocabulary()  # English (Latin)
    tgt_vocab = Vocabulary()  # Tamil

    # Add characters to vocab from training data
    for _, row in train_df.iterrows():
        src_vocab.add_sequence(row[1])
        tgt_vocab.add_sequence(row[0])

    # Create datasets
    train_dataset = DakshinaDataset(train_df, src_vocab, tgt_vocab)
    val_dataset = DakshinaDataset(val_df, src_vocab, tgt_vocab)
    test_dataset = DakshinaDataset(test_df, src_vocab, tgt_vocab)

    return train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab

# Collate function for DataLoader
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    # Pad sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded

# Wrapper function for easier access

def prepare_data_loaders(train_path, val_path, test_path, batch_size=32):
    train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab = load_dakshina_data(train_path, val_path, test_path)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


