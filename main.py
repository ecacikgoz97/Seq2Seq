import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq
from train import training
from utils import tokinezer_german, tokinezer_english, translate_sentence, bleu, save_checkpoint, load_checkpoint

import numpy as np
import spacy
import random

spacy_german = spacy.load("de_core_news_sm")
spacy_english = spacy.load("en_core_news_sm")
german = Field(tokenize=tokinezer_german(spacy_german), lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokinezer_english(spacy_english), lower=True, init_token='<sos>', eos_token='<eos>')

train, val, test = Multi30k.splits(exts=(".de", ".en"), fields=(german, english))

german.build_vocab(train, max_size=10000, min_freq=2)
english.build_vocab(train, max_size=10000, min_freq=2)

# Hyperparameters
EPOCHS = 100
LR = 0.001
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE_ENC = len(german.vocab)
INPUT_SIZE_DEC = len(english.vocab)
OUTPUT_SIZE = len(english.vocab)
EMBEDDING_SIZE_ENC = 300
EMBEDDING_SIZE_DEC = 300
HIDDEN_SIZE = 1024  # Needs to be the same for both RNN's
NUM_LAYERS = 2
SENTENCE = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

train_loader, val_loader, test_loader = BucketIterator.splits(
    (train, val, test),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=DEVICE,
)

encoder = Encoder(INPUT_SIZE_ENC, EMBEDDING_SIZE_ENC, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)

decoder = Decoder(INPUT_SIZE_DEC, EMBEDDING_SIZE_DEC, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS).to(DEVICE)

model = Seq2Seq(encoder, decoder).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

training(EPOCHS, train_loader, model, optimizer, criterion, SENTENCE, german, english, DEVICE)
score = bleu(test_loader[1:100], model, german, english, DEVICE)
print(f"Blue Score: {score*100:.3f}")