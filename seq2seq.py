import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_size):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_size = target_vocab_size

    def forward(self, source, target, ratio=0.5):
        len_target, batch_size = source.shape[0], source.shape[1]
        # <SOS>
        x = target[0]

        outputs = torch.zeros(len_target, batch_size, self.target_vocab_size)
        hidden, cell = self.encoder(source)

        for i in range(1, len_target):

            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[i] = output
            best_pred = output.argmax(1)

            if random.random() < ratio:
                x = target[i]
            else:
                x = best_pred

        return outputs

