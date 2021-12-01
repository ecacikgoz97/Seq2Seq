import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout_rate=0.5):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_rate)

    def forward(self, x):
        # x: [sequence_len, N]
        x = self.embedding(x)
        # x: [sequence_len, N, embedding_size]
        x = self.dropout(x)

        outputs, (hidden, cell) = self.lstm(x)
        # outputs: [sequence_len, N, hidden_size]
        return hidden, cell