import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout_rate):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_rate)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        x = self.embedding(x)
        # x: [1, N, embedding_size]
        x = self.dropout(x)

        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        # outputs: [1, N, hidden_size]

        outputs = self.fc(outputs).squeeze(0)
        # outputs: [N, output_size]
        return outputs, hidden, cell
