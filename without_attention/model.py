# Importing libraries

import torch
import torch.nn as nn

# Encoder RNN

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, cell_type='RNN', num_layers=1, dropout=0.0):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embed_size)
        self.cell_type = cell_type.upper()
        
        rnn_class = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[self.cell_type]
        self.rnn = rnn_class(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        if self.cell_type == 'LSTM':
            outputs, (hidden, cell) = self.rnn(embedded)
            return outputs, (hidden, cell)
        else:
            outputs, hidden = self.rnn(embedded)
            return outputs, hidden
        

# Decoder RNN

class DecoderRNN(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, cell_type='RNN', num_layers=1, dropout=0.0):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.cell_type = cell_type.upper()
        
        rnn_class = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[self.cell_type]
        self.rnn = rnn_class(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_char, hidden):
        embedded = self.embedding(input_char).unsqueeze(1)
        embedded = self.dropout(embedded)
        if self.cell_type == 'LSTM':
            output, (hidden, cell) = self.rnn(embedded, hidden)
            output = self.out(output.squeeze(1))
            return output, (hidden, cell)
        else:
            output, hidden = self.rnn(embedded, hidden)
            output = self.out(output.squeeze(1))
            return output, hidden

# Seq2Seq model

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.size(0)
        target_len = target.size(1)
        outputs = torch.zeros(batch_size, target_len, self.decoder.out.out_features).to(source.device)

        encoder_outputs, hidden = self.encoder(source)
        
        decoder_input = target[:, 0]
        
        # Handle differing encoder/decoder layers
        if self.encoder.cell_type == 'LSTM':
            hidden, cell = hidden
            if self.encoder.num_layers != self.decoder.num_layers:
                # Compute repeat factor or select layers
                if self.decoder.num_layers > self.encoder.num_layers:
                    factor = self.decoder.num_layers // self.encoder.num_layers + 1
                    hidden = hidden.repeat(factor, 1, 1)[:self.decoder.num_layers]
                    cell = cell.repeat(factor, 1, 1)[:self.decoder.num_layers]

                else:
                    hidden = hidden[-self.decoder.num_layers:]
                    cell = cell[-self.decoder.num_layers:]
            hidden = (hidden, cell)
        else:
            # For RNN and GRU
            if self.encoder.num_layers != self.decoder.num_layers:
                if self.decoder.num_layers > self.encoder.num_layers:
                    factor = self.decoder.num_layers // self.encoder.num_layers + 1
                    hidden = hidden.repeat(factor, 1, 1)[:self.decoder.num_layers]
                else:
                    hidden = hidden[-self.decoder.num_layers:]
        
        for t in range(1, target_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target[:, t] if teacher_force else top1

        return outputs


# Wrapper function to create model

def create_model(input_vocab_size, output_vocab_size, embed_size=256, hidden_size=512, 
                 cell_type='RNN', encoder_layers=1, decoder_layers=1, dropout=0.0):
    encoder = EncoderRNN(input_vocab_size, embed_size, hidden_size, cell_type, encoder_layers, dropout)
    decoder = DecoderRNN(output_vocab_size, embed_size, hidden_size, cell_type, decoder_layers, dropout)
    model = Seq2Seq(encoder, decoder)
    return model        