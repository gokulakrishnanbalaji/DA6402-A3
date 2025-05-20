class Attention(nn.Module):
    def __init__(self, hidden_size, attention_type='bahdanau'):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_type = attention_type.lower()
        
        if self.attention_type == 'bahdanau':
            self.Wa = nn.Linear(hidden_size * 2, hidden_size)
            self.Ua = nn.Linear(hidden_size, 1, bias=False)
        elif self.attention_type == 'dot':
            pass  # Dot-product attention uses no additional parameters
        else:
            raise ValueError("Unsupported attention type. Use 'bahdanau' or 'dot'.")

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (num_layers, batch_size, hidden_size) or tuple for LSTM
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Validate encoder_outputs shape
        assert encoder_outputs.dim() == 3, f"Expected encoder_outputs to be 3D, got {encoder_outputs.shape}"
        assert encoder_outputs.size(2) == self.hidden_size, f"Expected encoder_outputs hidden_size {self.hidden_size}, got {encoder_outputs.size(2)}"
        
        # Extract the last layer of decoder hidden state
        if isinstance(decoder_hidden, tuple):  # LSTM case
            decoder_hidden = decoder_hidden[0]  # Take hidden state, not cell state
        decoder_hidden = decoder_hidden[-1]  # (batch_size, hidden_size)
        
        assert decoder_hidden.dim() == 2, f"Expected decoder_hidden to be 2D, got {decoder_hidden.shape}"
        assert decoder_hidden.size(1) == self.hidden_size, f"Expected decoder_hidden hidden_size {self.hidden_size}, got {decoder_hidden.size(1)}"
        
        if self.attention_type == 'bahdanau':
            # Repeat decoder hidden to match seq_len
            decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_size)
            # Combine with encoder outputs
            combined = torch.cat((decoder_hidden, encoder_outputs), dim=2)  # (batch_size, seq_len, hidden_size*2)
            # Compute energy
            energy = torch.tanh(self.Wa(combined))  # (batch_size, seq_len, hidden_size)
            attention_scores = self.Ua(energy).squeeze(2)  # (batch_size, seq_len)
        else:  # dot
            # Compute dot-product attention
            decoder_hidden = decoder_hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)
            encoder_outputs_t = encoder_outputs.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
            # Verify shapes before bmm
            assert decoder_hidden.shape == (batch_size, 1, self.hidden_size), f"Expected decoder_hidden (batch_size, 1, hidden_size), got {decoder_hidden.shape}"
            assert encoder_outputs_t.shape == (batch_size, self.hidden_size, seq_len), f"Expected encoder_outputs_t (batch_size, hidden_size, seq_len), got {encoder_outputs_t.shape}"
            attention_scores = torch.bmm(decoder_hidden, encoder_outputs_t).squeeze(1)  # (batch_size, seq_len)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_size)
        
        return context, attention_weights
    

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
class DecoderRNN(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, cell_type='RNN', num_layers=1, dropout=0.0, attention_type='bahdanau'):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.cell_type = cell_type.upper()
        
        rnn_class = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[self.cell_type]
        self.rnn = rnn_class(
            input_size=embed_size + hidden_size,  # Input includes context vector
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = Attention(hidden_size, attention_type)
        self.out = nn.Linear(hidden_size * 2, output_size)  # Combine RNN output and context

    def forward(self, input_char, hidden, encoder_outputs):
        embedded = self.embedding(input_char).unsqueeze(1)  # (batch_size, 1, embed_size)
        embedded = self.dropout(embedded)
        
        # Compute attention
        context, attention_weights = self.attention(hidden, encoder_outputs)  # context: (batch_size, hidden_size)
        
        # Concatenate context with embedded input
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)  # (batch_size, 1, embed_size + hidden_size)
        
        if self.cell_type == 'LSTM':
            output, (hidden, cell) = self.rnn(rnn_input, hidden)
            output = output.squeeze(1)  # (batch_size, hidden_size)
            output = torch.cat((output, context), dim=1)  # (batch_size, hidden_size * 2)
            output = self.out(output)  # (batch_size, output_size)
            return output, (hidden, cell), attention_weights
        else:
            output, hidden = self.rnn(rnn_input, hidden)
            output = output.squeeze(1)
            output = torch.cat((output, context), dim=1)
            output = self.out(output)
            return output, hidden, attention_weights
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
                factor = self.decoder.num_layers // self.encoder.num_layers + 1
                hidden = hidden.repeat(factor, 1, 1)[:self.decoder.num_layers]
                cell = cell.repeat(factor, 1, 1)[:self.decoder.num_layers]
            hidden = (hidden, cell)
        else:
            if self.encoder.num_layers != self.decoder.num_layers:
                factor = self.decoder.num_layers // self.encoder.num_layers + 1
                hidden = hidden.repeat(factor, 1, 1)[:self.decoder.num_layers]
        
        for t in range(1, target_len):
            output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target[:, t] if teacher_force else top1

        return outputs
def create_model(input_vocab_size, output_vocab_size, embed_size=256, hidden_size=512, 
                 cell_type='RNN', encoder_layers=1, decoder_layers=1, dropout=0.0, attention_type='bahdanau'):
    encoder = EncoderRNN(input_vocab_size, embed_size, hidden_size, cell_type, encoder_layers, dropout)
    decoder = DecoderRNN(output_vocab_size, embed_size, hidden_size, cell_type, decoder_layers, dropout, attention_type)
    model = Seq2Seq(encoder, decoder)
    return model