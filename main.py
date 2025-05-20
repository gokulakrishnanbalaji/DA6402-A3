# main.py

import argparse
from train_model import train_model

def main():
    parser = argparse.ArgumentParser(description="Train a Seq2Seq Transliteration Model")

    parser.add_argument("--embed_size", type=int, required=True, help="Size of embedding vectors")
    parser.add_argument("--hidden_size", type=int, required=True, help="Number of hidden units in encoder/decoder")
    parser.add_argument("--encoder_layers", type=int, required=True, help="Number of layers in the encoder")
    parser.add_argument("--decoder_layers", type=int, required=True, help="Number of layers in the decoder")
    parser.add_argument("--cell_type", choices=["lstm", "gru", "rnn"], required=True, help="RNN cell type to use")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (0.0 means no dropout)")
    parser.add_argument("--beam_width", type=int, default=1, help="Beam width for beam search (1 means greedy)")
    parser.add_argument("--attention_type", choices=["no", "bahdanau", "dot"], default="no", help="Type of attention mechanism")

    args = parser.parse_args()

    # Call the training function with parsed arguments
    train_model(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        cell_type=args.cell_type,
        dropout=args.dropout,
        beam_width=args.beam_width,
        attention_type=args.attention_type
    )

if __name__ == "__main__":
    main()
