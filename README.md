# DA6402-A3

# Sequence-to-Sequence Transliteration using RNNs (with and without Attention)

This repository contains code for a sequence-to-sequence transliteration model built using RNNs with optional attention mechanisms. The task involves transliterating text using character-level encoder-decoder architecture.

## Directory Structure

```
├── with_attention/ # Models and scripts related to attention mechanisms
├── without_attention/ # Models and scripts without attention mechanisms
├── dataset.py # Dataset loader and preprocessing
├── train_model.py # Training script
├── main.py # Model runner for specific hyperparameters
├── connectivity_visualisation.py # Visualizes training/validation connectivity
├── part-a.ipynb # Sweep notebook - Part A
├── part-b.ipynb # Sweep notebook - Part B
├── ta.translit.sampled.*.tsv # Dev/Test/Train datasets
├── test_predictions_attention.csv # Predictions using attention-based model
├── test_predictions_vanilla.csv # Predictions using vanilla model
├── README.md # Project documentation
```

## How to Use

### 1. Running Hyperparameter Sweeps

Hyperparameter sweeps can be run using the Jupyter Notebooks:

- `part-a.ipynb` – for initial setup and model sweeps
- `part-b.ipynb` – for further experimentation and evaluation

These notebooks include configuration for Weights & Biases (W&B) sweeps and help in automating hyperparameter optimization.

### 2. Running a Model with Specific Hyperparameters

You can run a model directly by executing `main.py` with the following command-line arguments:

```bash
python main.py \
  --embed_size 128 \
  --hidden_size 256 \
  --encoder_layers 2 \
  --decoder_layers 2 \
  --cell_type gru \
  --dropout 0.2 \
  --beam_width 3 \
  --attention_type bahdanau

```

## Argument Details:
```
--embed_size (int, required): Size of the embedding vectors

--hidden_size (int, required): Number of hidden units in encoder/decoder

--encoder_layers (int, required): Number of layers in the encoder

--decoder_layers (int, required): Number of layers in the decoder

--cell_type (choices: lstm, gru, rnn, required): Type of RNN cell to use

--dropout (float, optional): Dropout rate (default is 0.0)

--beam_width (int, optional): Beam width for beam search (default is 1 for greedy decoding)

--attention_type (choices: no, bahdanau, dot, optional): Type of attention mechanism to use (default is no)
```
## Dataset

The dataset used is the Dakshina Dataset, and the following files are included:

`ta.translit.sampled.train.tsv`

`ta.translit.sampled.dev.tsv`

`ta.translit.sampled.test.tsv`

These contain Tamil transliteration samples for training and evaluation.

## Outputs
test_predictions_attention.csv: Model predictions using attention

test_predictions_vanilla.csv: Model predictions without attention

## Visualization
You can visualize training dynamics and attention connectivity using `connectivity_visualisation.py`.
