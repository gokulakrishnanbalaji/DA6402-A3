import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os
from matplotlib import font_manager

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from with_attention.model import create_model as attention_create_model
from without_attention.model import create_model

def train_model(
        embed_size,
        hidden_size,
        encoder_layers,
        decoder_layers,
        cell_type,
        dropout,
        beam_width,
        attention_type
):
    # Paths to your local TSV files (update as needed)
    train_path = 'ta.translit.sampled.train.tsv'
    val_path = 'ta.translit.sampled.dev.tsv'
    test_path = 'ta.translit.sampled.test.tsv'

    # Create data loaders
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = prepare_data_loaders(train_path, val_path, test_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if attention_type == 'no':
        model = create_model(embed_size,
        hidden_size,
        encoder_layers,
        decoder_layers,
        cell_type,
        dropout,
        beam_width)
        # Beam search decoding
        def beam_search_decode(model, src, max_len, beam_width, sos_idx, eos_idx):
            model.eval()
            src = src.to(device)
            batch_size = src.size(0)
            encoder_outputs, hidden = model.encoder(src)
            
            # Adjust hidden state to match decoder layers
            if model.encoder.cell_type == 'LSTM':
                hidden, cell = hidden
                if model.encoder.num_layers != model.decoder.num_layers:
                    factor = model.decoder.num_layers // model.encoder.num_layers
                    if factor > 1:
                        hidden = hidden.repeat(factor, 1, 1)
                        cell = cell.repeat(factor, 1, 1)
                    else:
                        hidden = hidden[-model.decoder.num_layers:]
                        cell = cell[-model.decoder.num_layers:]
                hidden = (hidden, cell)
            else:
                if model.encoder.num_layers != model.decoder.num_layers:
                    factor = model.decoder.num_layers // model.encoder.num_layers
                    if factor > 1:
                        hidden = hidden.repeat(factor, 1, 1)
                    else:
                        hidden = hidden[-model.decoder.num_layers:]
            
            # Initialize beam
            beams = [(torch.tensor([sos_idx], device=device), hidden, 0.0)]  # (sequence, hidden, score)
            completed = []
            
            for _ in range(max_len):
                new_beams = []
                for seq, hid, score in beams:
                    if seq[-1].item() == eos_idx:
                        completed.append((seq, score))
                        continue
                    output, new_hidden = model.decoder(seq[-1].unsqueeze(0), hid)
                    probs = torch.softmax(output, dim=-1)
                    top_probs, top_idx = probs.topk(beam_width)
                    
                    for i in range(beam_width):
                        new_seq = torch.cat([seq, top_idx[:, i]])
                        new_score = score - math.log(top_probs[:, i].item())
                        new_beams.append((new_seq, new_hidden, new_score))
                
                # Keep top beam_width beams
                new_beams = sorted(new_beams, key=lambda x: x[2])[:beam_width]
                beams = new_beams
                
                if len(completed) >= beam_width:
                    break
            
            # Return best sequence
            completed = sorted(completed, key=lambda x: x[1])
            if completed:
                return completed[0][0]
            return beams[0][0]
        
    else:
        model = attention_create_model(embed_size,
        hidden_size,
        encoder_layers,
        decoder_layers,
        cell_type,
        dropout,
        beam_width, 
        attention_type=attention_type)
        # Beam search decoding (adapted for attention)
        def beam_search_decode(model, src, max_len, beam_width, sos_idx, eos_idx):
            model.eval()
            src = src.to(device)
            batch_size = src.size(0)
            encoder_outputs, hidden = model.encoder(src)
            
            if model.encoder.cell_type == 'LSTM':
                hidden, cell = hidden
                if model.encoder.num_layers != model.decoder.num_layers:
                    factor = model.decoder.num_layers // model.encoder.num_layers
                    if factor > 1:
                        hidden = hidden.repeat(factor, 1, 1)
                        cell = cell.repeat(factor, 1, 1)
                    else:
                        hidden = hidden[-model.decoder.num_layers:]
                        cell = cell[-model.decoder.num_layers:]
                hidden = (hidden, cell)
            else:
                if model.encoder.num_layers != model.decoder.num_layers:
                    factor = model.decoder.num_layers // model.encoder.num_layers
                    if factor > 1:
                        hidden = hidden.repeat(factor, 1, 1)
                    else:
                        hidden = hidden[-model.decoder.num_layers:]
            
            beams = [(torch.tensor([sos_idx], device=device), hidden, 0.0)]
            completed = []
            
            for _ in range(max_len):
                new_beams = []
                for seq, hid, score in beams:
                    if seq[-1].item() == eos_idx:
                        completed.append((seq, score))
                        continue
                    output, new_hidden, _ = model.decoder(seq[-1].unsqueeze(0), hid, encoder_outputs)
                    probs = torch.softmax(output, dim=-1)
                    top_probs, top_idx = probs.topk(beam_width)
                    
                    for i in range(beam_width):
                        new_seq = torch.cat([seq, top_idx[:, i]])
                        new_score = score - math.log(top_probs[:, i].item())
                        new_beams.append((new_seq, new_hidden, new_score))
                
                new_beams = sorted(new_beams, key=lambda x: x[2])[:beam_width]
                beams = new_beams
                
                if len(completed) >= beam_width:
                    break
            
            completed = sorted(completed, key=lambda x: x[1])
            if completed:
                return completed[0][0]
            return beams[0][0]

    # Training and evaluation function
    def train_model(model, train_loader, val_loader, test_loader, num_epochs=30, beam_width=3):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for src, tgt in train_loader:
                src, tgt = src.to(device), tgt.to(device)
                optimizer.zero_grad()
                output = model(src, tgt, teacher_forcing_ratio=0.5)
                output = output[:, 1:].reshape(-1, output.size(-1))
                tgt_flat = tgt[:, 1:].reshape(-1)
                loss = criterion(output, tgt_flat)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Compute validation loss and accuracy (one batch for speed)
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for src, tgt in val_loader:
                    src, tgt = src.to(device), tgt.to(device)
                    output = model(src, tgt, teacher_forcing_ratio=0.0)
                    output = output[:, 1:].reshape(-1, output.size(-1))
                    tgt_flat = tgt[:, 1:].reshape(-1)
                    loss = criterion(output, tgt_flat)
                    val_loss += loss.item()
                    
                    # Compute accuracy on one batch
                    print(f"Epoch {epoch+1}, Validation batch - src shape: {src.shape}, tgt shape: {tgt.shape}")
                    for i in range(src.size(0)):
                        pred = beam_search_decode(
                            model, src[i:i+1], max_len=50,
                            beam_width=beam_width,
                            sos_idx=tgt_vocab.char2idx['<SOS>'],
                            eos_idx=tgt_vocab.char2idx['<EOS>']
                        )
                        pred_str = ''.join([tgt_vocab.idx2char[idx.item()] for idx in pred if idx.item() not in [0, 1, 2]])
                        tgt_seq = tgt[i]
                        tgt_str = ''.join([tgt_vocab.idx2char[idx.item()] for idx in tgt_seq[1:] if idx.item() not in [0, 1, 2]])
                        if pred_str == tgt_str:
                            val_correct += 1
                        val_total += 1
                    break  # Process only one batch for validation accuracy
            
            # Log metrics to WandB
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss / len(train_loader),
                "val_loss": val_loss / len(val_loader),
                "val_accuracy": val_correct / val_total
            })
            print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_correct / val_total:.4f}")
        
        # Compute test accuracy and save predictions
        test_correct = 0
        test_total = 0
        predictions = []
        with torch.no_grad():
            for src, tgt in test_loader:
                src, tgt = src.to(device), tgt.to(device)
                print(f"Test batch - src shape: {src.shape}, tgt shape: {tgt.shape}")
                for i in range(src.size(0)):
                    pred = beam_search_decode(
                        model, src[i:i+1], max_len=50,
                        beam_width=beam_width,
                        sos_idx=tgt_vocab.char2idx['<SOS>'],
                        eos_idx=tgt_vocab.char2idx['<EOS>']
                    )
                    pred_str = ''.join([tgt_vocab.idx2char[idx.item()] for idx in pred if idx.item() not in [0, 1, 2]])
                    tgt_seq = tgt[i]
                    tgt_str = ''.join([tgt_vocab.idx2char[idx.item()] for idx in tgt_seq[1:] if idx.item() not in [0, 1, 2]])
                    src_str = ''.join([src_vocab.idx2char[idx.item()] for idx in src[i, 1:] if idx.item() not in [0, 1, 2]])
                    if pred_str == tgt_str:
                        test_correct += 1
                    test_total += 1
                    predictions.append({
                        'input_english': src_str,
                        'actual_tamil': tgt_str,
                        'predicted_tamil': pred_str
                    })
        
        test_accuracy = test_correct / test_total
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Save predictions to CSV
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv('test_predictions_vanilla.csv', index=False)
        print("Test predictions saved to 'test_predictions.csv'")
        
        return test_accuracy


    # Train and evaluate
    test_accuracy = train_model(
        model, train_loader, val_loader, test_loader,
        num_epochs=30, beam_width=best_params['beam_width']
    )

    # Log test accuracy to WandB
    wandb.log({"test_accuracy": test_accuracy})
    wandb.finish()

    if attention_type != 'no':
        import matplotlib.pyplot as plt
        import seaborn as sns
        import torch
        import numpy as np
        import os
        from matplotlib import font_manager

        import os
        import requests
        from matplotlib import font_manager as fm

        # Define target directory and font file path
        font_dir = os.path.expanduser("~/.fonts")
        os.makedirs(font_dir, exist_ok=True)

        # Download and save a Tamil font instead of Devanagari
        font_file = os.path.join(font_dir, "NotoSansTamil.ttf")
        url = "https://github.com/google/fonts/raw/main/ofl/notosanstamil/NotoSansTamil%5Bwdth%2Cwght%5D.ttf"

        # Only download if not already present
        if not os.path.exists(font_file):
            with requests.get(url) as resp:
                with open(font_file, "wb") as f:
                    f.write(resp.content)

        # Clear cached fonts and register the new font
        os.system("rm -rf ~/.cache/matplotlib")
        fm.fontManager.addfont(font_file)
        tamil_font = fm.FontProperties(fname=font_file)

        def inference_with_attention(model, src_tensor, src_vocab, tgt_vocab, max_len=50, device='cuda'):
            """
            Perform inference on a single source sequence, collecting attention weights.
            
            Args:
                model: Trained Seq2Seq model
                src_tensor: Source tensor of shape (1, src_len)
                src_vocab: Source vocabulary
                tgt_vocab: Target vocabulary
                max_len: Maximum length of the output sequence
                device: Device to run the model on
            
            Returns:
                predicted_tokens: List of predicted token indices
                attention_weights: List of attention weight matrices for each decoding step
            """
            model.eval()
            src_tensor = src_tensor.to(device)
            batch_size = 1
            
            with torch.no_grad():
                # Encoder
                encoder_outputs, hidden = model.encoder(src_tensor)
                
                # Initialize decoder input with <SOS>
                decoder_input = torch.tensor([tgt_vocab.char2idx['<SOS>']], device=device)
                predicted_tokens = []
                attention_weights = []
                
                if model.decoder.cell_type == 'LSTM':
                    hidden = (hidden[0], hidden[1])
                
                for _ in range(max_len):
                    output, hidden, attn_weights = model.decoder(decoder_input, hidden, encoder_outputs)
                    attention_weights.append(attn_weights.squeeze(0).cpu().numpy())  # (seq_len,)
                    
                    # Get the predicted token
                    _, topi = output.topk(1)
                    decoder_input = topi.squeeze(-1).detach()  # (batch_size,)
                    predicted_token = decoder_input.item()
                    predicted_tokens.append(predicted_token)
                    
                    if predicted_token == tgt_vocab.char2idx['<EOS>']:
                        break
                        
            return predicted_tokens, attention_weights

        def plot_attention_heatmaps(model, test_loader, src_vocab, tgt_vocab, device='cuda', num_samples=9):
            """
            Plot a 3x3 grid of attention heatmaps for the specified number of test samples and log to WandB.
            
            Args:
                model: Trained Seq2Seq model
                test_loader: DataLoader for test data
                src_vocab: Source vocabulary
                tgt_vocab: Target vocabulary
                device: Device to run the model on
                num_samples: Number of samples to plot (default 9 for 3x3 grid)
            
            Returns:
                None (saves the plot to a file and logs to WandB)
            """
            model.eval()
            samples = []
            
            # Collect samples from test_loader
            for src, tgt in test_loader:
                for i in range(src.size(0)):
                    samples.append((src[i:i+1], tgt[i:i+1]))
                    if len(samples) >= num_samples:
                        break
                if len(samples) >= num_samples:
                    break
            
            # Set up the 3x3 grid
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            axes = axes.flatten()
            
            for idx, (src_tensor, tgt_tensor) in enumerate(samples[:num_samples]):
                # Run inference
                predicted_tokens, attention_weights = inference_with_attention(
                    model, src_tensor, src_vocab, tgt_vocab, device=device
                )
                
                # Convert source and target tokens to characters
                src_chars = [src_vocab.idx2char[idx.item()] for idx in src_tensor[0] if idx.item() != src_vocab.char2idx['<PAD>']]
                pred_chars = [tgt_vocab.idx2char[idx] for idx in predicted_tokens if idx != tgt_vocab.char2idx['<PAD>']]
                
                # Stack attention weights into a matrix (tgt_len, src_len)
                attention_matrix = np.stack(attention_weights, axis=0)  # (tgt_len, src_len)
                
                # Plot heatmap without yticklabels first
                sns.heatmap(
                    attention_matrix,
                    ax=axes[idx],
                    xticklabels=src_chars,
                    yticklabels=[],  # Empty initially
                    cmap='viridis',
                    cbar=False
                )
                
                # Manually set y-tick labels with Tamil font
                axes[idx].set_yticks(np.arange(len(pred_chars)) + 0.5)
                axes[idx].set_yticklabels(pred_chars, fontproperties=tamil_font, fontsize=10)
                
                axes[idx].set_title(f"Sample {idx+1}")
                axes[idx].set_xlabel("Source Sequence")
                axes[idx].set_ylabel("Predicted Sequence", fontproperties=tamil_font)
            
            plt.tight_layout()
            plt.plot()
            
            # Save the plot
            plot_path = 'attention_heatmaps.png'
            plt.savefig(plot_path, bbox_inches='tight')
            
            # Log the plot to WandB
            try:
                import wandb
                wandb.log({"attention_heatmaps": wandb.Image(plot_path)})
                print("Attention heatmaps logged to WandB")
            except Exception as e:
                print(f"Failed to log to WandB: {e}")
            
            plt.close()

            plot_attention_heatmaps(model, test_loader, src_vocab, tgt_vocab, device=device)

