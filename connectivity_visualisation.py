import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def plot_connectivity_visualization(model, src_text, tgt_text, src_vocab, tgt_vocab, device='cuda'):
    """
    Create a connectivity visualization showing which input characters the model
    attends to when generating each output character.
    
    Args:
        model: Trained Seq2Seq model with attention
        src_text: Source text as a string
        tgt_text: Target text as a string (for reference)
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: Device to run the model on
    """
    # Convert source text to tensor
    src_indices = [src_vocab.char2idx.get(c, src_vocab.char2idx['<UNK>']) for c in src_text]
    src_tensor = torch.tensor([src_indices], device=device)
    
    # Run inference with attention
    predicted_tokens, attention_weights = inference_with_attention(
        model, src_tensor, src_vocab, tgt_vocab, device=device
    )

   
    
    # Get predicted text
    predicted_chars = [tgt_vocab.idx2char[idx] for idx in predicted_tokens 
                      if idx not in [tgt_vocab.char2idx['<PAD>'], tgt_vocab.char2idx['<EOS>']]]
    predicted_text = ''.join(predicted_chars)


    
    # Stack attention weights into a matrix (tgt_len, src_len)
    attention_matrix = np.stack(attention_weights, axis=0)
    
    # Create a custom colormap similar to the one in the Distill article
    colors = [(1, 1, 1), (0.8, 0, 0.8)]  # White to purple
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the heatmap
    sns.heatmap(attention_matrix, cmap=cmap, vmin=0, vmax=np.max(attention_matrix),
                cbar=False, ax=ax)
    
    # Set the labels
    ax.set_xticks(np.arange(len(src_text)) + 0.5)
    ax.set_xticklabels(list(src_text), fontsize=12)
    
    ax.set_yticks(np.arange(len(predicted_chars)) + 0.5)
    ax.set_yticklabels(predicted_chars, fontsize=12, fontproperties=tamil_font)
    
    # Add grid lines to match the Distill visualization style
    ax.grid(False)
    for i in range(len(src_text) + 1):
        ax.axvline(i, color='black', linewidth=0.5)
    for i in range(len(predicted_chars) + 1):
        ax.axhline(i, color='black', linewidth=0.5)
    
    # Add titles and labels
    ax.set_title('Connectivity: Which input character is attended to for each output?', 
                 fontsize=16, pad=20)
    ax.set_xlabel('Input Sequence', fontsize=14, labelpad=10)
    ax.set_ylabel('Output Sequence', fontsize=14, labelpad=10)
    
    # Add the source and target text as subtitles
    plt.figtext(0.5, 0.01, f'Input: {src_text}', ha='center', fontsize=12)
    plt.figtext(0.5, 0.04, f'Output: {predicted_text}', ha='center', fontsize=12, 
               fontproperties=tamil_font)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.plot()
    
    # Save the figure
    plt.savefig('connectivity_visualization.png', dpi=300, bbox_inches='tight')
    
    # Log to WandB if available
    try:
        import wandb
        wandb.log({"connectivity_visualization": wandb.Image('connectivity_visualization.png')})
    except:
        pass
    
    plt.show()

# Example usage
src_text = "farm"  # English input
tgt_text = "ஃபார்ம்"  # Tamil output (for reference)

wandb.init(project = "DL-A3", name = "Question-6-attention")

plot_connectivity_visualization(
    model=model,
    src_text=src_text,
    tgt_text=tgt_text,
    src_vocab=src_vocab,
    tgt_vocab=tgt_vocab,
    device='cuda'
)

wandb.finish()
