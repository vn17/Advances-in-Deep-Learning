import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoProcessor
from PIL import Image
import numpy as np

from .base_vlm import BaseVLM
from .clip import CLIP 

def compare_pooling_methods():
    """Compare different text pooling strategies to diagnose the issue."""
    print("="*80)
    print("🔬 POOLING METHOD COMPARISON")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load components
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    vlm = BaseVLM()
    text_encoder = vlm.model.model.text_model
    
    # Test with diverse texts
    texts = [
        "A photo of a cute dog running in the park",
        "A scientific diagram of a molecule", 
        "Abstract art with red and blue colors",
        "A receipt from a grocery store"
    ]
    
    # Tokenize
    text_inputs = processor(
        text=texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(device)
    
    input_ids = text_inputs['input_ids']
    attention_mask = text_inputs['attention_mask']
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask:\n{attention_mask}\n")
    
    # Get sequence lengths (number of real tokens, not padding)
    sequence_lengths = attention_mask.sum(dim=1)
    print(f"Sequence lengths: {sequence_lengths.tolist()}\n")
    
    # Forward pass through text encoder
    with torch.no_grad():
        text_out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = text_out.last_hidden_state  # [batch, seq_len, hidden_dim]
    
    print(f"Hidden states shape: {hidden_states.shape}\n")
    print("="*80)
    
    # Test 3 different pooling methods
    pooling_results = {}
    
    # Method 1: LAST TOKEN (your original - WRONG for padding)
    print("\n1️⃣  LAST TOKEN POOLING (Original - Likely Wrong)")
    print("-" * 80)
    last_token_indices = (sequence_lengths - 1).clamp(min=0)
    batch_size = hidden_states.shape[0]
    tfeat_last = hidden_states[torch.arange(batch_size, device=device), last_token_indices]
    
    print(f"Last token indices: {last_token_indices.tolist()}")
    print(f"Features shape: {tfeat_last.shape}")
    
    # Normalize and compute similarity
    tfeat_last_norm = F.normalize(tfeat_last, p=2, dim=-1)
    sim_last = torch.matmul(tfeat_last_norm, tfeat_last_norm.T).float().cpu().numpy()
    
    mask = ~np.eye(4, dtype=bool)
    avg_sim_last = np.mean(sim_last[mask])
    print(f"Average pairwise similarity: {avg_sim_last:.4f}")
    
    if avg_sim_last > 0.90:
        print("❌ FAIL: Embeddings are nearly identical (likely pooling padding)")
    elif avg_sim_last < 0.70:
        print("✅ GOOD: Embeddings are distinct")
    else:
        print("⚠️  WARNING: Moderate similarity - needs investigation")
    
    pooling_results['Last Token'] = sim_last
    
    # Method 2: MEAN POOLING (Recommended)
    print("\n2️⃣  MEAN POOLING WITH ATTENTION MASK (Recommended)")
    print("-" * 80)
    
    # Expand attention mask to match hidden states dimensions
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    
    # Multiply hidden states by mask to zero out padding
    masked_hidden = hidden_states * mask_expanded
    
    # Sum over sequence dimension
    sum_embeddings = torch.sum(masked_hidden, dim=1)
    
    # Divide by number of real tokens (avoid division by zero)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    tfeat_mean = sum_embeddings / sum_mask
    
    print(f"Features shape: {tfeat_mean.shape}")
    
    # Normalize and compute similarity
    tfeat_mean_norm = F.normalize(tfeat_mean, p=2, dim=-1)
    sim_mean = torch.matmul(tfeat_mean_norm, tfeat_mean_norm.T).float().cpu().numpy()
    
    avg_sim_mean = np.mean(sim_mean[mask])
    print(f"Average pairwise similarity: {avg_sim_mean:.4f}")
    
    if avg_sim_mean > 0.90:
        print("❌ FAIL: Embeddings are nearly identical")
    elif avg_sim_mean < 0.70:
        print("✅ GOOD: Embeddings are distinct")
    else:
        print("⚠️  WARNING: Moderate similarity")
    
    pooling_results['Mean Pooling'] = sim_mean
    
    # Method 3: CLS TOKEN (if available)
    print("\n3️⃣  FIRST TOKEN (CLS-style) POOLING")
    print("-" * 80)
    tfeat_cls = hidden_states[:, 0, :]  # First token
    
    print(f"Features shape: {tfeat_cls.shape}")
    
    tfeat_cls_norm = F.normalize(tfeat_cls, p=2, dim=-1)
    sim_cls = torch.matmul(tfeat_cls_norm, tfeat_cls_norm.T).float().cpu().numpy()
    
    avg_sim_cls = np.mean(sim_cls[mask])
    print(f"Average pairwise similarity: {avg_sim_cls:.4f}")
    
    if avg_sim_cls > 0.90:
        print("❌ FAIL: Embeddings are nearly identical")
    elif avg_sim_cls < 0.70:
        print("✅ GOOD: Embeddings are distinct")
    else:
        print("⚠️  WARNING: Moderate similarity")
    
    pooling_results['First Token'] = sim_cls
    
    # Method 4: Check what happens if we IGNORE attention mask (bad!)
    print("\n4️⃣  MEAN POOLING WITHOUT MASK (Wrong - for comparison)")
    print("-" * 80)
    tfeat_mean_bad = hidden_states.mean(dim=1)
    
    tfeat_mean_bad_norm = F.normalize(tfeat_mean_bad, p=2, dim=-1)
    sim_mean_bad = torch.matmul(tfeat_mean_bad_norm, tfeat_mean_bad_norm.T).float().cpu().numpy()
    
    avg_sim_mean_bad = np.mean(sim_mean_bad[mask])
    print(f"Average pairwise similarity: {avg_sim_mean_bad:.4f}")
    print("(This should be HIGH because it averages padding tokens)")
    
    pooling_results['Mean (No Mask)'] = sim_mean_bad
    
    # Visualization
    print("\n" + "="*80)
    print("📊 GENERATING COMPARISON PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    methods = ['Last Token', 'Mean Pooling', 'First Token', 'Mean (No Mask)']
    text_labels = [t[:25] + "..." if len(t) > 25 else t for t in texts]
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        sim_matrix = pooling_results[method]
        
        sns.heatmap(
            sim_matrix, 
            annot=True, 
            fmt=".3f", 
            xticklabels=text_labels,
            yticklabels=text_labels,
            cmap="RdYlGn_r",
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            cbar_kws={'label': 'Cosine Similarity'}
        )
        
        avg_sim = np.mean(sim_matrix[mask])
        ax.set_title(f"{method}\nAvg Similarity: {avg_sim:.4f}", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("pooling_comparison.png", dpi=150)
    print("✓ Saved to 'pooling_comparison.png'\n")
    
    # Summary
    print("="*80)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    best_method = min(methods[:-1], key=lambda m: np.mean(pooling_results[m][mask]))
    print(f"\n✅ BEST METHOD: {best_method}")
    print(f"   Average similarity: {np.mean(pooling_results[best_method][mask]):.4f}")
    
    print("\n💡 INTERPRETATION:")
    print("   • Similarity < 0.60: Excellent - embeddings are very distinct")
    print("   • Similarity 0.60-0.75: Good - reasonable separation")
    print("   • Similarity 0.75-0.85: Warning - might need investigation")
    print("   • Similarity > 0.85: Critical - likely pooling padding tokens")
    
    print("\n🔧 RECOMMENDATION:")
    if avg_sim_mean < 0.75:
        print("   Use MEAN POOLING WITH ATTENTION MASK (Method 2)")
    elif avg_sim_last < avg_sim_mean:
        print("   Use LAST TOKEN POOLING (Method 1) - but verify it's correct")
    else:
        print("   Use MEAN POOLING (Method 2) - safest option")
    
    print("\n" + "="*80)
    
    return pooling_results


def run_full_model_sanity_check():
    """Test the full CLIP model with the chosen pooling method."""
    print("\n" + "="*80)
    print("🚀 FULL MODEL SANITY CHECK")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    vlm = BaseVLM()
    
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    
    model = CLIP(vision_encoder, text_encoder, proj_dim=256).to(device)
    if device == "cuda":
        model = model.bfloat16()
    model.eval()
    
    texts = [
        "A photo of a cute dog running in the park",
        "A scientific diagram of a molecule", 
        "Abstract art with red and blue colors",
        "A receipt from a grocery store"
    ]
    
    # Use random images (untrained projections will be random, but that's okay for this test)
    images = torch.randn(len(texts), 3, 224, 224).to(device)
    if device == "cuda":
        images = images.bfloat16()
    
    text_inputs = processor(
        text=texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        vnorm, tnorm, logits = model(
            pixel_values=images,
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask']
        )
    
    # Check text embedding diversity
    text_sim_matrix = torch.matmul(tnorm, tnorm.T).cpu().float().numpy()
    mask = ~np.eye(text_sim_matrix.shape[0], dtype=bool)
    avg_sim = np.mean(text_sim_matrix[mask])
    
    print(f"\n📊 Text Embedding Similarity: {avg_sim:.4f}")
    
    if avg_sim > 0.85:
        print("❌ CRITICAL: Text embeddings are too similar!")
        print("   → Your CLIP model is likely pooling padding tokens")
        print("   → Check that you're using the UPDATED forward() method")
    elif avg_sim > 0.75:
        print("⚠️  WARNING: Text embeddings have moderate similarity")
        print("   → This might be okay, but verify pooling logic")
    else:
        print("✅ PASS: Text embeddings are sufficiently distinct")
    
    # Check logits structure
    print(f"\n📊 Logits diagonal (should be highest): {torch.diag(logits).cpu().tolist()}")
    print(f"   Logits off-diagonal (should be lower):")
    for i in range(4):
        off_diag = [logits[i, j].item() for j in range(4) if i != j]
        print(f"   Row {i}: {off_diag}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # First, compare pooling methods to diagnose
    pooling_results = compare_pooling_methods()
    
    # Then test the full model
    run_full_model_sanity_check()