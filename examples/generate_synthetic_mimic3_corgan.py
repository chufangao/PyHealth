#!/usr/bin/env python3
"""
Generate synthetic MIMIC-III patients using a trained CorGAN checkpoint.
Uses Variable top-K sampling to maintain natural variation in code counts.
"""

import os
import sys
sys.path.insert(0, '/u/jalenj4/PyHealth-Medgan-Corgan-Port')
import argparse
import torch
import numpy as np
import pandas as pd
from pyhealth.models.generators.corgan import CorGANAutoencoder, CorGAN8LayerAutoencoder, CorGANGenerator, CorGANDiscriminator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic patients using trained CorGAN")
    parser.add_argument("--checkpoint", required=True, help="Path to trained CorGAN checkpoint (.pth)")
    parser.add_argument("--vocab", required=True, help="Path to ICD-9 vocabulary file (.txt)")
    parser.add_argument("--binary_matrix", required=True, help="Path to training binary matrix (.npy)")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of synthetic patients to generate")
    parser.add_argument("--mean_k", type=float, default=13, help="Mean K for Variable top-K sampling")
    parser.add_argument("--std_k", type=float, default=5, help="Std dev K for Variable top-K sampling")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vocabulary
    print(f"Loading vocabulary from {args.vocab}")
    with open(args.vocab, 'r') as f:
        code_vocab = [line.strip() for line in f]
    print(f"Loaded {len(code_vocab)} ICD-9 codes")

    # Load binary matrix to get architecture dimensions
    print(f"Loading binary matrix from {args.binary_matrix}")
    binary_matrix = np.load(args.binary_matrix)
    n_codes = binary_matrix.shape[1]
    print(f"Binary matrix shape: {binary_matrix.shape}")
    print(f"Real data avg codes/patient: {binary_matrix.sum(axis=1).mean():.2f}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Detect architecture from checkpoint
    # Check if this is 8-layer by looking at state dict keys
    state_keys = checkpoint['autoencoder_state_dict'].keys()
    is_8layer = any('encoder.18' in k or 'encoder.21' in k for k in state_keys)  # 8-layer has more layers

    # Initialize CorGAN components with correct architecture
    print("Initializing CorGAN model components...")
    if n_codes == 6955:
        if is_8layer:
            autoencoder = CorGAN8LayerAutoencoder(feature_size=n_codes).to(device)
            print("Detected 8-layer architecture")
        else:
            # Assume adaptive pooling
            autoencoder = CorGANAutoencoder(
                feature_size=n_codes,
                use_adaptive_pooling=True
            ).to(device)
            print("Detected 6-layer + adaptive pooling architecture")
    else:
        autoencoder = CorGANAutoencoder(
            feature_size=n_codes,
            use_adaptive_pooling=False
        ).to(device)
        print(f"Using standard 6-layer architecture for {n_codes} codes")

    generator = CorGANGenerator(latent_dim=128, hidden_dim=128).to(device)
    discriminator = CorGANDiscriminator(input_dim=n_codes, hidden_dim=256).to(device)

    # Load trained weights
    autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    autoencoder.eval()
    generator.eval()
    discriminator.eval()
    print("Model loaded successfully")

    # Generate synthetic patients
    print(f"\nGenerating {args.n_samples} synthetic patients...")

    with torch.no_grad():
        # Generate random noise
        z = torch.randn(args.n_samples, 128, device=device)
        # Generate latent codes
        generated_latent = generator(z)
        # Decode to probabilities
        synthetic_probs = autoencoder.decode(generated_latent)

        # Trim or pad if needed
        if synthetic_probs.shape[1] > n_codes:
            synthetic_probs = synthetic_probs[:, :n_codes]
        elif synthetic_probs.shape[1] < n_codes:
            padding = torch.zeros(synthetic_probs.shape[0], n_codes - synthetic_probs.shape[1], device=device)
            synthetic_probs = torch.cat([synthetic_probs, padding], dim=1)

    probs = synthetic_probs.cpu().numpy()

    # Apply Variable top-K sampling
    print(f"Applying Variable top-K sampling (μ={args.mean_k}, σ={args.std_k})...")
    binary_matrix_synthetic = np.zeros_like(probs)

    for i in range(args.n_samples):
        # Sample K from normal distribution, clip to reasonable range
        k = int(np.clip(np.random.normal(args.mean_k, args.std_k), 1, 50))
        # Get indices of top-K probabilities
        top_k_indices = np.argsort(probs[i])[-k:]
        binary_matrix_synthetic[i, top_k_indices] = 1

    # Calculate statistics
    avg_codes = binary_matrix_synthetic.sum(axis=1).mean()
    std_codes = binary_matrix_synthetic.sum(axis=1).std()
    min_codes = binary_matrix_synthetic.sum(axis=1).min()
    max_codes = binary_matrix_synthetic.sum(axis=1).max()
    sparsity = (binary_matrix_synthetic == 0).mean()

    print(f"\nSynthetic data statistics:")
    print(f"  Avg codes per patient: {avg_codes:.2f} ± {std_codes:.2f}")
    print(f"  Range: [{min_codes:.0f}, {max_codes:.0f}]")
    print(f"  Sparsity: {sparsity:.4f}")

    # Check heterogeneity
    unique_profiles = len(set(tuple(row) for row in binary_matrix_synthetic))
    print(f"  Unique patient profiles: {unique_profiles}/{args.n_samples} ({unique_profiles/args.n_samples*100:.1f}%)")

    # Convert to CSV format (SUBJECT_ID, ICD9_CODE)
    print(f"\nConverting to CSV format...")
    records = []
    for patient_idx in range(args.n_samples):
        patient_id = f"SYNTHETIC_{patient_idx+1:06d}"
        code_indices = np.where(binary_matrix_synthetic[patient_idx] == 1)[0]

        for code_idx in code_indices:
            records.append({
                'SUBJECT_ID': patient_id,
                'ICD9_CODE': code_vocab[code_idx]
            })

    df = pd.DataFrame(records)
    print(f"Created {len(df)} diagnosis records for {args.n_samples} patients")

    # Save to CSV
    print(f"\nSaving to {args.output}")
    df.to_csv(args.output, index=False)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved {file_size_mb:.1f} MB")

    print("\n✓ Generation complete!")
    print(f"Output: {args.output}")


if __name__ == '__main__':
    main()
