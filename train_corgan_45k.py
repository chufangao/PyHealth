#!/usr/bin/env python3
"""
Train CorGAN on 45k MIMIC-III train split with 1k holdout.

Uses V3 hyperparameters (matching original cor-gan):
- batch_size=512 (original default)
- lr=0.001 (original default)
- UNFROZEN decoder during GAN phase (joint training at lr=1e-4)
- ae_epochs=1 (minimal pretraining)
- gan_epochs=500
- Dynamic discriminator schedule (100 iters for first 25 gens + every 500)
"""

import os
import sys
sys.path.insert(0, '/u/jalenj4/PyHealth-Medgan-Corgan-Port')
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from pyhealth.models.generators.corgan import (
    CorGANAutoencoder, CorGAN8LayerAutoencoder, CorGANGenerator, CorGANDiscriminator,
    CorGANDataset, weights_init
)


def load_mimic3_data(data_path):
    """Load MIMIC-III diagnosis data from train split."""
    print(f"Loading MIMIC-III data from: {data_path}")

    diagnoses_df = pd.read_csv(os.path.join(data_path, "DIAGNOSES_ICD.csv"))
    diagnoses_df = diagnoses_df.dropna(subset=['ICD9_CODE'])
    diagnoses_df['ICD9_CODE'] = diagnoses_df['ICD9_CODE'].astype(str)

    patient_codes = defaultdict(set)
    for _, row in diagnoses_df.iterrows():
        patient_codes[row['SUBJECT_ID']].add(row['ICD9_CODE'])

    all_codes = sorted(set().union(*patient_codes.values()))
    code_to_idx = {code: idx for idx, code in enumerate(all_codes)}

    binary_matrix = np.zeros((len(patient_codes), len(all_codes)))
    for i, (patient_id, codes) in enumerate(sorted(patient_codes.items())):
        for code in codes:
            binary_matrix[i, code_to_idx[code]] = 1

    # Pad to 1,071 codes for synthEHRella architecture compatibility
    target_n_codes = 1071
    if len(all_codes) < target_n_codes:
        padding = np.zeros((len(patient_codes), target_n_codes - len(all_codes)))
        binary_matrix = np.hstack([binary_matrix, padding])
        print(f"Padded from {len(all_codes)} to {target_n_codes} codes for architecture compatibility")

    print(f"Loaded {len(patient_codes)} patients with {len(all_codes)} unique ICD-9 codes")
    print(f"Final matrix shape: {binary_matrix.shape}")
    print(f"Data sparsity: {(binary_matrix == 0).mean():.4f}")
    print(f"Avg codes per patient: {binary_matrix.sum(axis=1).mean():.2f}")

    return binary_matrix, all_codes


def evaluate_generation_quality(autoencoder, generator, n_codes, device, latent_dim, n_samples=100):
    """Generate samples and evaluate quality."""
    autoencoder.eval()
    generator.eval()

    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        generated_latent = generator(z)
        decoded = autoencoder.decode(generated_latent)

        # Trim or pad if needed (should not happen with CNN but keep for safety)
        if decoded.shape[1] > n_codes:
            decoded = decoded[:, :n_codes]
        elif decoded.shape[1] < n_codes:
            padding = torch.zeros(decoded.shape[0], n_codes - decoded.shape[1], device=device)
            decoded = torch.cat([decoded, padding], dim=1)

        probs = decoded
        binary = (probs >= 0.5).float().cpu().numpy()

        avg_codes = binary.sum(axis=1).mean()
        sparsity = (binary == 0).mean()
        prob_mean = probs.mean().item()
        prob_std = probs.std().item()

    return {
        'avg_codes_per_patient': avg_codes,
        'sparsity': sparsity,
        'prob_mean': prob_mean,
        'prob_std': prob_std,
        'latent_mean': generated_latent.mean().item(),
        'latent_std': generated_latent.std().item()
    }


def train_corgan(binary_matrix, n_codes, device, args):
    """Train CorGAN model with V2 hyperparameters."""

    # Determine architecture based on vocabulary size and user choice
    if n_codes == 1071:
        # Option B: Standard 6-layer CNN (no adaptive pooling)
        autoencoder = CorGANAutoencoder(
            feature_size=n_codes,
            latent_dim=args.latent_dim,
            use_adaptive_pooling=False
        ).to(device)
        print(f"✓ Using standard 6-layer CNN for {n_codes} codes (matches original CorGAN)")

    elif n_codes == 6955:
        # Option C: User chooses architecture
        if args.architecture == "adaptive":
            autoencoder = CorGANAutoencoder(
                feature_size=n_codes,
                latent_dim=args.latent_dim,
                use_adaptive_pooling=True
            ).to(device)
            print(f"⚠️  Using 6-layer CNN + adaptive pooling for {n_codes} codes")

        elif args.architecture == "8layer":
            autoencoder = CorGAN8LayerAutoencoder(
                feature_size=n_codes,
                latent_dim=args.latent_dim
            ).to(device)
            print(f"⚠️  Using 8-layer CNN for {n_codes} codes (experimental architecture)")

    else:
        raise ValueError(f"Unsupported vocabulary size: {n_codes}. Only 1071 or 6955 supported.")

    generator = CorGANGenerator(args.latent_dim, hidden_dim=128).to(device)
    discriminator = CorGANDiscriminator(input_dim=n_codes, hidden_dim=256).to(device)

    # Initialize weights
    for model in [autoencoder, generator, discriminator]:
        model.apply(weights_init)

    # Dataset
    dataset = CorGANDataset(binary_matrix)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Train autoencoder
    print(f"\nTraining autoencoder for {args.ae_epochs} epochs...")
    ae_optimizer = torch.optim.Adam(
        autoencoder.parameters(),
        lr=args.lr,
        betas=(args.b1, args.b2),
        weight_decay=args.weight_decay
    )
    ae_losses = []

    for epoch in range(args.ae_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"AE Epoch {epoch+1}/{args.ae_epochs}"):
            batch = batch.to(device)
            reconstructed = autoencoder(batch)

            # Trim or pad if needed
            if reconstructed.shape[1] > n_codes:
                reconstructed = reconstructed[:, :n_codes]
            elif reconstructed.shape[1] < n_codes:
                padding = torch.zeros(reconstructed.shape[0], n_codes - reconstructed.shape[1], device=device)
                reconstructed = torch.cat([reconstructed, padding], dim=1)

            loss = F.binary_cross_entropy(reconstructed, batch)

            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        ae_losses.append(avg_loss)
        print(f"AE Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # Train GAN
    print(f"\nTraining GAN for {args.gan_epochs} epochs...")
    print("Decoder training: UNFROZEN (V3 - matching original cor-gan)")

    # V3 FIX #1: Unfreeze decoder and train jointly with generator
    g_params = [
        {'params': generator.parameters()},
        {'params': autoencoder.decoder.parameters(), 'lr': 1e-4}
    ]
    g_optimizer = torch.optim.Adam(
        g_params,
        lr=args.lr,
        betas=(args.b1, args.b2)
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(args.b1, args.b2),
        weight_decay=args.weight_decay
    )

    # Freeze encoder, unfreeze decoder
    autoencoder.encoder.eval()
    autoencoder.decoder.train()

    g_losses = []
    d_losses = []
    quality_history = []
    best_quality_score = float('-inf')
    best_checkpoint = None

    # Early stopping tracking
    epochs_without_improvement = 0
    early_stop_patience = 10
    codes_degradation_count = 0
    latent_collapse_count = 0
    prev_codes_error = float('inf')

    gen_iteration = 0  # Track total generator iterations for dynamic schedule

    for epoch in range(args.gan_epochs):
        g_loss_total = d_loss_total = 0

        for batch in tqdm(dataloader, desc=f"GAN Epoch {epoch+1}/{args.gan_epochs}"):
            batch = batch.to(device)
            batch_size = batch.shape[0]

            # V3 FIX #2: Dynamic discriminator iteration schedule
            if gen_iteration < 25 or gen_iteration % 500 == 0:
                n_iter_D = 100
                if gen_iteration % 25 == 0:  # Log only occasionally
                    print(f"\n  Gen {gen_iteration}: Using n_iter_D={n_iter_D} (stabilization)")
            else:
                n_iter_D = args.n_iter_D

            # Train discriminator
            for _ in range(n_iter_D):
                d_optimizer.zero_grad()

                real_pred = discriminator(batch)
                d_loss_real = -torch.mean(real_pred)

                z = torch.randn(batch_size, args.latent_dim, device=device)
                gen_latent = generator(z)
                fake_samples = autoencoder.decode(gen_latent)

                # Trim or pad
                if fake_samples.shape[1] > n_codes:
                    fake_samples = fake_samples[:, :n_codes]
                elif fake_samples.shape[1] < n_codes:
                    padding = torch.zeros(fake_samples.shape[0], n_codes - fake_samples.shape[1], device=device)
                    fake_samples = torch.cat([fake_samples, padding], dim=1)

                fake_pred = discriminator(fake_samples.detach())
                d_loss_fake = torch.mean(fake_pred)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()

                # Weight clipping (WGAN) - use configurable bounds
                for p in discriminator.parameters():
                    p.data.clamp_(args.clamp_lower, args.clamp_upper)

            # Train generator
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_latent = generator(z)
            fake_samples = autoencoder.decode(gen_latent)

            # Trim or pad
            if fake_samples.shape[1] > n_codes:
                fake_samples = fake_samples[:, :n_codes]
            elif fake_samples.shape[1] < n_codes:
                padding = torch.zeros(fake_samples.shape[0], n_codes - fake_samples.shape[1], device=device)
                fake_samples = torch.cat([fake_samples, padding], dim=1)

            fake_pred = discriminator(fake_samples)
            g_loss = -torch.mean(fake_pred)

            g_loss.backward()
            g_optimizer.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

            gen_iteration += 1  # Increment generator iteration counter

        avg_g_loss = g_loss_total / len(dataloader)
        avg_d_loss = d_loss_total / len(dataloader)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        print(f"GAN Epoch {epoch+1}: G={avg_g_loss:.3f}, D={avg_d_loss:.3f}")

        # Evaluate quality every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == args.gan_epochs - 1:
            quality = evaluate_generation_quality(
                autoencoder, generator, n_codes, device, args.latent_dim
            )
            quality_history.append({'epoch': epoch + 1, **quality})

            target_codes = binary_matrix.sum(axis=1).mean()
            codes_error = abs(quality['avg_codes_per_patient'] - target_codes) / target_codes
            quality_score = (1 - codes_error) * quality['latent_std']

            print(f"  Quality metrics:")
            print(f"    Avg codes/patient: {quality['avg_codes_per_patient']:.2f} (target: {target_codes:.2f})")
            print(f"    Sparsity: {quality['sparsity']:.4f}")
            print(f"    Prob mean/std: {quality['prob_mean']:.4f} / {quality['prob_std']:.4f}")
            print(f"    Latent mean/std: {quality['latent_mean']:.4f} / {quality['latent_std']:.4f}")
            print(f"    Quality score: {quality_score:.4f}")

            if quality_score > best_quality_score:
                best_quality_score = quality_score
                best_checkpoint = {
                    'epoch': epoch + 1,
                    'autoencoder_state_dict': autoencoder.state_dict(),
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'quality': quality,
                    'quality_score': quality_score
                }
                print(f"    ✓ Best quality so far! Score: {quality_score:.4f}")

            # Early stopping checks
            if quality_score <= best_quality_score * 1.001:
                epochs_without_improvement += 1
            else:
                epochs_without_improvement = 0

            codes_error = abs(quality['avg_codes_per_patient'] - target_codes)
            if codes_error > prev_codes_error * 1.05:
                codes_degradation_count += 1
            else:
                codes_degradation_count = 0
            prev_codes_error = codes_error

            if quality['latent_std'] < 0.20:
                latent_collapse_count += 1
            else:
                latent_collapse_count = 0

            should_stop = False
            stop_reason = []

            if epochs_without_improvement >= early_stop_patience:
                should_stop = True
                stop_reason.append(f"No improvement for {epochs_without_improvement * 5} epochs")

            if codes_degradation_count >= 3:
                should_stop = True
                stop_reason.append(f"Code count degraded for {codes_degradation_count * 5} epochs")

            if latent_collapse_count >= 3:
                should_stop = True
                stop_reason.append(f"Latent collapse for {latent_collapse_count * 5} epochs")

            if should_stop:
                print(f"\n{'='*70}")
                print(f"EARLY STOPPING at epoch {epoch+1}")
                for reason in stop_reason:
                    print(f"  - {reason}")
                print(f"Best quality score: {best_quality_score:.4f} at epoch {best_checkpoint['epoch']}")
                print(f"{'='*70}")
                break

    history = {
        'autoencoder_losses': ae_losses,
        'generator_losses': g_losses,
        'discriminator_losses': d_losses,
        'quality_history': quality_history
    }

    return autoencoder, generator, best_checkpoint, history


def main():
    parser = argparse.ArgumentParser(description="Train CorGAN on 45k train split")
    parser.add_argument("--data_dir", "--train_path", dest="train_path", default="/u/jalenj4/pehr_scratch/data_files_train_3digit", help="Path to train data")
    parser.add_argument("--output_dir", "--output_path", dest="output_path", default="/scratch/jalenj4/corgan_45k_results", help="Output directory")
    # V3 FIX #3: Update default hyperparameters to match original cor-gan
    parser.add_argument("--ae_epochs", type=int, default=1, help="Autoencoder pretraining epochs (V3: 1, original default)")
    parser.add_argument("--gan_epochs", type=int, default=500, help="GAN training epochs")
    parser.add_argument("--latent_dim", type=int, default=128, help="Generator latent dimension")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size (V3: 512, original default)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (V3: 0.001, original default)")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam beta1 (V3: 0.5, original default)")
    parser.add_argument("--b2", type=float, default=0.9, help="Adam beta2 (V3: 0.9, original default)")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (V3: 0.0, original default)")
    parser.add_argument("--n_iter_D", type=int, default=5, help="Discriminator iterations per generator (baseline, dynamic schedule overrides)")
    parser.add_argument("--clamp_lower", type=float, default=-0.01, help="WGAN weight clamp lower bound")
    parser.add_argument("--clamp_upper", type=float, default=0.01, help="WGAN weight clamp upper bound")
    parser.add_argument("--architecture", type=str, default="adaptive",
                        choices=["adaptive", "8layer"],
                        help="Architecture for 6955 codes: 'adaptive' (6-layer+pool) or '8layer' (8-layer CNN)")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training on: {args.train_path}")
    print(f"Output to: {args.output_path}")

    # Load train data
    binary_matrix, code_vocab = load_mimic3_data(args.train_path)
    n_patients, n_codes = binary_matrix.shape

    # Train
    autoencoder, generator, best_checkpoint, history = train_corgan(
        binary_matrix, n_codes, device, args
    )

    # Save best checkpoint
    if best_checkpoint is not None:
        checkpoint_path = os.path.join(args.output_path, "corgan_best.pth")
        torch.save(best_checkpoint, checkpoint_path)
        print(f"\nSaved best checkpoint to: {checkpoint_path}")
        print(f"Best epoch: {best_checkpoint['epoch']}")
        print(f"Best quality: {best_checkpoint['quality']}")

    # Save training history
    history_path = os.path.join(args.output_path, "training_history.npy")
    np.save(history_path, history)
    print(f"Saved training history to: {history_path}")

    # Save vocabulary
    vocab_path = os.path.join(args.output_path, "icd9_vocabulary.txt")
    with open(vocab_path, 'w') as f:
        for code in code_vocab:
            f.write(f"{code}\n")
    print(f"Saved vocabulary to: {vocab_path}")

    # Save binary matrix
    matrix_path = os.path.join(args.output_path, "icd9_binary_matrix.npy")
    np.save(matrix_path, binary_matrix)
    print(f"Saved binary matrix to: {matrix_path}")

    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
