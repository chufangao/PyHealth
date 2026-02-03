# CorGAN Architecture: DO NOT MODIFY

## ⚠️ CRITICAL WARNING

The CorGAN architecture in this file **MUST match synthEHRella exactly**.

**DO NOT CHANGE** the following without experimental validation:
- Decoder kernel sizes: [5, 5, 7, 7, 7, 3]
- Decoder strides: [1, 4, 4, 3, 2, 2]
- Decoder activations: ReLU (not LeakyReLU)
- Decoder Layer 1: NO BatchNorm (ReLU only)
- Discriminator: 4 layers, hidden_dim=256
- Autoencoder loss: sum-then-mean (not standard BCE)
- Linear init: Xavier uniform, bias=0.01

## Why This Matters

Previous "V2 architecture" changes were found to **lack experimental validation**.
We reverted to synthEHRella's exact implementation to ensure:
- ✅ Reproducibility with published results
- ✅ Research integrity
- ✅ Fair comparison with other methods

## Investigation Summary

A thorough 7-month investigation (Jun 2025 - Jan 2026) found:
- ❌ No V1 vs V2 comparison tests
- ❌ No performance metrics showing V2 is superior
- ❌ No ablation studies justifying architectural choices
- ⚠️ Changes appear arbitrary with post-hoc labeling

**RED FLAGS:**
- Comments added retroactively without justification
- References to SynthEHRella suggest this is ported code, not original research
- Critical changes (residual connections, layer removal) lack mathematical or empirical justification

## If You Need to Modify

1. **Document your changes** with experimental results showing improvement
2. **Update tests** in `tests/test_corgan_architecture.py`
3. **Get approval** from project maintainers
4. **Update this README** with justification

## Tests

Architecture is protected by:
- Unit tests: `tests/test_corgan_architecture.py`
- Inline assertions in `corgan.py` (planned)
- CI/CD: `.github/workflows/test_corgan_architecture.yml` (planned)

## Reference

- **Original implementation**: `/u/jalenj4/synthEHRella/synthEHRella/data/methods/cor-gan/Generative/corGAN/pytorch/CNN/MIMIC/wgancnnmimic.py`
- **Investigation plan**: `/u/jalenj4/.claude/plans/magical-snuggling-kazoo.md`
- **Backup of V2 version**: `pyhealth/models/generators/corgan_v2_backup.py`

## Architecture Details

### CNN Decoder (Lines 88-113)

```python
# Layer 1: 128 → 64, k=5, s=1, ReLU (NO BatchNorm!)
# Layer 2: 64 → 32, k=5, s=4, BatchNorm + ReLU
# Layer 3: 32 → 16, k=7, s=4, BatchNorm + ReLU
# Layer 4: 16 → 8, k=7, s=3, BatchNorm + ReLU
# Layer 5: 8 → 4, k=7, s=2, BatchNorm + ReLU
# Layer 6: 4 → 1, k=3, s=2, Sigmoid
```

### Discriminator (Lines 231-259)

```python
# 4 layers: input → 256 → 256 → 256 → 1
# ReLU activations
# No sigmoid (WGAN uses unbounded critic outputs)
```

### Autoencoder Loss (Lines 275-291)

```python
# Custom BCE: sum over features, then mean over batch
epsilon = 1e-12
term = y_target * torch.log(x_output + epsilon) + (1. - y_target) * torch.log(1. - x_output + epsilon)
loss = torch.mean(-torch.sum(term, 1), 0)
```

### Weight Initialization (Lines 262-274)

```python
# Linear layers: Xavier uniform, bias=0.01
torch.nn.init.xavier_uniform_(m.weight)
m.bias.data.fill_(0.01)
```

## Vocabulary Size Support

### Option B (3-digit ICD9, ~1,071 codes)
- Uses original CNN architecture (matches synthEHRella)
- **Fixed stride bug** in decoder layer 3 (stride=4, not stride=2)
- Input/output dimensions match exactly: 1,071 → 1,071
- Set `use_adaptive_pooling=False` (default for 1,071 codes)

### Option C (Full ICD9, ~6,955 codes)
- **Requires architectural modification**: Adaptive pooling layer added to decoder
- Deviation from original: CorGAN paper only validated on 1,071 codes
- Rationale: No documented approach exists for scaling CNN to larger vocabularies
  - Original synthEHRella: Uses 1,071 CNN + post-hoc mapping to 595 phecodes
  - Our approach: Adaptive pooling forces output size while preserving CNN layers
- Set `use_adaptive_pooling=True` for vocabularies != 1,071

### Why Not Linear Autoencoder?
The codebase includes `CorGANLinearAutoencoder` which works for arbitrary vocabulary sizes.
However, this implementation maintains strict adherence to CNN-based architecture from original CorGAN.
Adaptive pooling is the minimal modification to enable variable vocabulary sizes while keeping the CNN architecture.

### Stride Bug Fix (Feb 2026)
**Issue**: Decoder layer 3 had `stride=2` instead of `stride=4`
**Impact**: Option B (1,071 codes) produced incorrect output dimensions
**Fix**: Changed line 101 from `stride=2` to `stride=4` to match comment on line 90
**Validation**: Input/output shapes now match exactly for 1,071 codes

## Version History

- **Feb 2026**: Fixed stride bug (layer 3: stride 2→4), added adaptive pooling for Option C
- **Jan 2026**: Reverted to synthEHRella exact implementation
- **Jun 2025 - Jan 2026**: "V2 architecture" (found to lack validation)
- **Original**: synthEHRella implementation (published, validated)

---

**Last Updated**: 2026-02-01
**Maintainer**: CorGAN Binary Mode Team
