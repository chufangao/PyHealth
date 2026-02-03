import functools
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import time

from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel
from pyhealth.tokenizer import Tokenizer


class CorGANDataset(Dataset):
    """Dataset wrapper for CorGAN training"""
    
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data.astype(np.float32)
        self.sampleSize = data.shape[0]
        self.featureSize = data.shape[1]

    def return_data(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        sample = np.clip(sample, 0, 1)

        if self.transform:
            pass

        return torch.from_numpy(sample)


class CorGANAutoencoder(nn.Module):
    """Autoencoder for CorGAN - uses 1D convolutions to capture correlations"""

    def __init__(self, feature_size: int, latent_dim: int = 128, use_adaptive_pooling: bool = False):
        super(CorGANAutoencoder, self).__init__()
        self.feature_size = feature_size
        self.latent_dim = latent_dim
        self.use_adaptive_pooling = use_adaptive_pooling
        n_channels_base = 4
        
        # calculate the size after convolutions
        # input: (batch, 1, feature_size)
        # conv1: kernel=5, stride=2 -> (batch, 4, (feature_size-4)//2)
        # conv2: kernel=5, stride=2 -> (batch, 8, ((feature_size-4)//2-4)//2)
        # conv3: kernel=5, stride=3 -> (batch, 16, (((feature_size-4)//2-4)//2-4)//3)
        # conv4: kernel=5, stride=3 -> (batch, 32, ((((feature_size-4)//2-4)//2-4)//3-4)//3)
        # conv5: kernel=5, stride=3 -> (batch, 64, (((((feature_size-4)//2-4)//2-4)//3-4)//3-4)//3)
        # conv6: kernel=8, stride=1 -> (batch, 128, ((((((feature_size-4)//2-4)//2-4)//3-4)//3-4)//3-7))
        
        # rough estimate for latent size
        latent_size = max(1, feature_size // 100)  # ensure at least 1
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_channels_base, kernel_size=5, stride=2, padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=n_channels_base, out_channels=2 * n_channels_base, kernel_size=5, stride=2, padding=0,
                      dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=2 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=4 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(8 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=8 * n_channels_base, out_channels=16 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(16 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=16 * n_channels_base, out_channels=32 * n_channels_base, kernel_size=8, stride=1,
                      padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.Tanh(),
        )

        # decoder - exact match to synthEHRella (wgancnnmimic.py lines 200-228)
        # Kernel sizes: [5, 5, 7, 7, 7, 3]
        # Strides: [1, 4, 4, 3, 2, 2]
        # Activations: ReLU (not LeakyReLU)
        # Note: First layer has NO BatchNorm
        decoder_layers = [
            nn.ConvTranspose1d(in_channels=32 * n_channels_base, out_channels=16 * n_channels_base, kernel_size=5, stride=1,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=5, stride=4,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(8 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=8 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=7, stride=4,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=4 * n_channels_base, out_channels=2 * n_channels_base, kernel_size=7, stride=3,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=2 * n_channels_base, out_channels=n_channels_base, kernel_size=7, stride=2,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=n_channels_base, out_channels=1, kernel_size=3, stride=2,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
        ]

        # Add adaptive pooling if enabled (for variable vocabulary sizes)
        if self.use_adaptive_pooling:
            decoder_layers.append(nn.AdaptiveAvgPool1d(output_size=feature_size))

        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # add channel dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Squeeze only the channel dimension (dim=1), not the batch dimension
        if decoded.dim() == 3 and decoded.shape[1] == 1:
            decoded = decoded.squeeze(1)
        return decoded

    def decode(self, x):
        # x shape: (batch, 128) from generator - unsqueeze for CNN decoder
        if x.dim() == 2:
            x = x.unsqueeze(2)  # (batch, 128, 1)
        decoded = self.decoder(x)  # (batch, 1, output_len)
        if decoded.dim() == 3 and decoded.shape[1] == 1:
            decoded = decoded.squeeze(1)  # (batch, output_len)
        return decoded


class CorGAN8LayerAutoencoder(nn.Module):
    """
    8-Layer CNN Autoencoder for CorGAN - designed for 6,955 codes.

    Extends the original 6-layer architecture to support larger vocabularies
    without adaptive pooling. The encoder compresses 6,955 codes down to a
    latent space of size (128, 1), then the decoder reconstructs exactly 6,955.

    This is an experimental architecture designed to test whether native
    dimension matching (no adaptive pooling) produces better synthetic data
    quality compared to the 6-layer + adaptive pooling approach.

    Args:
        feature_size: Must be 6955 (architecture is hardcoded for this size)
        latent_dim: Latent dimension (default: 128)
    """

    def __init__(self, feature_size: int = 6955, latent_dim: int = 128):
        super(CorGAN8LayerAutoencoder, self).__init__()
        assert feature_size == 6955, "8-layer architecture only supports 6955 codes"

        self.feature_size = feature_size
        self.latent_dim = latent_dim

        # Encoder: 6955 → 1 (8 layers)
        self.encoder = nn.Sequential(
            # Layer 1: 6955 → 3476
            nn.Conv1d(1, 4, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 3476 → 1736
            nn.Conv1d(4, 8, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 1736 → 578
            nn.Conv1d(8, 16, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 578 → 192
            nn.Conv1d(16, 32, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5: 192 → 63
            nn.Conv1d(32, 64, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 6: 63 → 20 [NEW]
            nn.Conv1d(64, 96, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm1d(96),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 7: 20 → 4 [NEW]
            nn.Conv1d(96, 112, kernel_size=5, stride=4, padding=0),
            nn.BatchNorm1d(112),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 8: 4 → 1 [NEW]
            nn.Conv1d(112, 128, kernel_size=4, stride=1, padding=0),
            nn.Tanh(),
        )

        # Decoder: 1 → 6955 (8 layers)
        self.decoder = nn.Sequential(
            # Layer 1: 1 → 4 (NO BatchNorm on first layer)
            nn.ConvTranspose1d(128, 112, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),

            # Layer 2: 4 → 20
            nn.ConvTranspose1d(112, 96, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm1d(96),
            nn.ReLU(),

            # Layer 3: 20 → 63
            nn.ConvTranspose1d(96, 64, kernel_size=6, stride=3, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Layer 4: 63 → 192
            nn.ConvTranspose1d(64, 32, kernel_size=6, stride=3, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Layer 5: 192 → 578
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            # Layer 6: 578 → 1736
            nn.ConvTranspose1d(16, 8, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            # Layer 7: 1736 → 3476
            nn.ConvTranspose1d(8, 4, kernel_size=6, stride=2, padding=0),
            nn.BatchNorm1d(4),
            nn.ReLU(),

            # Layer 8: 3476 → 6955
            nn.ConvTranspose1d(4, 1, kernel_size=5, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Squeeze only channel dimension
        if decoded.dim() == 3 and decoded.shape[1] == 1:
            decoded = decoded.squeeze(1)
        return decoded

    def decode(self, x):
        """Decode latent representation from generator."""
        if x.dim() == 2:
            x = x.unsqueeze(2)  # (batch, 128, 1)
        decoded = self.decoder(x)
        if decoded.dim() == 3 and decoded.shape[1] == 1:
            decoded = decoded.squeeze(1)
        return decoded


class CorGANLinearAutoencoder(nn.Module):
    """
    Linear autoencoder for CorGAN - simpler than CNN, appropriate for unordered codes.

    This variant replaces the CNN autoencoder with a simple linear architecture,
    which is more appropriate for unordered medical codes (ICD-9) where spatial
    locality doesn't exist. Based on:
    - SynthEHRella's commented linear decoder alternative (line 229 in wgancnnmimic.py)
    - MedGAN's proven linear architecture (achieves 10.66 codes/patient)
    - Simpler gradient flow to address mode collapse

    The core CorGAN components are preserved:
    - WGAN training with Wasserstein loss
    - Generator with residual connections
    - Discriminator with minibatch averaging

    This architecture is referred to as "CorGAN-Linear" to distinguish it from
    the original CNN-based CorGAN while maintaining the core WGAN design.
    """

    def __init__(self, feature_size: int, latent_dim: int = 128):
        super(CorGANLinearAutoencoder, self).__init__()
        self.feature_size = feature_size
        self.latent_dim = latent_dim

        # Encoder: feature_size → latent_dim
        # Use ReLU+BatchNorm (V11 achieved 4.49 codes, best linear result)
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim)
        )

        # Decoder: latent_dim → feature_size
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, feature_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass for autoencoder training.

        Args:
            x: Input tensor of shape (batch, feature_size)

        Returns:
            Decoded tensor of shape (batch, feature_size)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def decode(self, x):
        """
        Decode latent representation from generator.

        Args:
            x: Latent tensor from generator of shape (batch, latent_dim)

        Returns:
            Decoded tensor of shape (batch, feature_size)
        """
        return self.decoder(x)


class CorGANGenerator(nn.Module):
    """
    Generator for CorGAN - MLP with residual connections

    Architecture matches synthEHRella exactly (wgancnnmimic.py lines 242-263)
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128):
        super(CorGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Layer 1
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, eps=0.001, momentum=0.01)
        self.activation1 = nn.ReLU()

        # Layer 2
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim, eps=0.001, momentum=0.01)
        self.activation2 = nn.Tanh()

    def forward(self, x):
        # Layer 1 with residual connection
        residual = x
        temp = self.activation1(self.bn1(self.linear1(x)))
        out1 = temp + residual

        # Layer 2 with residual connection
        residual = out1
        temp = self.activation2(self.bn2(self.linear2(out1)))
        out2 = temp + residual

        return out2


class CorGANDiscriminator(nn.Module):
    """
    Discriminator for CorGAN - MLP with minibatch averaging

    Architecture matches synthEHRella exactly (wgancnnmimic.py lines 265-296):
    - 4 linear layers: input → 256 → 256 → 256 → 1
    - ReLU activations
    - No sigmoid (WGAN uses unbounded critic outputs)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, minibatch_averaging: bool = True):
        super(CorGANDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.minibatch_averaging = minibatch_averaging

        # adjust input dimension for minibatch averaging
        ma_coef = 1
        if minibatch_averaging:
            ma_coef = ma_coef * 2
        model_input_dim = ma_coef * input_dim

        # 4-layer architecture matching synthEHRella exactly
        self.model = nn.Sequential(
            nn.Linear(model_input_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, int(self.hidden_dim)),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, int(self.hidden_dim)),
            nn.ReLU(True),
            nn.Linear(int(self.hidden_dim), 1)
            # No sigmoid - WGAN uses unbounded critic outputs
        )

    def forward(self, x):
        if self.minibatch_averaging:
            # minibatch averaging: concatenate batch mean to each sample
            x_mean = torch.mean(x, dim=0).repeat(x.shape[0], 1)
            x = torch.cat((x, x_mean), dim=1)

        output = self.model(x)
        return output


def weights_init(m):
    """
    Custom weight initialization (synthEHRella implementation)

    Reference: synthEHRella wgancnnmimic.py lines 363-377
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def autoencoder_loss(x_output, y_target):
    """
    Autoencoder reconstruction loss (synthEHRella implementation)

    This implementation is equivalent to torch.nn.BCELoss(reduction='sum') / batch_size
    As our matrix is too sparse, first we will take a sum over the features and then
    do the mean over the batch.

    WARNING: This is NOT equivalent to torch.nn.BCELoss(reduction='mean') as the latter
    means over both features and batches.

    Reference: synthEHRella wgancnnmimic.py lines 312-323
    """
    epsilon = 1e-12
    term = y_target * torch.log(x_output + epsilon) + (1. - y_target) * torch.log(1. - x_output + epsilon)
    loss = torch.mean(-torch.sum(term, 1), 0)
    return loss


def discriminator_accuracy(predicted, y_true):
    """Calculate discriminator accuracy"""
    predicted = (predicted >= 0.5).float()
    accuracy = (predicted == y_true).float().mean()
    return accuracy.item()


class CorGAN(BaseModel):
    """
    CorGAN: Correlation-capturing Generative Adversarial Network
    
    Uses CNNs to capture correlations between adjacent medical features by combining
    Convolutional GANs with Convolutional Autoencoders.
    
    Args:
        dataset: PyHealth dataset object
        feature_keys: List of feature keys to use
        label_key: Label key (not used in unsupervised generation)
        mode: Training mode (not used in GAN context)
        latent_dim: Dimensionality of latent space
        hidden_dim: Hidden dimension for networks
        batch_size: Training batch size
        n_epochs: Number of training epochs
        n_epochs_pretrain: Number of autoencoder pretraining epochs
        lr: Learning rate
        weight_decay: Weight decay for optimization
        b1: Beta1 for Adam optimizer
        b2: Beta2 for Adam optimizer
        n_iter_D: Number of discriminator iterations per generator iteration
        clamp_lower: Lower bound for weight clipping
        clamp_upper: Upper bound for weight clipping
        minibatch_averaging: Whether to use minibatch averaging in discriminator
        **kwargs: Additional arguments
    
    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> dataset = MIMIC3Dataset(...)
        >>> model = CorGAN(dataset=dataset, feature_keys=["conditions"])
        >>> model.fit()
        >>> synthetic_data = model.generate(n_samples=50000)
    """
    
    def __init__(
        self,
        dataset: BaseDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str = "generation",
        latent_dim: int = 128,
        hidden_dim: int = 128,
        batch_size: int = 512,
        n_epochs: int = 1000,
        n_epochs_pretrain: int = 1,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        b1: float = 0.9,
        b2: float = 0.999,
        n_iter_D: int = 5,
        clamp_lower: float = -0.01,
        clamp_upper: float = 0.01,
        minibatch_averaging: bool = True,
        **kwargs
    ):
        super(CorGAN, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            **kwargs
        )
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_epochs_pretrain = n_epochs_pretrain
        self.lr = lr
        self.weight_decay = weight_decay
        self.b1 = b1
        self.b2 = b2
        self.n_iter_D = n_iter_D
        self.clamp_lower = clamp_lower
        self.clamp_upper = clamp_upper
        self.minibatch_averaging = minibatch_averaging
        
        # build unified vocabulary for all feature keys
        self.global_vocab = self._build_global_vocab(dataset, feature_keys)
        self.input_dim = len(self.global_vocab)
        self.tokenizer = Tokenizer(tokens=self.global_vocab, special_tokens=[])
        
        # initialize components
        # Determine if adaptive pooling is needed (only for non-standard vocabulary sizes)
        use_adaptive_pooling = (self.input_dim != 1071)
        self.autoencoder = CorGANAutoencoder(
            feature_size=self.input_dim,
            latent_dim=latent_dim,
            use_adaptive_pooling=use_adaptive_pooling
        )
        self.autoencoder_decoder = self.autoencoder.decoder  # separate decoder for generator
        
        self.generator = CorGANGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        
        self.discriminator = CorGANDiscriminator(
            input_dim=self.input_dim,
            hidden_dim=256,  # Match synthEHRella exactly (not hidden_dim * 2)
            minibatch_averaging=minibatch_averaging
        )
        
        # apply custom weight initialization
        self._init_weights()
        
        # setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # setup optimizers
        g_params = [
            {'params': self.generator.parameters()},
            {'params': self.autoencoder_decoder.parameters(), 'lr': 1e-4}
        ]
        self.optimizer_G = torch.optim.Adam(g_params, lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_A = torch.optim.Adam(self.autoencoder.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        
        # setup tensors for training
        self.one = torch.tensor(1.0, device=self.device)
        self.mone = torch.tensor(-1.0, device=self.device)
    
    def _build_global_vocab(self, dataset: BaseDataset, feature_keys: List[str]) -> List[str]:
        """Build unified vocabulary across all feature keys"""
        global_vocab = set()
        
        # collect all unique codes from all patients and feature keys
        for patient_id in dataset.patients:
            patient = dataset.patients[patient_id]
            for feature_key in feature_keys:
                if feature_key in patient:
                    for visit in patient[feature_key]:
                        if isinstance(visit, list):
                            global_vocab.update(visit)
                        else:
                            global_vocab.add(visit)
        
        return sorted(list(global_vocab))
    
    def _encode_patient_record(self, record: Dict) -> torch.Tensor:
        """Encode a patient record to binary vector"""
        # create binary vector
        binary_vector = np.zeros(self.input_dim, dtype=np.float32)
        
        for feature_key in self.feature_keys:
            if feature_key in record:
                for visit in record[feature_key]:
                    if isinstance(visit, list):
                        for code in visit:
                            if code in self.global_vocab:
                                idx = self.global_vocab.index(code)
                                binary_vector[idx] = 1.0
                    else:
                        if visit in self.global_vocab:
                            idx = self.global_vocab.index(visit)
                            binary_vector[idx] = 1.0
        
        return torch.from_numpy(binary_vector)
    
    def _init_weights(self):
        """Initialize network weights"""
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.autoencoder.apply(weights_init)
    
    def _extract_features_from_batch(self, batch_data, device: torch.device) -> torch.Tensor:
        """Extract features from batch data"""
        features = []
        for patient_id in batch_data:
            patient = self.dataset.patients[patient_id]
            feature_vector = self._encode_patient_record(patient)
            features.append(feature_vector)
        
        return torch.stack(features).to(device)
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass - not used in GAN context"""
        raise NotImplementedError("Forward pass not implemented for GAN models")
    
    def fit(self, train_dataloader: Optional[DataLoader] = None):
        """Train the CorGAN model"""
        print("Starting CorGAN training...")
        
        # create dataset and dataloader
        if train_dataloader is None:
            # create binary matrix from dataset
            data_matrix = []
            for patient_id in self.dataset.patients:
                patient = self.dataset.patients[patient_id]
                feature_vector = self._encode_patient_record(patient)
                data_matrix.append(feature_vector.numpy())
            
            data_matrix = np.array(data_matrix)
            dataset = CorGANDataset(data=data_matrix)
            
            sampler = torch.utils.data.sampler.RandomSampler(
                data_source=dataset, replacement=True
            )
            train_dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size,
                shuffle=False, 
                num_workers=0, 
                drop_last=True, 
                sampler=sampler
            )
        
        # pretrain autoencoder
        print(f"Pretraining autoencoder for {self.n_epochs_pretrain} epochs...")
        for epoch_pre in range(self.n_epochs_pretrain):
            for i, samples in enumerate(train_dataloader):
                # configure input
                real_samples = samples.to(self.device)
                
                # generate a batch of images
                recons_samples = self.autoencoder(real_samples)
                
                # loss measures autoencoder's ability to reconstruct
                a_loss = autoencoder_loss(recons_samples, real_samples)
                
                # reset gradients
                self.optimizer_A.zero_grad()
                a_loss.backward()
                self.optimizer_A.step()
                
                if i % 100 == 0:
                    print(f"[Epoch {epoch_pre + 1}/{self.n_epochs_pretrain}] [Batch {i}/{len(train_dataloader)}] [A loss: {a_loss.item():.3f}]")
        
        # adversarial training
        print(f"Starting adversarial training for {self.n_epochs} epochs...")
        gen_iterations = 0
        
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            
            for i, samples in enumerate(train_dataloader):
                # adversarial ground truths
                valid = torch.ones(samples.shape[0], device=self.device)
                fake = torch.zeros(samples.shape[0], device=self.device)
                
                # configure input
                real_samples = samples.to(self.device)
                
                # sample noise as generator input
                z = torch.randn(samples.shape[0], self.latent_dim, device=self.device)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                for p in self.discriminator.parameters():
                    p.requires_grad = True
                
                # train the discriminator n_iter_D times
                if gen_iterations < 25 or gen_iterations % 500 == 0:
                    n_iter_D = 100
                else:
                    n_iter_D = self.n_iter_D
                
                j = 0
                while j < n_iter_D:
                    j += 1
                    
                    # clamp parameters to a cube
                    for p in self.discriminator.parameters():
                        p.data.clamp_(self.clamp_lower, self.clamp_upper)
                    
                    # reset gradients of discriminator
                    self.optimizer_D.zero_grad()
                    
                    errD_real = torch.mean(self.discriminator(real_samples), dim=0)
                    errD_real.backward(self.one)
                    
                    # sample noise as generator input
                    z = torch.randn(samples.shape[0], self.latent_dim, device=self.device)
                    
                    # generate a batch of images
                    fake_samples = self.generator(z)
                    fake_samples = torch.squeeze(self.autoencoder_decoder(fake_samples.unsqueeze(dim=2)))
                    
                    errD_fake = torch.mean(self.discriminator(fake_samples.detach()), dim=0)
                    errD_fake.backward(self.mone)
                    errD = errD_real - errD_fake
                    
                    # optimizer step
                    self.optimizer_D.step()
                
                # -----------------
                #  Train Generator
                # -----------------
                
                for p in self.discriminator.parameters():
                    p.requires_grad = False
                
                # zero grads
                self.optimizer_G.zero_grad()
                
                # sample noise as generator input
                z = torch.randn(samples.shape[0], self.latent_dim, device=self.device)
                
                # generate a batch of images
                fake_samples = self.generator(z)
                fake_samples = torch.squeeze(self.autoencoder_decoder(fake_samples.unsqueeze(dim=2)))
                
                # loss measures generator's ability to fool the discriminator
                errG = torch.mean(self.discriminator(fake_samples), dim=0)
                errG.backward(self.one)
                
                # optimizer step
                self.optimizer_G.step()
                gen_iterations += 1
            
            # end of epoch
            epoch_end = time.time()
            print(f"[Epoch {epoch + 1}/{self.n_epochs}] [Batch {i}/{len(train_dataloader)}] "
                  f"Loss_D: {errD.item():.3f} Loss_G: {errG.item():.3f} "
                  f"Loss_D_real: {errD_real.item():.3f} Loss_D_fake: {errD_fake.item():.3f}")
            print(f"Epoch time: {epoch_end - epoch_start:.2f} seconds")
        
        print("Training completed!")
    
    def generate(self, n_samples: int, device: torch.device = None) -> torch.Tensor:
        """Generate synthetic data"""
        if device is None:
            device = self.device
        
        # set models to eval mode
        self.generator.eval()
        self.autoencoder_decoder.eval()
        
        # generate samples
        gen_samples = np.zeros((n_samples, self.input_dim), dtype=np.float32)
        n_batches = int(n_samples / self.batch_size)
        
        with torch.no_grad():
            for i in range(n_batches):
                # sample noise as generator input
                z = torch.randn(self.batch_size, self.latent_dim, device=device)
                gen_samples_tensor = self.generator(z)
                gen_samples_decoded = torch.squeeze(self.autoencoder_decoder(gen_samples_tensor.unsqueeze(dim=2)))
                gen_samples[i * self.batch_size:(i + 1) * self.batch_size, :] = gen_samples_decoded.cpu().data.numpy()
        
        # handle remaining samples
        remaining = n_samples % self.batch_size
        if remaining > 0:
            z = torch.randn(remaining, self.latent_dim, device=device)
            gen_samples_tensor = self.generator(z)
            gen_samples_decoded = torch.squeeze(self.autoencoder_decoder(gen_samples_tensor.unsqueeze(dim=2)))
            gen_samples[n_batches * self.batch_size:, :] = gen_samples_decoded.cpu().data.numpy()
        
        # binarize output
        gen_samples[gen_samples >= 0.5] = 1.0
        gen_samples[gen_samples < 0.5] = 0.0
        
        return torch.from_numpy(gen_samples)
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'autoencoder_state_dict': self.autoencoder.state_dict(),
            'autoencoder_decoder_state_dict': self.autoencoder_decoder.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'optimizer_A_state_dict': self.optimizer_A.state_dict(),
            'global_vocab': self.global_vocab,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        self.autoencoder_decoder.load_state_dict(checkpoint['autoencoder_decoder_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.optimizer_A.load_state_dict(checkpoint['optimizer_A_state_dict'])
        
        self.global_vocab = checkpoint['global_vocab']
        self.input_dim = checkpoint['input_dim']
        self.latent_dim = checkpoint['latent_dim'] 