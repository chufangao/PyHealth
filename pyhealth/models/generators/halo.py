import numpy as np
import torch
import torch.nn as nn
from typing import List
from pyhealth.models import BaseModel
# from pyhealth.generators.halo_resources.model import HALOModel as CoreHALOModel
# from pyhealth.generators.halo_resources.config import HALOConfig
from halo_resources.model import HALOModel as CoreHALOModel
from halo_resources.config import HALOConfig


class HaloGenerator(BaseModel):
    """
    PyHealth wrapper for the HALO autoregressive EHR generator model.

    Args:
        dataset (SampleEHRDataset): PyHealth dataset with "visits" and "labels" keys.
        feature_key (str): Key in samples for visit codes. Default is "visits".
        label_key (str): Key in samples for patient labels. Default is "labels".
    """
    def __init__(self, dataset, feature_key: str = "visits", label_key: str = "labels"):
        super(HaloGenerator, self).__init__(dataset=dataset, feature_keys=[feature_key], label_key=label_key)
        self.feature_key = feature_key
        self.label_key = label_key
        # Initialize HALO core model and config
        self.config = HALOConfig()
        self.model = CoreHALOModel(self.config).to(self.device)

    def prepare_batch(self, visits_batch: List[List[int]], labels_batch: List[np.ndarray]):
        """
        Convert list of visit-code lists and label vectors into model inputs.

        Returns:
            batch_ehr: np.ndarray of shape (B, n_ctx, total_vocab_size)
            batch_mask: np.ndarray of shape (B, n_ctx-1, 1)
        """
        batch_size = len(visits_batch)
        n_ctx = self.config.n_ctx
        vocab_size = self.config.total_vocab_size
        code_vocab_size = self.config.code_vocab_size
        label_vocab_size = self.config.label_vocab_size

        batch_ehr = np.zeros((batch_size, n_ctx, vocab_size), dtype=np.float32)
        batch_mask = np.zeros((batch_size, n_ctx, 1), dtype=np.float32)
        for i, visits in enumerate(visits_batch):
            # Encode sequence of visits
            for j, v in enumerate(visits):
                batch_ehr[i, j+2, v] = 1
                batch_mask[i, j+2] = 1
            # Encode patient labels at position 1
            batch_ehr[i, 1, code_vocab_size:code_vocab_size+label_vocab_size] = labels_batch[i]
            batch_mask[i, 1] = 1
            # Add special tokens
            start_tok = code_vocab_size + label_vocab_size
            end_tok = code_vocab_size + label_vocab_size + 1
            pad_tok  = code_vocab_size + label_vocab_size + 2
            batch_ehr[i, 0, start_tok] = 1
            end_pos = len(visits) + 1
            if end_pos < n_ctx:
                batch_ehr[i, end_pos, end_tok] = 1
                batch_ehr[i, end_pos+1:, pad_tok] = 1
        # Shift mask to match shifted labels
        batch_mask = batch_mask[:, 1:, :]
        return batch_ehr, batch_mask

    def forward(self, **kwargs):
        """
        Forward pass: computes training loss and predicted code probabilities.

        Inputs (in kwargs):
            visits: List[List[int]]
            labels: List[np.ndarray]

        Returns:
            Dict with keys:
                - loss: scalar training loss
                - y_prob: Tensor of shape (B, n_ctx-1, total_vocab_size)
                - y_true: Tensor of true shifted labels
        """
        visits = kwargs[self.feature_key]
        labels = kwargs[self.label_key]
        batch_ehr, batch_mask = self.prepare_batch(visits, labels)

        batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32, device=self.device)
        batch_mask = torch.tensor(batch_mask, dtype=torch.float32, device=self.device)

        # Use core HALOModel to compute loss and probabilities
        loss, code_probs, target = self.model(
            batch_ehr,
            ehr_labels=batch_ehr,
            ehr_masks=batch_mask,
            pos_loss_weight=self.config.pos_loss_weight
        )

        return {"loss": loss, "y_prob": code_probs, "y_true": target}
