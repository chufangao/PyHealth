"""
PromptEHR: Conditional Electronic Healthcare Records Generation with Prompt Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BartTokenizer, BartForConditionalGeneration, BartConfig,
    TrainingArguments, Trainer
)
from transformers.modeling_outputs import BaseModelOutput
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
from pathlib import Path
import warnings
from collections import defaultdict, OrderedDict
import random

from ...datasets import SampleDataset


class PromptEHRConfig(BartConfig):
    """config for promptehr model"""
    
    def __init__(
        self,
        vocab_size_diag: int = 5000,
        vocab_size_proc: int = 1000, 
        vocab_size_med: int = 3000,
        n_numerical_features: int = 8,
        n_categorical_features: int = 0,
        prompt_hidden_size: int = 768,
        max_seq_length: int = 512,
        max_visits: int = 20,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # medical code vocabularies
        self.vocab_size_diag = vocab_size_diag
        self.vocab_size_proc = vocab_size_proc
        self.vocab_size_med = vocab_size_med
        
        # patient features
        self.n_numerical_features = n_numerical_features
        self.n_categorical_features = n_categorical_features
        
        # prompt config
        self.prompt_hidden_size = prompt_hidden_size
        
        # sequence limits
        self.max_seq_length = max_seq_length
        self.max_visits = max_visits
        
        # special tokens
        self.bos_token_id = 0
        self.eos_token_id = 2
        self.pad_token_id = 1


class EHRTokenizer:
    """tokenizer for ehr sequences"""
    
    def __init__(self, 
                 vocab_size_diag: int = 5000,
                 vocab_size_proc: int = 1000,
                 vocab_size_med: int = 3000):
        
        # special tokens
        self.special_tokens = {
            "<pad>": 0,
            "<bos>": 1, 
            "<eos>": 2,
            "<unk>": 3,
            "<diag>": 4,
            "</diag>": 5,
            "<proc>": 6,
            "</proc>": 7,
            "<med>": 8,
            "</med>": 9,
            "<visit>": 10,
            "</visit>": 11
        }
        
        self.vocab_size_diag = vocab_size_diag
        self.vocab_size_proc = vocab_size_proc  
        self.vocab_size_med = vocab_size_med
        
        # offset for each code type
        n_special = len(self.special_tokens)
        self.diag_offset = n_special
        self.proc_offset = n_special + vocab_size_diag
        self.med_offset = n_special + vocab_size_diag + vocab_size_proc
        
        # total vocab size
        self.vocab_size = n_special + vocab_size_diag + vocab_size_proc + vocab_size_med
        
        # reverse mapping
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
    def encode_visit(self, visit_codes: List[str]) -> List[int]:
        """encode a single visit"""
        tokens = [self.special_tokens["<visit>"]]
        
        # group codes by type
        diag_codes = [c for c in visit_codes if c.startswith("diag_")]
        proc_codes = [c for c in visit_codes if c.startswith("proc_")]
        med_codes = [c for c in visit_codes if c.startswith("med_")]
        
        # encode diagnoses
        if diag_codes:
            tokens.append(self.special_tokens["<diag>"])
            for code in diag_codes:
                code_id = int(code.split("_")[1]) if "_" in code else 0
                token_id = self.diag_offset + (code_id % self.vocab_size_diag)
                tokens.append(token_id)
            tokens.append(self.special_tokens["</diag>"])
        
        # encode procedures  
        if proc_codes:
            tokens.append(self.special_tokens["<proc>"])
            for code in proc_codes:
                code_id = int(code.split("_")[1]) if "_" in code else 0
                token_id = self.proc_offset + (code_id % self.vocab_size_proc)
                tokens.append(token_id)
            tokens.append(self.special_tokens["</proc>"])
        
        # encode medications
        if med_codes:
            tokens.append(self.special_tokens["<med>"])
            for code in med_codes:
                code_id = int(code.split("_")[1]) if "_" in code else 0
                token_id = self.med_offset + (code_id % self.vocab_size_med)
                tokens.append(token_id)
            tokens.append(self.special_tokens["</med>"])
        
        tokens.append(self.special_tokens["</visit>"])
        return tokens
    
    def encode_sequence(self, visits: List[List[str]], max_length: int = 512) -> List[int]:
        """encode sequence of visits"""
        tokens = [self.special_tokens["<bos>"]]
        
        for visit in visits:
            visit_tokens = self.encode_visit(visit)
            if len(tokens) + len(visit_tokens) + 1 <= max_length:  # +1 for eos
                tokens.extend(visit_tokens)
            else:
                break
        
        tokens.append(self.special_tokens["<eos>"])
        
        # pad to max length
        while len(tokens) < max_length:
            tokens.append(self.special_tokens["<pad>"])
        
        return tokens[:max_length]
    
    def decode_tokens(self, tokens: List[int]) -> List[List[str]]:
        """decode tokens back to visit structure"""
        visits = []
        current_visit = []
        current_type = None
        
        i = 0
        while i < len(tokens):
            token_id = tokens[i]
            
            if token_id == self.special_tokens["<visit>"]:
                current_visit = []
            elif token_id == self.special_tokens["</visit>"]:
                if current_visit:
                    visits.append(current_visit)
                current_visit = []
            elif token_id == self.special_tokens["<diag>"]:
                current_type = "diag"
            elif token_id == self.special_tokens["<proc>"]:
                current_type = "proc"  
            elif token_id == self.special_tokens["<med>"]:
                current_type = "med"
            elif token_id in [self.special_tokens["</diag>"], 
                             self.special_tokens["</proc>"], 
                             self.special_tokens["</med>"]]:
                current_type = None
            elif current_type and token_id >= len(self.special_tokens):
                # decode medical code
                if current_type == "diag" and token_id >= self.diag_offset:
                    code_id = token_id - self.diag_offset
                    current_visit.append(f"diag_{code_id}")
                elif current_type == "proc" and token_id >= self.proc_offset:
                    code_id = token_id - self.proc_offset  
                    current_visit.append(f"proc_{code_id}")
                elif current_type == "med" and token_id >= self.med_offset:
                    code_id = token_id - self.med_offset
                    current_visit.append(f"med_{code_id}")
            
            i += 1
        
        return visits


class ConditionalPrompt(nn.Module):
    """conditional prompt module for patient features"""
    
    def __init__(self, 
                 n_numerical: int,
                 n_categorical: int, 
                 hidden_size: int):
        super().__init__()
        
        self.n_numerical = n_numerical
        self.n_categorical = n_categorical
        self.hidden_size = hidden_size
        
        # numerical feature encoding
        if n_numerical > 0:
            self.numerical_proj = nn.Linear(n_numerical, hidden_size)
            
        # categorical feature encoding  
        if n_categorical > 0:
            self.categorical_embed = nn.Embedding(n_categorical, hidden_size)
        
        # prompt generation
        total_features = (1 if n_numerical > 0 else 0) + (1 if n_categorical > 0 else 0)
        if total_features > 0:
            self.prompt_proj = nn.Linear(total_features * hidden_size, hidden_size)
        
    def forward(self, numerical_features: Optional[torch.Tensor] = None,
                categorical_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """generate conditional prompt"""
        
        features = []
        
        if self.n_numerical > 0 and numerical_features is not None:
            num_encoded = self.numerical_proj(numerical_features)  # [batch, hidden]
            features.append(num_encoded)
        
        if self.n_categorical > 0 and categorical_features is not None:
            cat_encoded = self.categorical_embed(categorical_features)  # [batch, hidden]
            features.append(cat_encoded)
        
        if features:
            combined = torch.cat(features, dim=-1)  # [batch, n_features * hidden]
            prompt = self.prompt_proj(combined)  # [batch, hidden]
            return prompt.unsqueeze(1)  # [batch, 1, hidden]
        else:
            # fallback - learnable prompt
            batch_size = numerical_features.size(0) if numerical_features is not None else 1
            device = numerical_features.device if numerical_features is not None else torch.device('cpu')
            return torch.zeros(batch_size, 1, self.hidden_size, device=device)


class PromptBartEncoder(nn.Module):
    """bart encoder with conditional prompts"""
    
    def __init__(self, config: PromptEHRConfig):
        super().__init__()
        
        # base bart encoder components
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.d_model)
        
        # transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.encoder_attention_heads,
                dim_feedforward=config.encoder_ffn_dim,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.encoder_layers)
        ])
        
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # conditional prompt
        self.conditional_prompt = ConditionalPrompt(
            config.n_numerical_features,
            config.n_categorical_features, 
            config.d_model
        )
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                numerical_features: Optional[torch.Tensor] = None,
                categorical_features: Optional[torch.Tensor] = None) -> BaseModelOutput:
        
        # token embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        
        # position embeddings
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.embed_positions(positions)
        
        # combine embeddings
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # add conditional prompt
        prompt = self.conditional_prompt(numerical_features, categorical_features)
        hidden_states = torch.cat([prompt, hidden_states], dim=1)
        
        # update attention mask for prompt
        if attention_mask is not None:
            batch_size = attention_mask.size(0)
            prompt_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None
        )


class PromptBartDecoder(nn.Module):
    """bart decoder with conditional prompts"""
    
    def __init__(self, config: PromptEHRConfig):
        super().__init__()
        
        # base bart decoder components  
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.d_model)
        
        # transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.decoder_attention_heads,
                dim_feedforward=config.decoder_ffn_dim,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.decoder_layers)
        ])
        
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # output projections for different code types
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
    def forward(self,
                input_ids: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None) -> BaseModelOutput:
        
        # token embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        
        # position embeddings
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.embed_positions(positions)
        
        # combine embeddings
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)
        
        # transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None,
                memory_key_padding_mask=~encoder_attention_mask.bool() if encoder_attention_mask is not None else None
            )
        
        # output projection
        logits = self.lm_head(hidden_states)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None
        ), logits


class PromptEHRModel(nn.Module):
    """main promptehr model"""
    
    def __init__(self, config: PromptEHRConfig):
        super().__init__()
        
        self.config = config
        self.tokenizer = EHRTokenizer(
            config.vocab_size_diag,
            config.vocab_size_proc,
            config.vocab_size_med
        )
        
        # encoder-decoder
        self.encoder = PromptBartEncoder(config)
        self.decoder = PromptBartDecoder(config)
        
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                decoder_input_ids: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None,
                numerical_features: Optional[torch.Tensor] = None,
                categorical_features: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # encode
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            numerical_features=numerical_features,
            categorical_features=categorical_features
        )
        
        # decode
        if decoder_input_ids is None and labels is not None:
            # shift labels for teacher forcing
            decoder_input_ids = labels.clone()
            decoder_input_ids[:, 1:] = labels[:, :-1]
            decoder_input_ids[:, 0] = self.config.bos_token_id
        
        decoder_outputs, logits = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask
        )
        
        # compute loss
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "encoder_last_hidden_state": encoder_outputs.last_hidden_state,
            "decoder_last_hidden_state": decoder_outputs.last_hidden_state
        }
    
    def generate(self,
                 input_ids: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 numerical_features: Optional[torch.Tensor] = None,
                 categorical_features: Optional[torch.Tensor] = None,
                 max_length: int = 512,
                 temperature: float = 1.0,
                 do_sample: bool = True) -> torch.Tensor:
        """generate synthetic sequences"""
        
        self.eval()
        with torch.no_grad():
            batch_size = input_ids.size(0)
            device = input_ids.device
            
            # encode
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_features=numerical_features,
                categorical_features=categorical_features
            )
            
            # init decoder input
            decoder_input_ids = torch.full(
                (batch_size, 1), 
                self.config.bos_token_id, 
                device=device, 
                dtype=torch.long
            )
            
            # generate tokens
            for _ in range(max_length - 1):
                decoder_attention_mask = torch.ones_like(decoder_input_ids)
                
                _, logits = self.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    attention_mask=decoder_attention_mask,
                    encoder_attention_mask=attention_mask
                )
                
                # sample next token
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # append token
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
                
                # check for eos
                if (next_token == self.config.eos_token_id).all():
                    break
            
            return decoder_input_ids


class PromptEHRDataset(Dataset):
    """dataset for promptehr training"""
    
    def __init__(self, 
                 samples: List[Dict[str, Any]], 
                 tokenizer: EHRTokenizer,
                 max_length: int = 512):
        
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # get visits and features
        visits = sample["v"]
        features = sample.get("x", [])
        
        # tokenize sequence
        input_tokens = self.tokenizer.encode_sequence(visits, self.max_length)
        
        # create input/target pairs for training
        input_ids = input_tokens[:-1]  # remove last token
        labels = input_tokens[1:]      # shift by one
        
        # pad to max length
        while len(input_ids) < self.max_length - 1:
            input_ids.append(self.tokenizer.special_tokens["<pad>"])
            labels.append(self.tokenizer.special_tokens["<pad>"])
        
        # attention mask
        attention_mask = [1 if token != self.tokenizer.special_tokens["<pad>"] else 0 for token in input_ids]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "numerical_features": torch.tensor(features, dtype=torch.float) if features else torch.zeros(8)
        }


class PromptEHRGenerator:
    """main generator class for promptehr"""
    
    def __init__(self, 
                 vocab_size_diag: int = 5000,
                 vocab_size_proc: int = 1000,
                 vocab_size_med: int = 3000,
                 n_numerical_features: int = 8,
                 n_categorical_features: int = 0,
                 hidden_size: int = 768,
                 num_layers: int = 6,
                 num_heads: int = 12,
                 max_seq_length: int = 512):
        
        # config
        self.config = PromptEHRConfig(
            vocab_size=12 + vocab_size_diag + vocab_size_proc + vocab_size_med,  # special + medical codes
            vocab_size_diag=vocab_size_diag,
            vocab_size_proc=vocab_size_proc,
            vocab_size_med=vocab_size_med,
            n_numerical_features=n_numerical_features,
            n_categorical_features=n_categorical_features,
            d_model=hidden_size,
            encoder_layers=num_layers,
            decoder_layers=num_layers,
            encoder_attention_heads=num_heads,
            decoder_attention_heads=num_heads,
            max_seq_length=max_seq_length
        )
        
        # model
        self.model = PromptEHRModel(self.config)
        self.tokenizer = self.model.tokenizer
        
        # training components
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self, 
            dataset: SampleDataset,
            val_dataset: Optional[SampleDataset] = None,
            batch_size: int = 16,
            learning_rate: float = 1e-4,
            num_epochs: int = 10,
            save_dir: Optional[str] = None):
        """train the model"""
        
        # move model to device
        self.model.to(self.device)
        
        # create datasets
        train_dataset = PromptEHRDataset(dataset.samples, self.tokenizer, self.config.max_seq_length)
        
        val_dataset_obj = None
        if val_dataset:
            val_dataset_obj = PromptEHRDataset(val_dataset.samples, self.tokenizer, self.config.max_seq_length)
        
        # dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset_obj, batch_size=batch_size) if val_dataset_obj else None
        
        # optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        print(f"training on {self.device}")
        print(f"train samples: {len(train_dataset)}")
        if val_loader:
            print(f"val samples: {len(val_dataset_obj)}")
        
        # training loop
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # forward
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"epoch {epoch+1}/{num_epochs}, batch {batch_idx}, loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            print(f"epoch {epoch+1} avg loss: {avg_loss:.4f}")
            
            # validation
            if val_loader:
                val_loss = self._evaluate(val_loader)
                print(f"epoch {epoch+1} val loss: {val_loss:.4f}")
        
        # save model
        if save_dir:
            self.save(save_dir)
            print(f"saved model to {save_dir}")
    
    def _evaluate(self, dataloader: DataLoader) -> float:
        """evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs["loss"].item()
        
        return total_loss / len(dataloader)
    
    def generate(self,
                 n_samples: int = 1000,
                 max_length: int = 512,
                 temperature: float = 1.0,
                 seed_visits: Optional[List[List[str]]] = None,
                 numerical_features: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """generate synthetic samples"""
        
        self.model.eval()
        synthetic_samples = []
        
        # default seed (empty visit)
        if seed_visits is None:
            seed_visits = [[]]
        
        # default features (zeros)
        if numerical_features is None:
            numerical_features = np.zeros((n_samples, self.config.n_numerical_features))
        elif len(numerical_features) != n_samples:
            # repeat or sample features to match n_samples
            if len(numerical_features) < n_samples:
                indices = np.random.choice(len(numerical_features), n_samples, replace=True)
                numerical_features = numerical_features[indices]
            else:
                numerical_features = numerical_features[:n_samples]
        
        batch_size = min(32, n_samples)  # process in batches
        
        print(f"generating {n_samples} synthetic samples...")
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            current_batch_size = batch_end - i
            
            # prepare batch
            batch_visits = [seed_visits[0]] * current_batch_size  # use same seed for all
            batch_features = numerical_features[i:batch_end]
            
            # tokenize input
            input_tokens = []
            for visits in batch_visits:
                tokens = self.tokenizer.encode_sequence(visits, max_length)
                input_tokens.append(tokens)
            
            # to tensors
            input_ids = torch.tensor(input_tokens, dtype=torch.long).to(self.device)
            attention_mask = (input_ids != self.tokenizer.special_tokens["<pad>"]).long()
            features_tensor = torch.tensor(batch_features, dtype=torch.float).to(self.device)
            
            # generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numerical_features=features_tensor,
                    max_length=max_length,
                    temperature=temperature
                )
            
            # decode generated sequences
            for j, gen_ids in enumerate(generated_ids):
                decoded_visits = self.tokenizer.decode_tokens(gen_ids.cpu().tolist())
                
                synthetic_samples.append({
                    "visits": decoded_visits,
                    "numerical_features": batch_features[j].tolist(),
                    "n_visits": len(decoded_visits),
                    "n_codes": sum(len(visit) for visit in decoded_visits)
                })
        
        print(f"generated {len(synthetic_samples)} synthetic samples")
        return synthetic_samples
    
    def save(self, save_dir: str):
        """save model and config"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # save model state
        torch.save(self.model.state_dict(), save_path / "model.pt")
        
        # save config
        with open(save_path / "config.pkl", "wb") as f:
            pickle.dump(self.config, f)
        
        print(f"saved model to {save_path}")
    
    @classmethod
    def load(cls, save_dir: str) -> "PromptEHRGenerator":
        """load model from checkpoint"""
        save_path = Path(save_dir)
        
        # load config
        with open(save_path / "config.pkl", "rb") as f:
            config = pickle.load(f)
        
        # create generator
        generator = cls(
            vocab_size_diag=config.vocab_size_diag,
            vocab_size_proc=config.vocab_size_proc,
            vocab_size_med=config.vocab_size_med,
            n_numerical_features=config.n_numerical_features,
            n_categorical_features=config.n_categorical_features,
            hidden_size=config.d_model,
            num_layers=config.encoder_layers,
            num_heads=config.encoder_attention_heads,
            max_seq_length=config.max_seq_length
        )
        
        # load model state
        generator.model.load_state_dict(torch.load(save_path / "model.pt", map_location=generator.device))
        
        print(f"loaded model from {save_path}")
        return generator