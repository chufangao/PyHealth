import pickle
import numpy as np
import torch
from torch.optim import Adam
from pyhealth.datasets import SampleEHRDataset, get_dataloader

# 1) Load your pre-built pickles
train_samples = pickle.load(open("./trainDataset.pkl", "rb"))
val_samples   = pickle.load(open("./valDataset.pkl",   "rb"))
index_to_code = pickle.load(open("./indexToCode.pkl",  "rb"))

# wrap into a PyHealth dataset
train_ds = SampleEHRDataset(samples=train_samples,  dataset_name="halo-train")
val_ds   = SampleEHRDataset(samples=val_samples,    dataset_name="halo-val")

# 2) Instantiate the PyHealth-friendly HALO wrapper
from models.generators.halo import HaloGenerator
generator = HaloGenerator(dataset=train_ds).to(generator.device)

# 3) Build PyHealth dataloaders and optimizer
train_loader = get_dataloader(train_ds, batch_size=16, shuffle=True)
val_loader   = get_dataloader(val_ds,   batch_size=16, shuffle=False)
optimizer    = Adam(generator.parameters(), lr=5e-4)

# 4) Simple train loop
for epoch in range(5):
    generator.train()
    total_loss = 0.0
    for batch in train_loader:
        # batch is a dict with keys "visits" and "labels"
        out = generator(visits=batch["visits"], labels=batch["labels"])
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} train loss: {total_loss/len(train_loader):.4f}")
    
    # validation
    generator.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch in val_loader:
            out = generator(visits=batch["visits"], labels=batch["labels"])
            val_loss += out["loss"].item()
        print(f"Epoch {epoch}  val loss: {val_loss/len(val_loader):.4f}")

# 5) Generation (inference)
# pick one example from your training set:
prefix_visits = train_samples[0]["visits"]   # e.g. [[12,34],[56],…]
prefix_labels = train_samples[0]["labels"]   # numpy array of length label_vocab_size

# generate up to 10 new visits, deterministically
synthetic = generator.generate(
    visits_batch=[prefix_visits], 
    labels_batch=[prefix_labels],
    max_steps=10,
    random=False
)[0]  # synthetic is a list of visits-per-patient

# 6) Decode code-indices → ICD9 strings
decoded = []
for visit in synthetic:
    decoded.append([ index_to_code[idx] for idx in visit ])
print("Synthetic visits:", decoded)
