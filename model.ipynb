from __future__ import annotations

import os
import re
import glob
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel, logging as hf_logging

# --- 1. SETUP & REPRODUCIBILITY ---
hf_logging.set_verbosity_error()
rng_seed = 42
random.seed(rng_seed)
np.random.seed(rng_seed)
torch.manual_seed(rng_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# --- 2. CONFIGURATION & HYPERPARAMETERS ---
DATA_ROOT = Path("/kaggle/input/dna-data")
TRAIN_FILE = DATA_ROOT / "train.tsv"
TEST_FILE  = DATA_ROOT / "test.tsv"
FASTA_FILE = DATA_ROOT / "sequences.fasta"

MODEL_ID = "LongSafari/hyenadna-small-32k-seqlen-hf"
MAX_LEN  = 16_384
EPOCHS   = 3
LOAD_KWARGS = {"batch_size": 1, "num_workers": 0}
LR = 2e-5
WD = 1e-2
VAL_FRAC = 0.1

# --- 3. MODEL INITIALIZATION (LINEAR PROBING) ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
config    = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, num_labels=2)
encoder   = AutoModel.from_pretrained(MODEL_ID, config=config, trust_remote_code=True)

# Freeze the base foundation model to prevent catastrophic forgetting and save compute
for p in encoder.parameters():
    p.requires_grad = False

hidden_dim = (
    getattr(config, "hidden_size", None)
    or getattr(config, "d_model", None)
    or getattr(config, "n_embd", None)
)
assert hidden_dim, "Cannot determine hidden_dim"

class TokenSequenceTagger(nn.Module):
    """Custom classification head attached to the frozen foundation model."""
    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.dropout    = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids=None, labels=None):
        x = self.base_model(input_ids=input_ids).last_hidden_state
        x = self.dropout(x)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            # Mask out padding tokens (-100) so they do not contribute to the loss calculation
            mask = labels.view(-1) != -100  
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, 2)[mask], labels.view(-1)[mask])
            
        return {"loss": loss, "logits": logits}

model = TokenSequenceTagger(encoder, hidden_dim).to(device)

# --- 4. DATA PIPELINE & PREPROCESSING ---
@dataclass
class SequenceEntry:
    id: str
    seq: str
    tags: Optional[List[int]]

class SequenceDataset(Dataset):
    """Maps gene IDs to their raw nucleotide sequences and binary labels."""
    def __init__(self, df: pd.DataFrame, fasta_map: Dict[str,str], tok, is_test=False):
        self.records: List[SequenceEntry] = []
        for _, row in df.iterrows():
            sid = row.iloc[0]
            lbls = None if is_test else [int(ch) for ch in row['label'].strip()]
            self.records.append(SequenceEntry(sid, fasta_map[sid], lbls))
        self.tok = tok
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        ids = self.tok(rec.seq, add_special_tokens=False)["input_ids"]
        return {"id": rec.id, "input_ids": ids, "labels": rec.tags}

class BatchCollator:
    """Dynamically chunks massive DNA sequences to fit within context limits and pads batches."""
    def __init__(self, tok, max_chunk=MAX_LEN):
        self.tok = tok
        self.max_chunk = max_chunk
        self.pad_id = tok.pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict:
        chunks, lbls, ids = [], [], []
        
        # Segment long sequences into smaller, fixed-size windows
        for sample in batch:
            seq_ids, seq_lbl = sample['input_ids'], sample['labels']
            for s in range(0, len(seq_ids), self.max_chunk):
                chunk = seq_ids[s:s+self.max_chunk]
                chunks.append(chunk)
                lbls.append(None if seq_lbl is None else seq_lbl[s:s+self.max_chunk])
                ids.append(sample['id'])
                
        # Pad chunks to match the longest sequence in the current batch for efficient matrix ops
        max_len = max(len(c) for c in chunks)
        for i, c in enumerate(chunks):
            pad = max_len - len(c)
            c.extend([self.pad_id] * pad)
            if lbls[i] is not None:
                lbls[i].extend([-100] * pad) # -100 is ignored by CrossEntropyLoss
                
        return {
            'id': ids,
            'input_ids': torch.tensor(chunks, device=device),
            'labels': None if lbls[0] is None else torch.tensor(lbls, device=device)
        }

def parse_fasta(path: Path) -> Dict[str,str]:
    """Loads raw FASTA sequences into memory."""
    seqs: Dict[str,str] = {}
    cur = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                cur = line[1:]
                seqs[cur] = ''
            else:
                seqs[cur] += line.upper()
    return seqs

# --- 5. DATASET LOADING & BATCHING ---
seq_map = parse_fasta(FASTA_FILE)
print(f"FASTA entries: {len(seq_map)}")

train_df = pd.read_csv(TRAIN_FILE, sep='\t')
val_n = int(len(train_df) * VAL_FRAC)
train_set, val_set = random_split(
    SequenceDataset(train_df, seq_map, tokenizer),
    [len(train_df)-val_n, val_n]
)

test_df = pd.read_csv(TEST_FILE, sep='\t')
test_set = SequenceDataset(test_df, seq_map, tokenizer, is_test=True)

dataloader_kwargs = LOAD_KWARGS
collator = BatchCollator(tokenizer)
dls = {
    'train': DataLoader(train_set, shuffle=True, collate_fn=collator, **dataloader_kwargs),
    'val':   DataLoader(val_set,   shuffle=False, collate_fn=collator, **dataloader_kwargs),
    'test':  DataLoader(test_set,  shuffle=False, collate_fn=collator, **dataloader_kwargs)
}

# --- 6. CHECKPOINTING & OPTIMIZATION ---
ckpts = sorted(glob.glob('checkpoint_e*.pt'), key=lambda fn: int(re.search(r'e(\d+)', fn).group(1)))
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
epoch_start = 1

# Resume training state if interrupted
if ckpts:
    ckpt = torch.load(ckpts[-1], map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optim.load_state_dict(ckpt['optim_state'])
    epoch_start = ckpt['epoch'] + 1
    print(f"Resumed at epoch {epoch_start}")

# --- 7. TRAINING LOOP ---
for ep in range(epoch_start, EPOCHS+1):
    model.train()
    total_loss = 0.0
    iters = 0
    
    for batch in tqdm(dls['train'], desc=f"Training ep {ep}"):
        optim.zero_grad()
        out = model(input_ids=batch['input_ids'], labels=batch['labels'])
        out['loss'].backward()
        optim.step()
        total_loss += out['loss'].item()
        iters += 1
    print(f"Epoch {ep} loss: {total_loss/iters:.4f}")

    # Validation Phase
    model.eval()
    correct = tot = 0
    with torch.no_grad():
        for batch in dls['val']:
            log = model(input_ids=batch['input_ids'], labels=batch['labels'])['logits']
            preds = log.argmax(-1)
            
            # Ensure padding tokens are excluded from accuracy calculations
            mask = batch['labels'] != -100
            correct += ((preds == batch['labels']) & mask).sum().item()
            tot += mask.sum().item()
    print(f"Val acc: {correct/tot:.2%}")

    torch.save({'epoch': ep, 'model_state': model.state_dict(), 'optim_state': optim.state_dict()}, f"checkpoint_e{ep}.pt")

# --- 8. INFERENCE & OUTPUT ---
model.eval()
outputs: Dict[str,List[int]] = {}

with torch.no_grad():
    for batch in tqdm(dls['test'], desc="Predicting"):
        log = model(input_ids=batch['input_ids'])['logits']
        preds = log.argmax(-1).cpu().tolist()
        
        # Reassemble chunked predictions back into continuous strings per gene ID
        for sid, seq in zip(batch['id'], preds):
            outputs.setdefault(sid, []).extend(seq)

# Format and export predictions for Codabench scoring
pd.DataFrame({
    'id': list(outputs), 
    'prediction': [''.join(map(str, v)) for v in outputs.values()]
}).to_csv('predictions.tsv', sep='\t', index=False)
