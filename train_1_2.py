"""
Problem 1-2: Learned Multiplier Training
Using fixed protocol: 100K data, 200 epochs, AdamW + CosineAnnealingLR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import time
import math

# ============================================================
# PART 1: Utility Functions & Data Generation
# ============================================================

def int_to_lsb_bits(n, num_bits):
    """Convert integer to LSB-first binary list."""
    bits = []
    for _ in range(num_bits):
        bits.append(n & 1)
        n >>= 1
    return bits


def lsb_bits_to_int(bits):
    """Convert LSB-first binary list to integer."""
    result = 0
    for i, b in enumerate(bits):
        result += b * (2 ** i)
    return result


def generate_multiplication_data(num_samples, seed=42):
    """Generate multiplication training data."""
    random.seed(seed)
    data = []
    for _ in range(num_samples):
        a = random.randint(0, 63)
        b = random.randint(0, 63)
        p = a * b
        a_bits = int_to_lsb_bits(a, 6)
        b_bits = int_to_lsb_bits(b, 6)
        p_bits = int_to_lsb_bits(p, 12)
        input_seq = a_bits + b_bits
        output_seq = p_bits
        data.append((input_seq, output_seq))
    return data


class MultiplicationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, output_seq = self.data[idx]
        full_seq = input_seq + output_seq
        return torch.tensor(full_seq, dtype=torch.long)


# ============================================================
# PART 2: Model Architecture (Problem 1-2)
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiplicationTransformer(nn.Module):
    """
    Trainable transformer for 6-bit binary multiplication.
    Autoregressive Decoder-only architecture.
    """
    def __init__(self, d_model=48, nhead=4, num_layers=4, d_ff=128):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(2, d_model)
        self.pos_emb = SinusoidalPositionalEncoding(d_model, max_len=32)
        
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            batch_first=True, norm_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 2)
        
    def forward(self, x):
        B, L = x.shape
        h = self.embedding(x)
        h = self.pos_emb(h)
        
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        
        h = self.encoder(h, mask=mask, is_causal=True)
        h = self.ln_f(h)
        logits = self.output_proj(h)
        return logits

    def generate(self, inp, n=12):
        self.eval()
        cur = inp.clone()
        with torch.no_grad():
            for _ in range(n):
                logits = self.forward(cur)
                nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                cur = torch.cat([cur, nxt], dim=1)
        return cur[:, 12:]


# ============================================================
# PART 3: Training Functions
# ============================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model_1_2(model, train_loader, num_epochs=200, device="cuda", learning_rate=1e-3):
    """
    Train Problem 1-2 model with fixed protocol.
    - Optimizer: AdamW(lr=1e-3, weight_decay=0.01)
    - Schedule: CosineAnnealingLR for 200 epochs
    - Loss: Cross entropy on 12 output token positions
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model.train()
    best_loss = float('inf')
    
    print(f"\n{'='*70}")
    print(f"Training Problem 1-2: Learned Multiplier")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model parameters: {count_parameters(model)}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {len(train_loader) * train_loader.batch_size // len(train_loader) if len(train_loader) > 0 else 'N/A'}")
    print(f"Learning rate: {learning_rate}")
    print(f"Optimizer: AdamW(weight_decay=0.01)")
    print(f"Schedule: CosineAnnealingLR")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            input_ids = batch[:, :-1]  # length 23
            targets = batch[:, 1:]     # length 23
            
            logits = model(input_ids)  # [B, 23, 2]
            # Loss only on 12 output token positions (which are indices 11 to 22 in predictions)
            logits_out = logits[:, 11:23, :].reshape(-1, 2)
            targets_out = targets[:, 11:23].reshape(-1)
            loss = F.cross_entropy(logits_out, targets_out)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        scheduler.step()
        best_loss = min(best_loss, avg_loss)
        
        # Progress report every 20 epochs
        if (epoch + 1) % 20 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (epoch + 1) * (num_epochs - epoch - 1)
            print(f"Epoch {epoch + 1:3d}/{num_epochs} | Loss: {avg_loss:.6f} | "
                  f"Best: {best_loss:.6f} | Time: {elapsed:.0f}s (ETA: {eta:.0f}s)")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training completed in {elapsed:.1f}s")
    print(f"Final loss: {avg_loss:.6f}")
    print(f"{'='*70}\n")
    
    return model


def evaluate_model_1_2(model, test_data, device="cuda"):
    """Evaluate model accuracy on test data."""
    model.eval()
    correct = 0
    total = len(test_data)
    
    with torch.no_grad():
        for idx, (input_seq, expected_output) in enumerate(test_data):
            a_bits, b_bits = input_seq[:6], input_seq[6:]
            inp = torch.tensor([a_bits + b_bits], dtype=torch.long).to(device)
            pred_output = model.generate(inp, n=12)[0]
            pred_value = lsb_bits_to_int(pred_output.cpu().tolist())
            expected_value = lsb_bits_to_int(expected_output)
            
            if pred_value == expected_value:
                correct += 1
            
            if (idx + 1) % 2000 == 0:
                print(f"  Evaluated {idx + 1}/{total} samples...")
    
    acc = correct / total
    return acc


# ============================================================
# PART 4: Main - Problem 1-2 Training Pipeline
# ============================================================

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Build model
    print("\n[1/4] Building model...")
    model = MultiplicationTransformer(d_model=36, nhead=12, d_ff=48).to(device)
    p_2 = count_parameters(model)
    print(f"  Model parameters (P_2): {p_2}")
    
    # Generate training data (100,000 samples per protocol)
    print("\n[2/4] Generating training data (100,000 samples)...")
    train_data = generate_multiplication_data(100000, seed=42)
    train_dataset = MultiplicationDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    print(f"  Training batches: {len(train_loader)}")
    
    # Train with fixed protocol
    print("\n[3/4] Training...")
    model = train_model_1_2(
        model, train_loader, 
        num_epochs=200, 
        device=device, 
        learning_rate=1e-3
    )
    
    # Evaluate on 10,000 test samples
    print("\n[4/4] Evaluating accuracy (10,000 test samples)...")
    test_data = generate_multiplication_data(10000, seed=555)
    acc_2 = evaluate_model_1_2(model, test_data, device=device)
    
    # Results
    print("\n" + "="*70)
    print("PROBLEM 1-2 RESULTS")
    print("="*70)
    print(f"P_2 (Parameters): {p_2}")
    print(f"Acc_2 (Accuracy):  {acc_2:.4f} ({acc_2*100:.2f}%)")
    print("="*70 + "\n")
    
    # Return results
    return {
        'P_2': p_2,
        'Acc_2': acc_2
    }


if __name__ == "__main__":
    results = main()
    print(f"✓ Training complete!")
    print(f"  P_2 = {results['P_2']}")
    print(f"  Acc_2 = {results['Acc_2']:.4f}")
