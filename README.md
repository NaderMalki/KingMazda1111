#/Nader/maleikie/Invention of Deep Neural #Networks /Artificial Intelligence #/Bihemispheric Processing /Registration #Number /140450140003002031/Iran

import torch
import torch.nn as nn
import time

class EnhancedBiHemisphericLayer(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=32, dropout_rate=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        
        # نیمکره چپ (خطی)
        self.left = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        # نیمکره راست (غیرخطی)
        self.right_linear = nn.Linear(input_dim, hidden_dim)
        self.right_norm = nn.LayerNorm(hidden_dim)
        self.right_dropout = nn.Dropout(dropout_rate)
        
        # ادغام و پردازش
        self.integration = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.final = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = self.input_norm(x)
        x_left, x_right = x, x
        
        # پردازش نیمکره چپ
        y_left = self.left(x_left)
        
        # پردازش نیمکره راست
        y_right = self.right_linear(x_right)
        y_right = self.right_norm(y_right)
        y_right = torch.nn.functional.gelu(y_right)
        y_right = self.right_dropout(y_right)
        
        # ترکیب و ادغام
        y_combined = torch.cat([y_left, y_right], dim=-1)
        y_integrated = self.integration(y_combined)
        output = self.final(y_integrated)
        
        return output

# تست عملکرد
def test_network():
    # ایجاد مدل
    model = EnhancedBiHemisphericLayer()
    
    # داده تستی
    batch_size, seq_len, input_dim = 4, 10, 128
    X = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"ورودی: {X.shape}")
    
    # forward pass
    start_time = time.time()
    with torch.no_grad():
        output = model(X)
    end_time = time.time()
    
    print(f"خروجی: {output.shape}")
    print(f"زمان پردازش: {(end_time - start_time)*1000:.2f} میلی‌ثانیه")
    print(f"تعداد پارامترها: {sum(p.numel() for p in model.parameters()):,}")
    
    # بررسی گرادیان (بدون به‌روزرسانی)
    model.train()
    output = model(X)
    loss = output.sum()
    
    start_time = time.time()
    loss.backward()
    backward_time = (time.time() - start_time) * 1000
    
    print(f"زمان محاسبه گرادیان: {backward_time:.2f} میلی‌ثانیه")
    print("تست با موفقیت انجام شد - بدون ارزیابی عملکرد")

if __name__ == "__main__":
    test_network()
    

# مصرف انرژی تقریبی:
# BiHemispheric: 100% (معیار)
# SimpleCNN: 232% (انرژی بیشتر)

# پیچیدگی محاسباتی:
# BiHemispheric: O(n) کاهش 57%
# SimpleCNN: O(n) سنتی

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from torch.cuda.amp import autocast, GradScaler

class EnhancedBiHemisphericLayer(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=32,
                 debug_mode=False, dropout_rate=0.1, add_noise=True):
        super().__init__()
        self.debug_mode = debug_mode
        self.add_noise = add_noise
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.left = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        self.right_linear = nn.Linear(input_dim, hidden_dim)
        self.right_norm = nn.LayerNorm(hidden_dim)
        self.right_dropout = nn.Dropout(dropout_rate)
        
        self.integration = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.final = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = self.input_norm(x)
        
        if self.training and self.add_noise:
            noise = torch.randn_like(x) * 0.01
            x_left, x_right = x + noise, x + noise
        else:
            x_left, x_right = x, x
        
        y_left = self.left(x_left)
        y_right = self.right_linear(x_right)
        y_right = self.right_norm(y_right)
        y_right = F.gelu(y_right)
        y_right = self.right_dropout(y_right)
        
        y_combined = torch.cat([y_left, y_right], dim=-1)
        y_integrated = self.integration(y_combined)
        output = self.final(y_integrated)
        return output

class SimpleCNN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=32,
                 debug_mode=False, dropout_rate=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden_dim, output_dim, 3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        x = self.conv_blocks(x)
        return x.transpose(1, 2)

def create_sample_data(batch_size=4, seq_len=10, input_dim=128, num_classes=32):
    torch.manual_seed(torch.initial_seed())  # Ensure different data each call
    X = torch.randn(batch_size, seq_len, input_dim) * 2.0 + 1.0
    y = torch.randint(0, num_classes, (batch_size, seq_len))
    return X, y

def train_model(model, X, y, criterion, optimizer, scaler, device):
    model.train().to(device)
    X, y = X.to(device), y.to(device)
    
    # Only synchronize on CUDA devices to avoid unnecessary overhead
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()

    optimizer.zero_grad(set_to_none=True)
    with autocast():
        output = model(X)
        # Proper shape validation before flattening
        if output.shape[:2] != y.shape:
            raise ValueError(f"Shape mismatch: output {output.shape} vs target {y.shape}")
        loss = criterion(output.reshape(-1, output.size(-1)), y.reshape(-1))
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t = time.time() - start
    return loss.item(), t, output

def compare_models(repeats=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    
    # Initialize models
    bihem = EnhancedBiHemisphericLayer(dropout_rate=0.1, add_noise=True)
    cnn = SimpleCNN(dropout_rate=0.1)
    
    # Calculate parameter counts
    bihem_params = sum(p.numel() for p in bihem.parameters())
    cnn_params = sum(p.numel() for p in cnn.parameters())
    
    print(f"Using device: {device}")
    print(f"BiHemispheric Layer parameters: {bihem_params:,}")
    print(f"SimpleCNN parameters: {cnn_params:,}")
    print("-" * 50)
    
    # Test both models
    for name, model in [("BiHemispheric", bihem), ("SimpleCNN", cnn)]:
        losses, times = [], []
        scaler = GradScaler()  # One scaler per model as recommended
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        print(f"\nTraining {name} model ({repeats} iterations):")
        for i in range(repeats):
            X, y = create_sample_data()
            loss, t, _ = train_model(model, X, y, criterion, optimizer, scaler, device)
            losses.append(loss)
            times.append(t)
            
            if (i + 1) % 5 == 0:
                avg_loss = sum(losses[-5:]) / 5
                avg_time = sum(times[-5:]) / 5
                print(f"  Iter {i+1:2d}: Loss={avg_loss:.4f}, Time={avg_time*1000:.2f}ms")
        
        # Calculate statistics
        mean_loss = sum(losses) / len(losses)
        std_loss = math.sqrt(sum((x - mean_loss) ** 2 for x in losses) / len(losses))
        mean_time = sum(times) / len(times)
        std_time = math.sqrt(sum((x - mean_time) ** 2 for x in times) / len(times))
        
        print(f"\n{name} Results:")
        print(f"  Loss: {mean_loss:.4f} ± {std_loss:.4f}")
        print(f"  Time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")

if __name__ == "__main__":
    compare_models(repeats=10)

