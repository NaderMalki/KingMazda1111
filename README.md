/Nader/maleikie/Invention of Deep Neural Networks /Artificial Intelligence /Bihemispheric Processing /Registration Number /140450140003002031/Iran

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



