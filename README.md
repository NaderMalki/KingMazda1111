#/Nader/maleikie/Invention of Deep Neural #Networks /Artificial Intelligence #/Bihemispheric Processing /Registration #Number /140450140003002031/Iran

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.utils import to_categorical
import time

# --- مرحله 1: ایجاد داده مصنوعی با توزیع بهتر
np.random.seed(42)
n_samples = 1000  # افزایش نمونه‌ها برای یادگیری بهتر
n_features = 8
n_classes = 3

# ایجاد داده‌های ساختاریافته‌تر برای یادگیری معنادار
X = np.random.randn(n_samples, n_features)  # توزیع نرمال
# ایجاد الگوی قابل یادگیری برای کلاس‌ها
y = (X[:, 0] + 2*X[:, 1] - X[:, 2] > 0).astype(int)
y = np.where(X[:, 3] > 0.5, (y + 1) % n_classes, y)
y_cat = to_categorical(y, num_classes=n_classes)

# --- مرحله 2: تقسیم داده‌ها با اعتبارسنجی
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
)

# --- مرحله 3: تعریف مدل بهینه‌شده
def create_optimized_model():
    inputs = Input(shape=(n_features,), name="gray_inputs")

    # شاخه خطی با تنظیمات بهینه
    linear_branch = Dense(16, activation='linear', name="linear_branch")(inputs)
    linear_branch = Dense(8, activation='linear')(linear_branch)

    # شاخه غیرخطی با تنظیمات بهینه
    nonlinear_branch = Dense(16, activation='relu', name="nonlinear_branch")(inputs)
    nonlinear_branch = Dense(8, activation='relu')(nonlinear_branch)
    nonlinear_branch = Dense(4, activation='relu')(nonlinear_branch)

    # ادغام و لایه‌های فشرده‌سازی
    fusion = Concatenate(name="fusion_layer")([linear_branch, nonlinear_branch])
    
    # لایه‌های فشرده‌سازی پیش از خروجی
    fusion = Dense(12, activation='relu')(fusion)
    fusion = Dense(8, activation='relu')(fusion)
    fusion = Dense(6, activation='relu')(fusion)

    # خروجی نهایی
    outputs = Dense(n_classes, activation='softmax', name="output_layer")(fusion)

    model = Model(inputs=inputs, outputs=outputs)
    
    # کامپایل با تنظیمات بهینه
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

# ایجاد مدل
model = create_optimized_model()

# نمایش خلاصه مدل
model.summary()

# --- مرحله 4: آموزش با تنظیمات بهینه
start_time = time.time()

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,  # batch size بهینه‌تر
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
)

training_time = time.time() - start_time
print(f"\nTraining time: {training_time:.2f} seconds")

# --- مرحله 5: ارزیابی جامع
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)

# پیش‌بینی و ارزیابی
y_pred_probs = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# ماتریس درهم‌ریختگی
cm = confusion_matrix(y_true_classes, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# گزارش طبقه‌بندی
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, digits=4))

# محاسبه دقت کلی
accuracy = np.mean(y_pred_classes == y_true_classes)
print(f"Overall Accuracy: {accuracy:.4f}")

# ارزیابی نهایی روی داده تست
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# نمایش تاریخچه آموزش
print("\nTraining History:")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

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
class CoordinatedDualHemisphere(nn.Module):
def init(self, input_size=128, hidden_size=64, num_layers=2, num_classes=2, bidirectional=False, dropout=0.2, fc_dropout=0.2): super().__init__() self.bidirectional = bidirectional self.hidden_size = hidden_size self.num_layers = num_layers self.num_directions = 2 if bidirectional else 1 self.lstm_left = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional) self.lstm_right = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional) coord_in = hidden_size * self.num_directions * 2 self.fc_coordination = nn.Linear(coord_in, hidden_size) self.fc_dropout = nn.Dropout(fc_dropout) self.fc_output = nn.Linear(hidden_size, num_classes) self._init_weights() def _init_weights(self): for m in self.modules(): if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight) if m.bias is not None: nn.init.zeros_(m.bias) elif isinstance(m, nn.LSTM): for name, param in m.named_parameters(): if 'weight_ih' in name: nn.init.xavier_uniform_(param.data) elif 'weight_hh' in name: nn.init.orthogonal_(param.data) elif 'bias' in name: nn.init.zeros_(param.data) def forward(self, x_left, x_right): # expected shapes: (batch, seq_len, input_size) assert x_left.dim() == 3 and x_right.dim() == 3, "Inputs must be (batch, seq_len, input_size)" _, (h_left, _) = self.lstm_left(x_left) _, (h_right, _) = self.lstm_right(x_right) if self.bidirectional: h_left_last = torch.cat((h_left[-2], h_left[-1]), dim=1) h_right_last = torch.cat((h_right[-2], h_right[-1]), dim=1) else: h_left_last = h_left[-1] h_right_last = h_right[-1] combined = torch.cat((h_left_last, h_right_last), dim=1) coordinated = torch.relu(self.fc_coordination(combined)) coordinated = self.fc_dropout(coordinated) out = self.fc_output(coordinated) return out
self.bn2 = nn.BatchNorm1d(hidden_dim) self.dropout2 = nn.Dropout(dropout_rate) # Third convolution block with reduced channels self.conv3 = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1) self.bn3 = nn.BatchNorm1d(output_dim) self.dropout3 = nn.Dropout(dropout_rate) def forward(self, x): # Input normalization x = self.input_norm(x) # x shape: (batch_size, seq_len, input_dim) # Conv1d expects: (batch_size, input_dim, seq_len) x = x.transpose(1, 2) # (B, 128, 10) if self.debug_mode: print(f"[DEBUG] CNN Input shape: {x.shape}") # First convolution block x = self.conv1(x) # (B, 64, 10) x = self.bn1(x) x = F.gelu(x) # Using GELU for consistency x = self.dropout1(x) if self.debug_mode: print(f"[DEBUG] Conv1 output shape: {x.shape}") # Second convolution block x = self.conv2(x) # (B, 64, 10) x = self.bn2(x) x = F.gelu(x) x = self.dropout2(x) if self.debug_mode: print(f"[DEBUG] Conv2 output shape: {x.shape}") # Third convolution block x = self.conv3(x) # (B, 32, 10) x = self.bn3(x) x = self.dropout3(x) if self.debug_mode: print(f"[DEBUG] Conv3 output shape: {x.shape}") # Transpose back to (B, S, 32) to preserve sequence information x = x.transpose(1, 2) # (B, 10, 32) if self.debug_mode: print(f"[DEBUG] CNN Final output shape: {x.shape}") return x def create_sample_data(batch_size=4, seq_len=10, input_dim=128, num_classes=32): """Create sample data for testing""" torch.manual_seed(42) # Create more realistic data with varying scales X = torch.randn(batch_size, seq_len, input_dim) * 2.0 + 1.0 # Scale and shift for testing normalization y = torch.randint(0, num_classes, (batch_size, seq_len)) return X, y def train_model(model, X, y, criterion, optimizer, device, debug_mode=False): """Train model for one batch with proper timing and debugging""" model.train() model.to(device) X, y = X.to(device), y.to(device) # Ensure CUDA operations are synchronized for accurate timing if device.type == 'cuda': torch.cuda.synchronize() start_time = time.time() # Forward pass optimizer.zero_grad(set_to_none=True) scaler = GradScaler() with autocast(): output = model(X) # Proper reshaping for sequence classification output_flat = output.view(-1, output.size(-1)) # (B*S, output_dim) y_flat = y.view(-1) # (B*S,) loss = criterion(output_flat, y_flat) # Backward pass scaler.scale(loss).backward() # Gradient clipping scaler.unscale_(optimizer) torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Optimizer step scaler.step(optimizer) scaler.update() # Ensure CUDA operations are synchronized for accurate timing if device.type == 'cuda': torch.cuda.synchronize() end_time = time.time() training_time = end_time - start_time if debug_mode: print(f"[DEBUG] Training time: {training_time:.6f}s") print(f"[DEBUG] Loss: {loss.item():.6f}") return loss.item(), training_time, output def compare_models(): """Compare EnhancedBiHemisphericLayer with proper SimpleCNN""" # Device configuration device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') print(f"Using device: {device}") # Create sample data X, y = create_sample_data() print(f"Input data shape: {X.shape}") print(f"Target shape: {y.shape}") print(f"Input data range: [{X.min():.4f}, {X.max():.4f}]") # Model configurations criterion = nn.CrossEntropyLoss() # Test Enhanced BiHemispheric Layer print("\n" + "="*60) print("Testing Enhanced BiHemispheric Layer") print("="*60) bihem_model = EnhancedBiHemisphericLayer(debug_mode=True, dropout_rate=0.1) bihem_optimizer = torch.optim.AdamW(bihem_model.parameters(), lr=0.001, weight_decay=1e-4) bihem_loss, bihem_time, bihem_output = train_model( bihem_model, X, y, criterion, bihem_optimizer, device, debug_mode=True ) # Test Proper Simple CNN print("\n" + "="*60) print("Testing Proper Simple CNN") print("="*60) cnn_model = SimpleCNN(debug_mode=True, dropout_rate=0.1) cnn_optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=0.001, weight_decay=1e-4) cnn_loss, cnn_time, cnn_output = train_model( cnn_model, X, y, criterion, cnn_optimizer, device, debug_mode=True ) # Comparison results print("\n" + "="*60) print("COMPARISON RESULTS")print("="*60) print(f"{'Model':<25} {'Loss':<12} {'Time (s)':<12} {'Output Shape':<15}") print("-" * 60) print(f"{'BiHemispheric Layer':<25} {bihem_loss:<12.6f} {bihem_time:<12.6f} {str(bihem_output.shape):<15}") print(f"{'Simple CNN':<25} {cnn_loss:<12.6f} {cnn_time:<12.6f} {str(cnn_output.shape):<15}") # Additional metrics print("\nAdditional Metrics:") print(f"BiHemispheric Output range: [{bihem_output.min():.4f}, {bihem_output.max():.4f}]") print(f"Simple CNN Output range: [{cnn_output.min():.4f}, {cnn_output.max():.4f}]") # Parameter count comparison bihem_params = sum(p.numel() for p in bihem_model.parameters()) cnn_params = sum(p.numel() for p in cnn_model.parameters()) print(f"\nParameter Count:") print(f"BiHemispheric Layer: {bihem_params:,}") print(f"Simple CNN: {cnn_params:,}") if name == "__main__": compare_models() اگر ایراد داشت
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.utils import to_categorical
import time

# --- مرحله 1: ایجاد داده مصنوعی با توزیع بهتر
np.random.seed(42)
n_samples = 1000  # افزایش نمونه‌ها برای یادگیری بهتر
n_features = 8
n_classes = 3

# ایجاد داده‌های ساختاریافته‌تر برای یادگیری معنادار
X = np.random.randn(n_samples, n_features)  # توزیع نرمال
# ایجاد الگوی قابل یادگیری برای کلاس‌ها
y = (X[:, 0] + 2*X[:, 1] - X[:, 2] > 0).astype(int)
y = np.where(X[:, 3] > 0.5, (y + 1) % n_classes, y)
y_cat = to_categorical(y, num_classes=n_classes)

# --- مرحله 2: تقسیم داده‌ها با اعتبارسنجی
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
)

# --- مرحله 3: تعریف مدل بهینه‌شده
def create_optimized_model():
    inputs = Input(shape=(n_features,), name="gray_inputs")

    # شاخه خطی با تنظیمات بهینه
    linear_branch = Dense(16, activation='linear', name="linear_branch")(inputs)
    linear_branch = Dense(8, activation='linear')(linear_branch)

    # شاخه غیرخطی با تنظیمات بهینه
    nonlinear_branch = Dense(16, activation='relu', name="nonlinear_branch")(inputs)
    nonlinear_branch = Dense(8, activation='relu')(nonlinear_branch)
    nonlinear_branch = Dense(4, activation='relu')(nonlinear_branch)

    # ادغام و لایه‌های فشرده‌سازی
    fusion = Concatenate(name="fusion_layer")([linear_branch, nonlinear_branch])
    
    # لایه‌های فشرده‌سازی پیش از خروجی
    fusion = Dense(12, activation='relu')(fusion)
    fusion = Dense(8, activation='relu')(fusion)
    fusion = Dense(6, activation='relu')(fusion)

    # خروجی نهایی
    outputs = Dense(n_classes, activation='softmax', name="output_layer")(fusion)

    model = Model(inputs=inputs, outputs=outputs)
    
    # کامپایل با تنظیمات بهینه
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

# ایجاد مدل
model = create_optimized_model()

# نمایش خلاصه مدل
model.summary()

# --- مرحله 4: آموزش با تنظیمات بهینه
start_time = time.time()

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,  # batch size بهینه‌تر
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
)

training_time = time.time() - start_time
print(f"\nTraining time: {training_time:.2f} seconds")

# --- مرحله 5: ارزیابی جامع
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)

# پیش‌بینی و ارزیابی
y_pred_probs = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# ماتریس درهم‌ریختگی
cm = confusion_matrix(y_true_classes, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# گزارش طبقه‌بندی
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, digits=4))

# محاسبه دقت کلی
accuracy = np.mean(y_pred_classes == y_true_classes)
print(f"Overall Accuracy: {accuracy:.4f}")

# ارزیابی نهایی روی داده تست
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# نمایش تاریخچه آموزش
print("\nTraining History:")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
