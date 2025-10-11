
#PCT/IR2025/050026.nader.malki
class HyperbolicBlock(Layer):
    def init(self, units=8, kwargs):
        super(HyperbolicBlock, self).__init__(kwargs)
        self.dense = Dense(units, activation='linear')
    
    def call(self, x):
        x = self.dense(x)
        return hyperbolic_activation(x)

# ----------------------------
# 4. مکانیزم توجه پویا بین بلوک‌ها
# ----------------------------
class CrossAttentionFusion(Layer):
    def init(self, kwargs):
        super(CrossAttentionFusion, self).__init__(kwargs)
        self.w_q = Dense(8)  # query
        self.w_k = Dense(8)  # key
        self.w_v = Dense(8)  # value

    def call(self, linear_out, hyper_out):
        # query از بلوک خطی، key/value از بلوک هذلولوی
        Q = self.w_q(linear_out)
        K = self.w_k(hyper_out)
        V = self.w_v(hyper_out)
        
        # توجه ساده
        attn_scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(8.0)
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        attended = tf.matmul(attn_weights, V)
        
        # فیوژن نهایی
        fused = linear_out + attended
        return fused

# ----------------------------
# 5. مونتاژ مدل
# ----------------------------
inputs = Input(shape=(4,), name='input')

# مرحله 1: پردازش ورودی
proj1, proj2, modulated = InputProcessor(proj_dim=8, name='input_processor')(inputs)

# مرحله 2: دو بلوک موازی
linear_branch = Dense(8, activation='linear', name='linear_branch')(modulated)
hyper_branch = HyperbolicBlock(units=8, name='hyperbolic_branch')(modulated)

# مرحله 3: تبادل اطلاعات با توجه پویا
fused = CrossAttentionFusion(name='cross_attention')([linear_branch, hyper_branch])

# مرحله 4: خروجی
outputs = Dense(3, activation='softmax', name='output')(fused)

model = Model(inputs=inputs, outputs=outputs)

# کامپایل
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ معماری مدل با موفقیت ساخته شد!")
model.summary()

# ----------------------------
# 6. آموزش و تست
# ----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=80,
    batch_size=8,
    verbose=1
)

# ارزیابی
test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"\n🎯 دقت نهایی تست: {test_acc:.4f}")

# تست یک نمونه
sample = X_test[:1]
pred = model.predict(sample)
print(f"\n🧪 پیش‌بینی نمونه: {pred[0]} → کلاس: {np.argmax(pred)}")
`

---

