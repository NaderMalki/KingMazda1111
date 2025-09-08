import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# مرحله 1: شبیه‌سازی مواد هوشمند
def simulate_piezoelectric(d33, sigma, thickness, resistance):
    """ شبیه‌سازی ولتاژ و جریان مواد پیزوالکتریک """
    voltage = d33 * sigma * thickness  # ولتاژ تولیدی
    current = voltage / resistance  # جریان تولیدی
    return voltage, current

# خواص مواد
d33 = 3e-12  # ثابت پیزوالکتریک (C/N)
sigma = 1e5  # فشار واردشده (N/m^2)
thickness = 2e-3  # ضخامت مواد (m)
resistance = 1e3  # مقاومت مدار (Ohm)

# شبیه‌سازی مواد هوشمند
voltage, current = simulate_piezoelectric(d33, sigma, thickness, resistance)
print(f"ولتاژ تولیدشده: {voltage:.6f} V")
print(f"جریان تولیدشده: {current:.6f} A")

# مرحله 2: شبیه‌سازی تغییر شکل مواد
def spring_damper(F, k, c, x0, v0, t, m=1.0):
    """ شبیه‌سازی رفتار فنر-دمپر برای تغییر شکل مواد """
    dt = t[1] - t[0]
    x = np.zeros(len(t))
    v = np.zeros(len(t))
    x[0], v[0] = x0, v0

    for i in range(1, len(t)):
        a = (F - k * x[i-1] - c * v[i-1]) / m  # اصلاح معادله شتاب
        v[i] = v[i-1] + a * dt
        x[i] = x[i-1] + v[i] * dt

    return x

# پارامترهای فنر-دمپر
F = 10  # نیروی واردشده (N)
k = 1000  # ثابت فنر (N/m)
c = 10  # ضریب دمپر (Ns/m)
t = np.linspace(0, 1, 100)  # زمان (s)

# شبیه‌سازی تغییر شکل
x = spring_damper(F, k, c, x0=0, v0=0, t=t)

# نمایش تغییر شکل
plt.plot(t, x)
plt.title("تغییر شکل مواد هوشمند")
plt.xlabel("زمان (s)")
plt.ylabel("تغییر شکل (m)")
plt.grid()
plt.show()

# مرحله 3: شبکه عصبی برای پردازش سیگنال‌ها
X_data = np.random.rand(100, 2)  # 100 نمونه با 2 ویژگی
y_data = np.random.rand(100, 1)  # خروجی هدف

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(X_data, y_data, epochs=100, batch_size=10, verbose=0)

new_data = np.array([[voltage, current]])
prediction = model.predict(new_data)
print(f"پیش‌بینی خروجی: {prediction[0][0]:.2f}")
