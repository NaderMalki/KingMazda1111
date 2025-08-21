/Nader/maleikie/Invention of Deep Neural Networks /Artificial Intelligence /Bihemispheric Processing /Registration Number /140450140003002031/Iran

python
import numpy as np
import matplotlib.pyplot as plt
 
d33 = 3e-1  (C/N)
sigma = 1e5   (N/m^2)
thickness = 2e-3   (m)

voltage = d33 sigma  thickness
print(f": {voltage:.6f} V")

resistance = 1e3   
current = voltage / resistance
print( : {current:.6f} A")

def spring_damper(F, k, c, x0, v0, t):
    dt = t[1] - t[0]
    x = np.zeros(len(t))
    v = np.zeros(len(t))
    x[0], v[0] = x0, v0
    for i in range(1, len(t)):
        a = (F - k  x[i-1] - c v[i-1]) / k
        v[i] = v[i-1] + a dt
        x[i] = x[i-1] + v[i]  dt
    return x

    return x
 
F = 10  (N)
k = 1000    (N/m)
c = 10    (Ns/m)
t = np.linspace(0, 1, 100)   (s)
  
x = spring_damper(F, k, c, x0=0, v0=0, t=t)
plt.plot(t, x)
plt.title
plt.xlabel (s)
plt.ylabel(  (m)
plt.grid()
plt.show
TensorFlow:
python
import tensorflow as tf
import numpy as np


X_data = np.random.rand(100, 2)   100 
y_data = np.random.rand(100, 1)   
140450140003002031/Nader.m
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),  
    tf.keras.layers.Dense(64, activation=relu)
    tf.keras.layers.Dense(1 activation=sigmoid')  
])

model.compile(optimizer=adam loss='mse', metrics=[accurac  مدل
model.fit(X_data, y_data, epochs=100, batch_size=10)

new_data = np.array([[0.5, 0.6]])      
prediction = model.predict(new_data)
print(f {prediction[0][0]:.2f})
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
def simulate_piezoelectric(d33, sigma, thickness, resistance):  
    voltage = d33 sigma  thickness   
    current = voltage / resistance     return voltage, current
d33 = 3e-12   (C/N)
sigma = 1e5   (N/m^2)
thickness = 2e-3 (m)
resistance = 1e3  (Ohm)
voltage, current = simulate_piezoelectric(d33, sigma, thickness, resistance)
print(f {voltage:.6f} V)
print(f: {current:.6f} A)
def spring_damper(F, k, c, x0, v0, t):
    dt = t[1] - t[0]
    x = np.zeros(len(t))
    v = np.zeros(len(t))
    x[0], v[0] = x0, v0
    for i in range(1, len(t)):
        a = (F - k  x[i-1] - c  v[i-1]) / k
        v[i] = v[i-1] + a  dt
        x[i] = x[i-1] + v[i]  dt
    return x
F = 10   (N)
k = 1000  (N/m)
c = 10   (Ns/m)
t = np.linspace(0, 1, 100)   (s)
x = spring_damper(F, k, c, x0=0, v0=0, t=t)
plt.plot(t, x)
plt.title
plt.xlabel s)
plt.ylabel(  (m)
plt.grid()
plt.show()
X_data = np.random.rand(100, 2)       
y_data = np.random.rand(100, 1)


---


