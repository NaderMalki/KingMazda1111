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

