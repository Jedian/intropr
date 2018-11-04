import sys
import matplotlib.pyplot as plt
from chirp import chirp as mychirp
from scipy.signal import chirp
import numpy as np

t = np.linspace(0, 1, 200)

# t, freqfrom, duration, freqto
w = chirp(t, 1, 1, 10, method='li')
w2 = chirp(t, 1, 1, 10, method='log')
# t, duration, freqfrom, freqto
t1, w3 = mychirp(200, 1, 1, 10, True)
t2, w4 = mychirp(200, 1, 1, 10, False)

fig, ((plt1, plt2), (plt3, plt4)) = plt.subplots(2, 2)

plt1.plot(t, w)
plt2.plot(t, w2)
plt3.plot(t1, w3)
plt4.plot(t2, w4)

plt.show()
