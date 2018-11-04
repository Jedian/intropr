import numpy as np

def createTriangleSignal(samples, frequency, kMax):
    t = np.linspace(0, 1, samples)
    ff = np.linspace(0, 0, samples)
    for k in range(kMax+1):
        m = pow(-1, k)
        ff += m * np.sin(2*np.pi*((2*k) + 1)*t*frequency)/pow((2*k)+1, 2)
    ff *= 8/pow(np.pi, 2)
    return t, ff

def createSquareSignal(samples, frequency, kMax):
    t = np.linspace(0, 1, samples)
    ff = np.linspace(0, 0, samples)
    for k in range(1, kMax+1):
        ff += np.sin(2*np.pi*((2*k) - 1)*t*frequency)/((2*k)-1)
    ff *= 4/np.pi
    return t, ff


def createSawtoothSignal(samples, frequency, kMax, amplitude=1.0):
    t = np.linspace(0, 1, samples)
    ff = np.linspace(0, 0, samples)
    for k in range(1, kMax+1):
        ff += np.sin(2*np.pi*k*t*frequency)/k
    ff = amplitude/2 - ((amplitude/np.pi) * ff)
    return t, ff
