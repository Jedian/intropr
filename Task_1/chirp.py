import numpy as np

def createChirpSignal(samplingrate, duration, freqfrom, freqto, linear=True):
    t = np.linspace(0, duration, samplingrate)
    
    duration = float(duration)

    if linear:
        beta = (freqto - freqfrom)/duration
        phase = 2*np.pi*((freqfrom*t) + (0.5*beta*t*t))

    else:
        if freqfrom*freqto <= 0.0:
            raise Exception("If its an exponential chirp, source and destination frequencies must have the same signal and be nonzero values.")
        if freqfrom == freqto:
            phase = 2*np.pi*freqfrom*t
        else:
            beta = duration/np.log(freqto/freqfrom)
            phase = 2*np.pi*beta*freqfrom*(pow(freqto/freqfrom, t/duration) - 1.0)

    return t, np.sin(phase)
