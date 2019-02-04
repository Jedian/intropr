import numpy as np

def calculate_R_Distance(Rx,Ry,k):
    #TODO: Check if Rx size == Ry size - Calculate R distance (ring-like)
    if len(Rx) != len(Ry):
        raise Exception("Size of RX differs of size of RY")

    dist = 0
    for i in range(k):
        dist += abs(Rx[i] - Ry[i])

    dist/=(k+0.0)
    return dist

def calculate_Theta_Distance(Thetax,Thetay,k):
    #TODO: Check if thetax size == thetay size - Calculate D-Theta distance (fan-like)
    if len(Thetax) != len(Thetay):
        raise Exception("Size of ThetaX differs of size of ThetaY")

    lx = 0
    ly = 0
    for i in range(k):
        lx += Thetax[i]/(k+0.0)
        ly += Thetay[i]/(k+0.0)

    lyy = 0
    lxx = 0
    lxy = 0
    for i in range(k):
        lyy += (Thetay[i] - ly)**2
        lxx += (Thetax[i] - lx)**2
        lxy += (Thetax[i] - lx) * (Thetay[i] - ly)

    return (1 - ((lxy**2)/(lxx*lyy)))*100
