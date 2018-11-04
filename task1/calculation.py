import math

def pqsolver(p, q):
    aux = p/2.0
    if aux*aux < q:
        raise Exception('Negative radicand! Aborting calculations')

    x1 = -aux + math.sqrt((aux*aux) - q)
    x2 = -aux - math.sqrt((aux*aux) - q)
    return x1, x2

