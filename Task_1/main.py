import calculation
import chirp
import decomposition
import matplotlib.pyplot as plt

def test_pqsolver():
    print("Testing pqsolver with valid values (p = 16, q = 5):")
    print("Answer: x1 = %.5f, x2 = %.5f" % calculation.pqsolver(16, 5))
    print

    print("Testing pqsolver with valid values (p = 16, q = 15):")
    print("Answer: x1 = %.5f, x2 = %.5f" % calculation.pqsolver(16, 15))
    print

    print("Testing pqsolver with invalid values (p = 16, q = 150):")
    try:
        print("Answer: x1 = %.5f, x2 = %.5f -- should not have worked! Not real answer expected!" % calculation.pqsolver(16, 150))
    except Exception as e:
        print("Occured an exception, as expected: " + str(e))
    print
        
def test_chirp():
    print("Testing chirp with samplingrate = 200, duration = 1, freq from 1 to 10")
    print
    x1, y1 = chirp.createChirpSignal(200, 1, 1, 10, linear=True)
    x2, y2 = chirp.createChirpSignal(200, 1, 10, 1, linear=False)

    fig, ((plt1, plt2)) = plt.subplots(2, 1)
    plt1.title.set_text('Linear chirp')
    plt1.plot(x1, y1)
    plt2.title.set_text('Exponential chirp')
    plt2.plot(x2, y2)
    plt.show()

def test_fdecomposition():
    print("Testing fourier decompositions with samples = 200, freq = 2, kmax = 10000, [amplitude] = 1")
    print
    x1, y1 = decomposition.createTriangleSignal(200, 2, 10000)
    x2, y2 = decomposition.createSquareSignal(200, 2, 10000)
    x3, y3 = decomposition.createSawtoothSignal(200, 2, 10000)

    fig, ((plt1, plt2, plt3)) = plt.subplots(3, 1)
    plt1.title.set_text('Triangle signal')
    plt1.plot(x1, y1)
    plt2.title.set_text('Square signal')
    plt2.plot(x2, y2)
    plt3.title.set_text('Sawtooth signal')
    plt3.plot(x3, y3)
    plt.show()

if __name__ == "__main__":
    print('----Testing pqsolver----')
    test_pqsolver()

    print('----Testing chirp----')
    test_chirp()

    print('----Testing fdecomposition----')
    test_fdecomposition()
    
