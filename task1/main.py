import calculation
import chirp
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
    x1, y1 = chirp.chirp(200, 1, 1, 10, linear=True)
    x2, y2 = chirp.chirp(200, 1, 1, 10, linear=False)

    fig, ((plt1, plt2)) = plt.subplots(2, 1)
    plt1.title.set_text('Linear chirp')
    plt1.plot(x1, y1)
    plt2.title.set_text('Exponential chirp')
    plt2.plot(x2, y2)
    plt.show()

if __name__ == "__main__":
    print('----Testing pqsolver----')
    test_pqsolver()

    print('----Testing chirp----')
    test_chirp()
    
