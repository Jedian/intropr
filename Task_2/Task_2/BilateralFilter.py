import numpy as np

def computeGeometricCloseness(i, j, x, y,sigma_d):
    #TODO: Compute the geometric closeness
    p = computeEuclidianDistance(i, j, x, y)/sigma_d
    p = p*p

    return np.exp(-0.5*p)
    
def computeEuclidianDistance(i, j, x, y):
    #TODO: Compute the euclidian distance
    difx = i-x;
    dify = j-y;
    
    return np.sqrt(difx*difx + dify*dify)
   
def computeIntensityDistance(img, i, j, x, y):
    #TODO: Compute the intensity difference (absolute value)
    dif = img[i][j] - img[x][y]
    
    return abs(dif)

def computePhotometricDistance(img, i, j, x, y,sigma_r):
    #TODO: Compute the photometric distance
    p = computeIntensityDistance(img, i, j, x, y)/sigma_r
    p = p*p

    return np.exp(-0.5*p)

def bilateralFilterHelper (img, x, y, width, sigma_d, sigma_r):
    # Compute the bilateral filtered image. Do not filter at the image boundaries!
    # Use the functions defined above
    # Hint: if (filterRunsOverBoundary) do this - else: apply algorithm
    if (x-width) < 0 or (y-width) < 0 or (x+width) >= img.shape[1] or (y+width) >= img.shape[0]:
        return img[y][x]
    else:
        res = 0
        knorm = 0
        for i in range(x-width, x+width+1, 1):
            for j in range(y-width, y+width+1, 1):
                c = computeGeometricCloseness(j, i, y, x, sigma_d)
                s = computePhotometricDistance(img, j, i, y, x, sigma_r)
                res += img[j][i]*c*s
                knorm += c*s

    # talvez precisa colocar a normalizacao aqui, ctz
    return res/knorm

def bilateralFilter(img, width, sigma_d, sigma_r):
    
    result = img[:]
    
    for  i in range(0,img.shape[1],1):
        for j in range(0,img.shape[0],1):    
                 
            result[j,i] = bilateralFilterHelper(img, i, j,width, sigma_d, sigma_r)
            
        print(i, j)
    return result
   
        
