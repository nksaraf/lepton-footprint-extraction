# segment.py - Segments an input image.
# Cornell University CS 4670/5670: Intro Computer Vision
import math
import numpy as np
import scipy
import scipy.sparse
from scipy.sparse import spdiags
from scipy.signal import convolve2d
import cv2
 
#########################################################
###    Part A: Image Processing Functions
######################################################### 

# TODO:PA2 Fill in this function
def normalizeImage(cvImage, minIn, maxIn, minOut, maxOut):
    '''
    Take image and map its values linearly from [min_in, max_in]
    to [min_out, max_out]. Assume the image does not contain 
    values outside of the [min_in, max_in] range.
    
    Parameters:
        cvImage - a (m x n) or (m x n x 3) image.
        minIn - the minimum of the input range
        maxIn - the maximum of the input range
        minOut - the minimum of the output range
        maxOut - the maximum of the output range
        
    Return:
        renormalized - the linearly rescaled version of cvImage.
    '''
    return ((cvImage - minIn) * (maxOut - minOut) * 1.0/(maxIn - minIn)) + minOut

# TODO:PA2 Fill in this function
def getDisplayGradient(gradientImage):
    """
    Use the normalizeImage function to map a 2d gradient with one
    or more channels such that where the gradient is zero, the image
    has 50% percent brightness. Brightness should be a linear function 
    of the input pixel value. You should not clamp, and 
    the output pixels should not fall outside of the range of the uint8 
    datatype.
    
    Parameters:
        gradientImage - a per-pixel or per-pixel-channel gradient array
                        either (m x n) or (m x n x 3). May have any 
                        numerical datatype.
    
    Return:
        displayGrad - a rescaled image array with a uint8 datatype.
    """
    absoluteImage = np.absolute(gradientImage)
    a = np.max(absoluteImage)
    return normalizeImage(gradientImage, -a, a, 0, 255).astype(np.uint8)

# TODO:PA2 Fill in this function
def takeXGradient(cvImage):
    '''
    Compute the x-derivative of the input image with an appropriate
    Sobel implementation. Should return an array made of floating 
    point numbers.
    
    Parameters:
        cvImage - an (m x n) or (m x n x 3) image
        
    Return:
        xGrad - the derivative of cvImage at each position w.r.t. the x axis.
    
    '''
    if cvImage.ndim == 3:
        imageChannels = np.dsplit(cvImage, 3)
        imageChannels = [takeXGradient(channel.reshape(channel.shape[0:2])) for channel in imageChannels]
        imageChannels = [channel.reshape(channel.shape + (1,)) for channel in imageChannels]
        return np.concatenate(imageChannels, 2)
    else:
        kernel = [[1],[2],[1]]
        kernel2 = [[1, 0, -1]]
        stage1 = convolve2d(cvImage, kernel, 'same')
        return convolve2d(stage1, kernel2, 'same')
    
# TODO:PA2 Fill in this function
def takeYGradient(cvImage):
    '''
    Compute the y-derivative of the input image with an appropriate
    Sobel implementation. Should return an array made of floating 
    point numbers.
    
    Parameters:
        cvImage - an (m x n) or (m x n x 3) image
        
    Return:
        yGrad - the derivative of cvImage at each position w.r.t. the y axis.
    '''
    if cvImage.ndim == 3:
        imageChannels = np.dsplit(cvImage, 3)
        imageChannels = [takeYGradient(channel.reshape(channel.shape[0:2])) for channel in imageChannels]
        imageChannels = [channel.reshape(channel.shape + (1,)) for channel in imageChannels]
        return np.concatenate(imageChannels, 2)
    else:
        kernel = [[1],[0],[-1]]
        kernel2 = [[1, 2, 1]]
        stage1 = convolve2d(cvImage, kernel, 'same')
        return convolve2d(stage1, kernel2, 'same')
    
# TODO:PA2 Fill in this function
def takeGradientMag(cvImage):
    '''
    Compute the gradient magnitude of the input image for each 
    pixel in the image. 
    
    Parameters:
        cvImage - an (m x n) or (m x n x 3) image
        
    Return:
        gradMag - the magnitude of the 2D gradient of cvImage. 
                  if multiple channels, handle each channel seperately.
    '''
    x = takeXGradient(cvImage)
    y = takeYGradient(cvImage)
    return np.sqrt((x*x) + (y*y))

#########################################################
###    Part B: k-Means Segmentation Functions
######################################################### 

# TODO:PA2 Fill in this function
def chooseRandomCenters(pixelList, k):
    """
    Choose k random starting point from a list of candidates.
    
    Parameters:
        pixelList - an (n x 6) matrix 
        
    Return:
        centers - a (k x 6) matrix composed of k random rows of pixelList
    """
    indices = np.random.choice(pixelList.shape[0], k)
    result = []
    for index in indices:
        result.append(pixelList[index])
    return np.array(result)

# TODO:PA2 Fill in this function
def kMeansSolver(pixelList, k, centers=None, eps=0.001, maxIterations=100):
    '''
    Find a local optimum for the k-Means problem in 3D with
    Lloyd's Algorithm
    
    Assign the index of the center closest to each pixel
    to the fourth element of the row corresponding that 
    pixel.
    
    Parameters:
        pixelList - n x 6 matrix, where each row is <H, S, V, x, y, c>,
                    and c is index the center closest to it.
        centers - a k x 5 matrix where each row is one of the centers
                  in the k means algorithm. Each row is of the format
                  <H, S, V, x, y>
        eps - a positive real number user to test for convergence.
        maxIterations - a positive integer indicating how many 
                        iterations may occur before aborting.
    
    Return:
        iter - the number of iterations before convergence.
    '''
    # TODO:PA2 
    # H,S,V, x, and y values into the [0, 1] range.
    pixelList[:,0:3] = pixelList[:,0:3]/255.0
    pixelList[:,3] = pixelList[:,3]/pixelList[-1,3]
    pixelList[:,4] = pixelList[:,4]/pixelList[-1,4]
    # Initialize any data structures you need.

    # END TODO:PA2
    if centers is None:
        centers = chooseRandomCenters(pixelList,k)[:,0:5]
   
    for iter in range(maxIterations):
        # TODO:PA2 Assign each point to the nearest center
        # raise NotImplementedError
        # END TODO:PA2
        
        # TODO:PA2 Recalculate centers
        cent_sum = np.zeros(centers.shape)
        cent_count = np.zeros(centers.shape[0])
        for pixel in pixelList:
            ind = np.argmin([np.linalg.norm(pixel[0:5]-center) for center in centers])
            pixel[5] = ind
            cent_sum[ind] += pixel[0:5]
            cent_count[ind] += 1
        newCenters = np.array([cent_sum[i]/cent_count[i] for i in range(centers.shape[0])])
        # END TODO:PA2
        converge = True
        for i in range(newCenters.shape[0]):
            if (np.linalg.norm(centers[i] - newCenters[i], 2) > eps):
                converge = False
                break
        if converge:
            center = newCenters
            return iter
        else:
            centers = newCenters
    return iter 
      
def convertToHsv(rgbTuples):
    """
    Convert a n x 3 matrix whose rows are RGB tuples into
    an n x 3 matrix whose rows are the corresponding HSV tuples.
    The entries of rgbTuples should lie in [0,1]
    """
    B = rgbTuples[:,0]
    G = rgbTuples[:,1]
    R = rgbTuples[:,2]
    
    hsvTuples = np.zeros_like(rgbTuples)
    H = hsvTuples[:,0]
    S = hsvTuples[:,1]
    V = hsvTuples[:,2]
    
    alpha = 0.5 * (2*R - G - B)
    beta = np.sqrt(3)/2 * (G - B)
    H = np.arctan2(alpha, beta)
    V = np.max(rgbTuples,1)
    
    chroma = np.sqrt(np.square(alpha) + np.square(beta))
    S[V != 0] = np.divide(chroma[V != 0], V[V != 0])
    
    hsvTuples[:,0] = H  
    hsvTuples[:,1] = S  
    hsvTuples[:,2] = V  
    
    return hsvTuples
            
def kMeansSegmentation(cvImage, k, useHsv=True, eps=1e-14):
    """
    Execute a color-based k-means segmentation 
    """
    # Reshape the imput into a list of R,G,B,X,Y,C tuples, where
    # means that a pixel has not yet been assigned to a segment.
    m, n = cvImage.shape[0:2]
    numPix = m*n
    pixelList = np.zeros((numPix,6))
    pixelList[:,0:3] = cvImage.reshape((numPix,3))
    pixelList[:,3] = np.tile(np.arange(n),m)
    pixelList[:,4] = np.repeat(np.arange(m), n)
    
    # Convert the image to hsv.
    if useHsv:
        pixelList[:,:3] = convertToHsv(pixelList[:,:3]/255.)*255
    
    # Initialize k random centers in the color-position space.
    centers = (np.max(pixelList[:,0:5],0)-np.min(pixelList[:,0:5],0))*np.random.random((k,5))+np.min(pixelList[:,0:5],0)

    # Run Lloyd's algorithm until convergence
    iter = kMeansSolver(pixelList, k, eps=eps)
    
    # Color the pixels based on their centers
    if k <= 64:
        colors = np.array(COLORS[:k])
    else:
        colors = np.random.uniform(0,255,(k,3))
    
    R = pixelList[:,0]
    G = pixelList[:,1]
    B = pixelList[:,2]
    centerIndices = pixelList[:,5]
    
    for j in range(k):
       R[centerIndices == j] = colors[j,0] 
       G[centerIndices == j] = colors[j,1]
       B[centerIndices == j] = colors[j,2]
       
    return pixelList[:,:3].reshape(cvImage.shape).astype(np.uint8), iter
       
#########################################################
###    Part C: Normalized Cuts Segmentation Functions
######################################################### 

# TODO:PA2 Fill in this function
def getTotalNodeWeights(W):
    """
    Calculate the total weight of all edges leaving each 
    node.
    
    Parameters:
        W - the m*n x m*n weighted adjecency matrix of a graph
    
    Return:
        d - a vector whose ith component is the total weight
            leaving node i in W's graph.
    """
    d = [np.sum(i) for i in W]
    return d
    
# TODO:PA2 Fill in this function
def approxNormalizedBisect(W, d):
    """
    Find the eigenvector approximation to the normalized cut
    segmentation problem with weight matrix W and diagonal d.
    
    Parameters:
        W - a (n*m x n*m) array of weights (floats)
        d - a n*m vector

    Return:
        y_1 - the second smallest eigenvector of D-W
    """
    D_ = np.diag(1.0/(np.sqrt(d)))
    I = np.identity(len(d))
    A = I - np.dot(D_,W).dot(D_)
    w, vr = scipy.linalg.eigh(A, eigvals = (1,1))
    return np.dot(D_,vr[:,0])

def mag(v):
    return np.sqrt(v.dot(v))

# TODO:PA2 Fill in this function
def getColorWeights(cvImage, r, sigmaF=5, sigmaX=6):
    """
    Construct the matrix of the graph of the input image,
    where weights between pixels i, and j are defined
    by the exponential feature and distance terms.
    
    Parameters:
        cvImage - the m x n uint8 input image
        r - the maximum distance below which pixels are 
            considered to be connected
        sigmaF - the standard deviation of the feature term
        sigmaX - the standard deviation of the distance term
    
    Return:
        W - the m*n x m*n matrix of weights representing how 
            closely each pair of pixels is connected
    
    """
    m, n = cvImage.shape[0:2]
    numPix = m*n
    W = np.zeros((numPix, numPix))
    sigFsq = sigmaF * sigmaF
    sigXsq = sigmaX * sigmaX

    for i in range(numPix):
        for j in range(numPix):
            x1,y1,x2,y2 = (i/n, i%n, j/n, j%n)
            dist = np.linalg.norm([x1-x2,y1-y2])
            if dist <= r:
                cdist = (-np.linalg.norm(cvImage[x1][y1]-cvImage[x2][y2])/(sigFsq*1.0))
                xdist = (-dist/(sigXsq*1.0))
                W[i][j] = np.exp(cdist) * np.exp(xdist)
    return W

# TODO:PA2 Fill in this function
def reconstructNCutSegments(cvImage, y, threshold=0):
    """
    Create an output image that is yellow wherever y > threshold
    and blue wherever y < threshold
    
    Parameters:
        cvImage - an (m x n x 3) BGR image
        y - the (m x n)-dimensional eigenvector of the normalized 
            cut approximation algorithm,
        threshold - the cutoff between pixels in the two segments.
        
    Return:
        segmentedImage - an (n x m x 3) image that is yellows
                         for pixels with values above the threshold
                         and blue otherwise.
    """
    segmentedImage = np.zeros(cvImage.shape)
    m, n = cvImage.shape[0:2]
    for i in range(m):
        for j in range(n):
            if y[(i*n)+j] > threshold:
                segmentedImage[i][j] = [0, 255, 255]
            else:
                segmentedImage[i][j] = [255, 0, 0]
    return segmentedImage.astype(np.uint8)
    
def nCutSegmentation(cvImage, sigmaF=5, sigmaX=6):
    print("Getting Weight Matrix")
    W = getColorWeights(cvImage, 7)
    print(str(W.shape[0]) + "x" + str(W.shape[1]) + " Weight matrix generated")
    d = getTotalNodeWeights(W)
    print("Calculated weight totals")
    y = approxNormalizedBisect(W, d)
    print("Reconstructing segments")
    segments = reconstructNCutSegments(cvImage, y, 0)
    return segments
    
if __name__ == "__main__":
    pass
    # You can test your code here.