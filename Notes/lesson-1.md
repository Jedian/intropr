Short summary - Intro PR

1 -> Introduction - Pattern Recognition Pipeline

| Rec | -> |Preprocessing| -> |Feature Extraction| -> |Classification|

Train and test phase, same pipeline.

We saw many different applications. (speech dialog, autonomous driving, emotion recognition)

2 -> Recording 
- Factors of Influence
    - Which sensors are suitable for this classification task?
    - Same conditioning for both samples, to avoid the pattern to be the
    environment, instead of the subject.
- Analogic/Digital conversion
    - Sampling and quantization
    - For most problems, frequencies are bandwidth limited and then we can put
    a limit to the sampling (like how human ear is limited). - Sampling Theory
    - Fourier Transform (We should be able to write it down and explain what it does (convolution, blabla) (matrix mult?) )
    - Inverse fourier transform (the same)
    - Sampling Theorem - we have to at least sample the double some thing.

    - Quantization
    - Adding error and noise to the data


3 -> Preprocessing
- To create a transform that maps the input signal into the input signal domain. 
- Histogram Equalization
- Thresholding and binarization 
    - Gaussian
    - Otsus method (Difference between this and gaussian)
    - Unimodal ditributions
- Filtering
    - Linear shift-invariant filters
        - Box/mean filter ((1 1 1) (1 1 1) (1 1 1))
        - Gauss ((1 2 1) (2 4 2) (1 2 1))
    - Convolution in frequency domain means multiplication in image domain and vice-versa.
    - We should be able to write down the definition of a discrete convolution.
    - Non linear filter - Median
    - Edge detection (-1 0 1)
        - Sobel filter
        - Canny edge detector
        - Laplace filter (should be able to write down the matrix and explain the concept)
    - Recursive filters (not too many questions, efficient, because can be inplace)
    - Homomorphic filters
    - Morphologic Filters (dilatation, erosion, dila+eros, opening and closing)
- Normalization
    - Geometric moments
    - Central moments
    - Orientation Normalization --- PCA - should be able to explain PCA

4 -> Feature Extraction
- Heuristic
    - Fourier Basis
    - SHould be able to explain (2d fourier space -> if you have a point in some place, it means a frequency in image space) (P(-1, 1) ======== |__/__/__/__/)
    - Walsh basis
    - Haur(Hour/Huur) Basis
    - Hu moments
    - LPC coefficients - speech processing
    - Short-time FT 
    - Wavelet transform - Tradeoff between high-frequency very well or locate the loadfnwak ?

- Analytic
    - Principal component Analysis
    - Linear Discriminant Analysis

5 -> Numerical Optimization
    - Gradient Descent - always got into negtive direction of gradient
    - Coordinate Descent - (one coordinate at a time?)
        - Problemas insoluveis: paralelogramos- pesquisar pra desenhar provavelmente

6 -> Feature Selection
    - Optimization function
    - Optimization method
    - bayesian, entropy-based
    - Defining up and low bounds for the error.
    - Branch and bound

7 -> Classification
    - Stathistical classifcation
    - Optimal decision rule (individual costs) minimize the overall risk
    - 0-1 cost decision function
    - Bayesian classification -> gaussian classifier
    - Polynomial Classifier - explain how it works, feed polynomial and stuff
    - Non-parametric classifier (direct, histogram, kde, k-nn classifier)

    
