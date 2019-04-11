Short summary - Intro PR

1 -> Introduction - Pattern Recognition Pipeline

| Rec | -> |Preprocessing| -> |Feature Extraction| -> |Classification|

Train and test phase, same pipeline.

We saw many different applications. (speech dialog, autonomous driving, emotion recognition)

2 -> Recording 
- Factors of Influence
    - Which sensors are suitable for this classification task?
> We gotta compare between cameras, microphones, weighing machine...

    - Same conditioning for both samples, to avoid the pattern to be the
    environment, instead of the subject.
> We have to be sure our data focus on the thing we want to recognize...

- Analogic/Digital conversion
    - Sampling and quantization
> Slides, I think

    - For most problems, frequencies are bandwidth limited and then we can put
    a limit to the sampling (like how human ear is limited). - Sampling Theory
> True

    - Fourier Transform (We should be able to write it down and explain what it does (convolution, blabla) (matrix mult?) )
    - Inverse fourier transform (the same)
> Video aulas, for sure

    - Sampling Theorem - we have to at least sample the double some thing.
> éoq

    - Quantization
> Meio que slides, meio que vaulas

    - Adding error and noise to the data
> ???


3 -> Preprocessing
- To create a transform that maps the input signal into the input signal domain. 
> ???

- Histogram Equalization
> Vaulas deve ser izi. na real que a ideia é só olhar umas imgs no google, mas comofas eu nao sei ainda.

- Thresholding and binarization 
> Thresholding é um tipo de de discretização de intervalos. pode ser usado pra binarização.

    - Gaussian
> Umas coisas muito doidas
> "Gaussian functions arise by composing exponential function with a concave
> quadratic function."
> Aí tem o método do Bimodal Histograms pra thresholding que usa gaussian.
> Pra binarização por exemplo, vc cria dois histogramas modais, e a interseção
> deles é um ótimo thresholder. Funciona melhor em imagens com uma distribuição
> já bimodal.

    - Otsus method (Difference between this and gaussian)
> Minimizar a intra-class variance <---> maximizar a inter-class variance
> Busca exaustiva pelo threshold que minimiza a variance intra-class, definida
> como soma ponderada (weighted sum) of variances of the two classes. Here,
> weights are the probabilities of the two classes separated by certain
> threshold.

    - Unimodal ditributions
> For unimodal distributions, the heuristic is to put a point in the gray value
> with the highest number of occurences, then determine the number of
> occurences for gray value 0. Connect these two points with a straight line
> and the threshold is the intensity of the point on the histogram that has de
> maximum perpendicular distance to the line.

- Filtering
> something about shift variant systems: the clap of the hands on the classroom

    - Linear shift-invariant filters
        - Box/mean filter 1/9((1 1 1) (1 1 1) (1 1 1))
        - Gauss 1/16((1 2 1) (2 4 2) (1 2 1))
> nao sei mt bem o que falar sobre isso, eles fazem uma média mais ou menos ponderada da imagem
> esses ambos servem pra smoothing

    - Convolution in frequency domain means multiplication in image domain and vice-versa.
> talvez isso sirva pra agilizar uma transformação com grandes kerneis...

    - We should be able to write down the definition of a discrete convolution.
> H[j][k] = soma de u: -inf -> inf, soma de v: -inf -> inf, F[u][v] * G[j-u][k-v]

    - Non linear filter - Median
> non-linear smoothing, slow(sorting) and *cannot* be implemented via convolution.
> preserves frequency details, works well if less than 50% of the neighboring pixels are corrupted by noise.
> not really good in gaussian noise.
> amazingly good in salt & pepper noise.

    - Edge detection (-1 0 1)
> susceptible to noise
> tradeoff between edge strength and noise reduction

        - Sobel filter
> Sobel is a common edge detection mask:
> (-1, 0, 1)
> (-2, 0, 2)
> (-1, 0, 1)

        - Canny edge detector
> optimal edge detection method
> systematically cleans noise effects
> slower
> steps:
>> convert to gray scale
>> smooth gaussian filter
>> sobel in both directions
>> discretizing edges into four groups based on orientation
>> non-maximum suppression

        - Laplace filter (should be able to write down the matrix and explain the concept)
> (0, 1, 0) (4 neighbours)
> (1, -4, 1)
> (0, 1, 0)
> look for zero crossings of the second derivative

    - Recursive filters (not too many questions, efficient, because can be inplace)
> They work as a sliding window. (example of average tipo surf)

    - Homomorphic filters
> all i understood was the state machine from non linear to linear

    - Morphologic Filters (dilatation, erosion, dila+eros, opening and closing)
> erosion: only paint when all the filter area is covered
> dilation: paint whenever any of the area is convered
> opening: erosion followed by dilation (eliminates spikes and cut off "bridges")
> bolas e ripas
> closing: dilation followed by erosion (eliminates bays along the boundary and fills up holes)
> botão de roupa furado
> opening and erosion remove salt noise
> dilation and closing make it bigger
> the reverse hapens for pepper noise

- Normalization
> to standardize the pose, orientation, size of the relevant objects to make it
> more time or space efficient and/or more reliable.

    - Geometric moments
> used to gather information about the center and the orientation of an image.

    - Central moments
> almost the same, but using the centroid

    - Orientation Normalization --- PCA - should be able to explain PCA
> Use PCA to compute direction and length of axis of ellipse given by covariance matrix.
> ??

4 -> Feature Extraction
- Heuristic
    - Fourier Basis
    - SHould be able to explain (2d fourier space -> if you have a point in some place, it means a frequency in image space) (P(-1, 1) ======== |__/__/__/__/)
> Applying FT to images and comparing them in fourier domain.

    - Walsh basis
> applying walsh functions is cheaper than sinusoidals as FT, and easily to compress and reconstruct.

    - Haur(Hour/Huur) Basis
> X

    - Hu moments
> a collection of moments can be used as a feature vector
> first order moments: size, mass, area, volume
> second order: related to variance
> third order: simmetry or distribution
> fourth: tall and skinny or short and squat?

    - LPC coefficients - speech processing
> X

    - Short-time FT 
> Shorter FT, with shorter windows, overlapping windows. Tradeoff between time resolution and frequency resolution.

    - Wavelet transform - Tradeoff between high-frequency very well or locate the loadfnwak ?
> X

- Analytic
    - Principal component Analysis
> the feature vector should describe best the variation of the original data
> and spread across. Maximize the square distance of all pair of feature
> vectors.
> we get a eigenvector problem solvable via SVD (singular value decomposition)
> adidas problem

    - Linear Discriminant Analysis
> clusters should be compact
> cluster centers should be far apart

5 -> Numerical Optimization
    - Gradient Descent - always got into negtive direction of gradient
> converges to the closest local minimum
> global only for unimodal functions
> problems: stopping criteria not clearly defined
> well-known problematic behavior: zig-zagging

    - Coordinate Descent - (one coordinate at a time?)
>

        - Problemas insoluveis: paralelogramos- pesquisar pra desenhar provavelmente

6 -> Feature Selection
> Curse of dimensionality >> we need to select the best subset of features!
> optimal subset: d'' < d features which minimizes the probability of misclassification
> property: no subset with k <= d'' has a smaller probability of misclassification
> many algorithms, characterized by a objective function (goodness of a subset)
> and a optimization method used to search the space of possible subsets

    - Optimization function
> should be simple to evaluate
> should approximate the misclassification error
> should avoid complete testing
> some examples: minimize the error rate
> maximize the bayesian distance (bayes: condiitonal probability)
> minimize the conditional entropy
> maximize the mutual information

    - Optimization method
> searching
> approaches: random, exhaustive, greedy, branch and bound
    - bayesian, entropy-based
    - Defining up and low bounds for the error.
    - Branch and bound

7 -> Classification
    - Stathistical classifcation
> compute the risk associated with the classification of a pattern
> compute the decision rule by minimizing the total risk leading to the optimal classifier

    - Optimal decision rule (individual costs) minimize the overall risk
> calculate the probability of confusion, and using it to minimize the overall risk

    - 0-1 cost decision function
> enforced decision: no rejection possible

    - Bayesian classification -> gaussian classifier
> bayesian is optimal if (0,1) is used.
> bayesian minimizes the error probability

> a gaussian classifier is a bayesian with gaussian distributed class-conditional feature vectors
> it uses maximum likelihood estimation
> often too many parameters, but can be simplified by sharing parameters.

    - Polynomial Classifier - explain how it works, feed polynomial and stuff
> no explicit use of statistical information
> approximation of the ideal decision function by a polinomial

    - Non-parametric classifier (direct, histogram, kde, k-nn classifier)
> methods valid for many, possibly _all_ pdfs.
> requires often the storage of the entire sample set
> histograms (?) parece facil mas n tendi no slide
> probabilidade de knn errar: pnn; probabilidade bayes errar: pbayes
> pbayes <= pnn <= 2 * pbayes

    
