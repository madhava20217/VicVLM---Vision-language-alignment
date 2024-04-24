# README

VicVLM is a novel algorithm for vision-text alignment that builds upon MAMO (Zhao et al.), MaskVLM (Kwon et al.) and VICReg (Bardes et al.). It aims to be computationally more efficient while providing performance exceeding or on par with MAMO.

## Benchmarks

Pretraining, training and testing happened on Flickr30k. The reported results are for the testing split of Flickr30k.

A ViT-S model was used as the image encoder and a BERT-S was used as the text encoder. 

VicVLM performs close to MAMO in Image Retrieval

### Image Retrieval

| Algorithm                  | R@1       | R@5      | R@10      |
|----------------------------|-----------|----------|-----------|
| MAMO                       | **55.56** | 79.14    | **84.94** |
| VicVLM (pixel pred).       | 47.96     | 74.84    | 82.42     |
| VicVLM (BYOL latent pred.) | 55.5      | **79.2** | 84.92     |

### Text Retrieval

| Algorithm                  | R@1      | R@5      | R@10     |
|----------------------------|----------|----------|----------|
| MAMO                       | 67.8     | 88.1     | 94.3     |
| VicVLM (pixel pred).       | 63.4     | 85.6     | 90.0     |
| VicVLM (BYOL latent pred.) | **72.3** | **92.3** | **95.4** |

## How to run (pretraining training and benchmarks)
- Fetch code from repo
- Fetch Flickr30k dataset
- Run respective `pretrain_*` script
  
## Run retrieval and fetching
-  "https://drive.google.com/drive/u/0/folders/1t-_7XrazX1xXd1L7CWIlgygqKW6UnmEu"
-  Load model from gdrive or train your own
-  Run `retrieval_*.py` to get retrieval results
  
Note: we do not provide a frontend webapp for retrieval, you can use the code to get the results and use them in your own frontend.

# VicVLM implementation

Given an image, text pair (I, T), create masked image I', and masked text T'. Initialize a target network with weights equal to the online network's vision encoder. Freeze the target network's weights and update it as an exponentially weighted moving average of the online network's vision encoder. Generate a target representation $R_{i}^t$ of the image I from the target network.

1. Masked Image modeling: given image I', obtain representation $R_i^o$ from vision encoder of the online network. Take L2 loss between this and the masked positions of $R_{i}^t$.
2. Joint Masked Image Modeling: given image I', obtain representation $R_{i-joint}^o$ from multimodal encoder. Take L2 loss between this and the masked positions of $R_{i}^t$.
3. Masked Language Modeling: given text T', predict missing words from representation $R_t^o$ obtained from the text encoder of the online network.
4. VICReg:
    - Variance maximization: 
    $$
    \begin{align} v(Z) &= \frac{1}{d}\Sigma_{j = 1}^d max(0, 1 - S(z^j, \epsilon))\\
    S(x, \epsilon) &= \sqrt{Var(Z_j, \epsilon)}
    \end{align}
    $$
    where $d$ is the number of dimensions in the representation and $S(x, \epsilon)$ is the regularized standard deviation.
    - Covariance minimization:
    $$
    \begin{align}
    C(Z) = \frac{1}{d} \Sigma_{i \ne j} (Cov(Z_{i,j})^2)
    \end{align}
    $$
    - Invariance: given two sets of features F and F', minimize $\frac{1}{n}\Sigma_{i = 1}^n(F_i - F_i')^2$.

    We performed variance maximization and covariance minimization on features of (I, T), (I', T), (I, T') obtained from the multimodal encoder. Invariance was taken between (I, T) and (I', T), and (I, T) and (I, T').
4. Image-text contrastive learning: Minimize the InfoNCE loss between all possible image-text pairs.
5. Image text matching: given an image-text pair, determine the probability that they correspond to each other.
