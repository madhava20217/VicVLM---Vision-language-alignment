

## How to run (pretraining training and benchmarks)
- Fetch code from repo
- Fetch Flickr30k dataset
- Run respective `pretrain_*` script
  
## Run retrieval and fetching
-  "https://drive.google.com/drive/u/0/folders/1t-_7XrazX1xXd1L7CWIlgygqKW6UnmEu"
-  Load model from gdrive or train your own
-  Run `retrieval_*.py` to get retrieval results
  
Note: we do not provide a frontend webapp for retrieval, you can use the code to get the results and use them in your own frontend.

# Summary of our impl

init: I, T
get:
- I', T : image patches masked
- I, T' : text patches masked

multimodal representation f(theta)(I', T), f(theta)(I, T')
network: theta

1. masked representation modeling
- target network: theta_avg
- theta_avg = alpha*theta_avg + (1-alpha)theta [Exponentially weighted moving average]
- no gradient propagation

- send (I, T) into target network, get latent multimodal representation f(theta_avg)(I, T)
- f(theta_avg)(I, T) serves as prediction targets for for f(theta)(I', T) after passing through a nonlinear projector g.
- Minimiise L2 loss on each masked position.


Note: MRM is effective empirically, self distillation may collapse. Introduce two explicit prediction targets for vision and language.

2. Masked image modeling:
- uses masked image with the help of text
- uses momentum visual features extracted by target network's image encoder : f(theta_avg)_visual
- calculates l1 loss between MLP predictor on joint multimodal representation of trainee network after passing thru an MLP and target network visual representations.

3. Masked language modeling:
- uses masked text only, no vision since semantically rich
- use word tokens as the explicit target.
- uses crossentropy loss between word prediction probability and ground truth on each masked text position.

4. global text image alignment (Image-text contrastive learning)
- info nce

5. Image text matching
- similarity probability in ITC to sample an in-batch hard negative example for each image and each text. Then use CLS token from multimodal fusion encoder's output to predict whether image-text pair is matched.
- determine whether image text pair is matched using 

