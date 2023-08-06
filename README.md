# MHADiv
MultiHeadAttention Layer: 


Our goal is to tackle the problem of having an input size that represents the number of features which could be not divisible by the number of heads. Usually the multihead attention layer requires the input size to be divisible by the number of heads. I added an embedding dimension that is very close to the input size and divisible by a specific number of head. I also modified the MultiheadAttentionDiv linear layer's output's dimensions. 


## Colab Code

- A Colab Notebook PyTorch Example is linked [here](https://colab.research.google.com/drive/1hVluyhBqicHS3I9lfqwhe57IP92zrurj?usp=sharing).

## References

- The code is adapted from the following [link](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html#Set-Anomaly-Detection). 

