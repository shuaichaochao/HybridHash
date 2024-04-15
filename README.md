# HybridHash: Hybrid Convolutional and Self-Attention Deep Hashing for Image Retrieval
## The Overall Architecture Of HybridHash
![brief_architecture](https://github.com/shuaichaochao/HybridHash/assets/49743419/372b311e-1e27-4fc7-bef4-ed1c5b953a17)
Figure.1. The detailed architecture of the proposed HybridHash. We adopt similar segmentation as ViT to divide the image with finer granularity and feed the generated image patches into the Transformer Block. The whole hybrid network consists of three stages to gradually decrease the resolution and increase the channel dimension. Interaction modules followed by each stage to promote the communication of information about the image blocks. Finally, the binary codes are output after the hash layer.
## Image Retrieval Related Work Instructions
