# HybridHash: Hybrid Convolutional and Self-Attention Deep Hashing for Image Retrieval (ICMR 2024)

## The Overall Architecture Of HybridHash
![brief_architecture](https://github.com/shuaichaochao/HybridHash/assets/49743419/372b311e-1e27-4fc7-bef4-ed1c5b953a17)
Figure.1. The detailed architecture of the proposed HybridHash. We adopt similar segmentation as ViT to divide the image with finer granularity and feed the generated image patches into the Transformer Block. The whole hybrid network consists of three stages to gradually decrease the resolution and increase the channel dimension. Interaction modules followed by each stage to promote the communication of information about the image blocks. Finally, the binary codes are output after the hash layer.

![image](https://github.com/shuaichaochao/HybridHash/assets/49743419/bda9a6cb-892c-406b-b9dc-467e35be5ebc)


## Image Retrieval Related Work Instructions

There are three main types of prevailing image retrieval hashing methods, which are symmetric Deep Supervised Hashing, Asymmetric Deep Supervised Hashing and Center Similarity Deep Supervised Hashing. They are different. 
### Symmetric Deep Supervised Hashing in Image Retrieval
Due to the rapid development of deep learning, deep supervised hashing methods have achieved superior performance on image retrieval tasks. The majority of these methods are symmetric deep supervised hashing methods such as CNNH[1], DSH[2], DNNH[3], HashNet[4], DCH[5], etc., which exploit the similarity or dissimilarity between pairs of images to learn one deep hash function for query images and database (retrieval) images. Symmetric deep supervised hashing methods utilize only fewer training images and supervised information to obtain superior performance, which have been preferred by many researchers.
### Asymmetric Deep Supervised Hashing in Image Retrieval
Symmetric deep supervised hashing methods are broadly used in image retrieval tasks and have achieved well performance. Nevertheless, the training of these hashing methods is typically time-consuming, which makes them hard to
effectively utilize the supervised information for cases with large-scale database. Asymmetric Deep Supervised Hashing (ADSH)[6] is the first asymmetric strategy method for large-scale nearest neighbor search that treats query images and database (retrieval) images in an asymmetric way. More specifically, ADSH learns a deep hash function only for query points, while the hash codes for database points are directly learned. On this basis, many asymmetric hashing methods have been proposed such as ADMH[7], JLDSH[8]. Each of these methods imposes some extent of improvement on the network architecture or loss function. Compared with traditional symmetric deep supervised hashing methods, asymmetric deep supervised hashing method can utilize the supervised information of database (retrieval) images to obtain superior performance which has better training efficiency. However, it is impractical for tens of millions of images to all have label information.
### Central Similarity Deep Supervised Hashing in Image Retrieval
All of the above deep supervised hashing methods learn hash functions by capturing pairwise or triplet data similarities from a local perspective, which may   harm the discriminability of the generated hash codes. CSQ[9] proposes a new global similarity metric, called central similarity. Here, hash values of similar data points are encouraged to approach a common center, whereas pairs of dissimilar hash codes converge to distinct centers. Immediately after that, MDSHC [10] proposes an optimization method that finds hash centers with a constraint on the minimum distance between any pair of hash centers. The central similarity deep supervised hashing methods can capture the global data distribution, which tackle the problem of data imbalance and generate high-quality hash functions efficiently. 

Therefore the above three types of methods are not generally compared at the same time since it is an unfair comparison. Our proposed HybridHash belongs to the first type of hashing methods mentioned above and hence also compared with only the state-of-the-art hashing methods in the first type.

## Datasets

The following references are also derived from a [swuxyj](https://github.com/swuxyj/DeepHash-pytorch)

There are three different configurations for cifar10

   * config["dataset"]="cifar10" will use 1000 images (100 images per class) as the query set, 5000 images( 500 images per class) as training set , the remaining 54,000 images are used as database.
    
   * config["dataset"]="cifar10-1" will use 1000 images (100 images per class) as the query set, the remaining 59,000 images are used as database, 5000 images( 500 images per class) are randomly sampled from the database as training set.
    
   * config["dataset"]="cifar10-2" will use 10000 images (1000 images per class) as the query set, 50000 images( 5000 images per class) as training set and database.

You can download NUS-WIDE [here](https://github.com/swuxyj/DeepHash-pytorch)

Use data/nus-wide/code.py to randomly select 100 images per class as the query set (2,100 images in total). The remaining images are used as the database set, from which we randomly sample 500 images per class as the training set (10, 500 images in total).

You can download ImageNet, NUS-WIDE-m and COCO dataset [here](https://github.com/swuxyj/DeepHash-pytorch) where is the data split copy from

NUS-WIDE-m is different from NUS-WIDE, so i made a distinction.

269,648 images in NUS-WIDE , and 195834 images which are associated with 21 most frequent concepts.

NUS-WIDE-m has 223,496 images which are associated with 81 concepts, and NUS-WIDE-m is used in HashNet(ICCV2017). Of these, removing the incorrectly labeled images, there are 173692 images, including 5000 images for the test set and the rest for the retrieval set.

## Reference
[1] Xia, R.K., Pan, Y., Lai, H.J., Liu, C., Yan, S.C.: Supervised Hashing for Image Retrieval via Image Representation Learning. In: Proceedings of the Twenty-Eighth AAAI Conference on Artificial Intelligence (AAAI), pp. 2156-2162. AAAI, (2014)

[2] Liu, H., Wang, R., Shan, S., Chen. X.: Deep supervised hashing for fast image retrieval. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2064–2072. IEEE, (2016)

[3] Lai, H., Pan, Y., Liu, Y., Yan, S.: Simultaneous feature learning and hash coding with deep neural networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3270–3278. IEEE, (2015)

[4] Cao, Z.J., Long, M.S., Wang, J. M., Yu, P.S.: HashNet: Deep Learning to Hash by Continuation. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 5609-5618. IEEE, (2017)

[5] Cao, Y., Long, M.S., Liu, B., Wang, J.M.: Deep Cauchy Hashing for Hamming Space Retrieval. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1229-1237. IEEE, (2018)

[6] Jiang, Q.Y., Li, W.J.: Asymmetric Deep Supervised Hashing. In: Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence (AAAI), pp. 3342-3349. AAAI, (2018)

[7] Ma, L, Li, H.L., Meng, F.M., Wu, Q.B., Ngan, K.N.: Discriminative deep metric learning for asymmetric discrete hashing. Neurocomputing 380, 115-124 (2020)

[8] Gu, G.H., Liu, J.T., Li, Z.Y., Huo, W.H., Zhao, Y.:Joint learning based deep supervised hashing for large-scale image retrieval. Neurocomputing 385, 348-357 (2020)

[9] Yuan, L., Wang, T., Zhang, X.P., Tay, E.H., Jie, Z.Q., Liu, W., Feng, J.S.: Central Similarity Quantization for Efficient Image and Video Retrieval. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3080-3089. IEEE, (2020)

[10] Liangdao Wang, Yan Pan, Cong Liu, Hanjiang Lai, Jian Yin, and Ye Liu, “Deep hashing with minimal-distance-separated hash centers,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 23455–
23464.
