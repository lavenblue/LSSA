# Long- and Short-Term Self-Attention Network for Sequential Recommendation

This is our TensorFlow implementation for our paper and SASRec:

## Datasets

The preprocessed datasets are included in the data (`e.g. data/gowalla/gowalla_dataset.csv`), where each line contains an `user id` and 
an `session` meaning an items set (splitted by one day in time order).

For the handling of the file (`i.e. data/gowalla/gowalla_dataset.csv`), please refer to this paper and the code(https://github.com/uctoronto/SHAN).
Haochao Ying, Fuzhen Zhuang, Fuzheng Zhang, Yanchi Liu, Guandong Xu, Xing Xie, Hui Xiong, and Jian Wu. *[Sequential recommender system based on hierarchical attention networks]. In IJCAI, 2018.
