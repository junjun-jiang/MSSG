Paper: https://ieeexplore.ieee.org/document/9237144/ (IEEE Xplore) 

### Multilayer Spectral-Spatial Graphs for Label Noisy Robust Hyperspectral Image Classification

In hyperspectral image (HSI) analysis, label information is a scarce resource and it is unavoidably affected by human and nonhuman factors, resulting in a large amount of label noise. Although most of the recent supervised HSI classification methods have achieved good classification results, their performance drastically decreases when the training samples contain label noise. To address this issue, we propose a label noise cleansing method based on spectral-spatial graphs (SSGs). In particular, an affinity graph is constructed based on spectral and spatial similarity, in which pixels in a superpixel segmentation-based homogeneous region are connected, and their similarities are measured by spectral feature vectors. Then, we use the constructed affinity graph to regularize the process of label noise cleansing. In this manner, we transform label noise cleansing to an optimization problem with a graph constraint. To fully utilize spatial information, we further develop multiscale segmentation-based multilayer SSGs (MSSGs). It can efficiently merge the complementary information of multilayer graphs and thus provides richer spatial information compared with any single-layer graph obtained from isolation segmentation. Experimental results show that MSSG reduces the level of label noise. Compared with the state of the art, the proposed MSSG method exhibits significantly enhanced classification accuracy toward the training data with noisy labels. The significant advantages of the proposed method over four major classifiers are also demonstrated. 

### If you find our work useful in your research or publication, please cite our work:
```
@article{jiang2020learning,
 author={J. {Jiang} and J. {Ma} and X. {Liu}}, 
 journal={IEEE Transactions on Neural Networks and Learning Systems}, 
 title={Multilayer Spectral-Spatial Graphs for Label Noisy Robust Hyperspectral Image Classification}, 
 year={2020}, 
 volume={}, 
 number={}, 
 DOI={10.1109/TNNLS.2020.3029523},}
```

```
@article{jiang2019hyperspectral,
  title={Hyperspectral Image Classification in the Presence of Noisy Labels},
  author={Jiang, Junjun and Ma, Jiayi and Wang, Zheng and Chen, Chen and Liu, Xianming},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={57},
  number={2},
  pages={851--865},
  year={2019}
}
