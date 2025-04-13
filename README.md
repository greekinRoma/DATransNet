# DATransNet: Dynamic Attention Transformer Network for Infrared Small Target Detection

 Official implementation of paper "DATransNet: Dynamic Attention Transformer Network for Infrared Small Target Detection".

# Network Structure

![Backbone](backbone.png)
![GradFormer](fig_0.png)
![Global Feature Extraction Module](GFEM.png)

# Requirements

* **Python 3.8**
* **Windows10, Ubuntu18.04 or higher**
* **NVDIA GeForce RTX 4080**
* **Pytorch 1.13.0**
* **More details from requirements.txt**

# Dataset

We used NUDT-SIRST and IRSTD-1K for training. The two datasets could be found and downloaded in: [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection) and [IRSTD-1K](https://github.com/RuiZhang97/ISNet).

Or you can download in [Baidu Cloud](https://pan.baidu.com/s/19DOSJZTHC0KO-wKyGRSldQ?pwd=mxhe) with code of "mxhe".

# Commands for Training

* **Run train.py to train our network**
  ```Run
  Python train.py
  ```

# Cited by

[《Adaptive Strategies for Multiscale Gradient Fusion in Neural Networks》](https://www.researchgate.net/profile/Xinyi-Zhang-235/publication/385103761_Adaptive_Strategies_for_Multiscale_Gradient_Fusion_in_Neural_Networks/links/6716a74209ba2d0c76174965/Adaptive-Strategies-for-Multiscale-Gradient-Fusion-in-Neural-Networks.pdf) indicates that our network is suitable for the tasks of visual light targets detection.

# Citation

```Citation
@ARTICLE{10947728,
  author={Hu, Chen and Huang, Yian and Li, Kexuan and Zhang, Luping and Long, Chang and Zhu, Yiming and Pu, Tian and Peng, Zhenming},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={DATransNet: Dynamic Attention Transformer Network for Infrared Small Target Detection}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Transformers;Data mining;Training;Object detection;Image edge detection;Head;Measurement;Geoscience and remote sensing;Artificial intelligence;Infrared small target detection (ISTD);convolution neural network (CNN);Dynamic Attention Transformer;global feature extraction},
  doi={10.1109/LGRS.2025.3557021}}
```

# Weights

We could offer the weights for IRSTD-1K [Weight_for_IRSTD_1K](best_ckpt_for_IRSTD_1K.pth.tar) and NUDT-SIRST
