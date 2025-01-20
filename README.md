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
@misc{hu2025datransnetdynamicattentiontransformer,
      title={DATransNet: Dynamic Attention Transformer Network for Infrared Small Target Detection}, 
      author={Chen Hu and Yian Huang and Kexuan Li and Luping Zhang and Chang Long and Yiming Zhu and Tian Pu and Zhenming Peng},
      year={2025},
      eprint={2409.19599},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.19599}, 
}
```
