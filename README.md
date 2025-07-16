# DATransNet: Dynamic Attention Transformer Network for Infrared Small Target Detection

 Official implementation of paper "DATransNet: Dynamic Attention Transformer Network for Infrared Small Target Detection". 
 Our paper is accepted in [GRSL](https://ieeexplore.ieee.org/document/10947728).

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

Or you can download in [Baidu Cloud](https://pan.baidu.com/s/19DOSJZTHC0KO-wKyGRSldQ?pwd=mxhe) with code of "mxhe".

# Commands for Training

* **Run train.py to train our network**
  ```Run
  Python train.py
  ```

# Cited by

[《Adaptive Strategies for Multiscale Gradient Fusion in Neural Networks》](https://www.researchgate.net/profile/Xinyi-Zhang-235/publication/385103761_Adaptive_Strategies_for_Multiscale_Gradient_Fusion_in_Neural_Networks/links/6716a74209ba2d0c76174965/Adaptive-Strategies-for-Multiscale-Gradient-Fusion-in-Neural-Networks.pdf) indicates that our network is suitable for the tasks of visual light targets detection.

[《Hybrid attention and adaptive feature fusion network for infrared small target detection》](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://pdf.sciencedirectassets.com/271471/1-s2.0-S0143816625X00099/1-s2.0-S0143816625003999/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEYaCXVzLWVhc3QtMSJIMEYCIQCmXp5zD1mV1ORFSc5LvzsVLdHKhw4DFNAb1hzXU9NNigIhAMsxGsRjIriND1CqQexGBJyXtpU628YF5LdD0JGQNhe0KrIFCF4QBRoMMDU5MDAzNTQ2ODY1IgwBn2fDVELOWsJjEa8qjwWA2jYT3qRGSdsUizT93m2TEj%2BM6WtC4c%2FT4B97Cwe7XJlfHWV8m29jWEoo8qDq4cIXrDPk%2B0iqRT6ELVOL5hU7BeZYxZ%2Bb9RZN1YKa90VdzOuDMN08ij%2F3EEMDOewJNY%2FoDmGpMoDxWRHQOlTyNSb3YjYhtEUZTzJgr%2FLCJ241pCXwJUb4KrKxRquq6zUO%2F3hFV4YXHAy3nB641lHHODmVA1K%2F40t8In3oHej6B1bM%2BhSxFTpHuA2lAPFOoVpL08notqmPV8dO0vQPg3ya0jbzEG%2FDlxRX4TV4Q6EdXq06Klw%2FyVQKV%2BOeSdX6PsoNsExgEnnTHcEJOdDCcX%2BuCuaDEshZpQ5%2F8Mg7dff3ZurMkfSVzYz0tQEMLvTwVx1wz1nAf0CF695LIyd5aIEAcKx1EUk64E0q%2FEbi83N7zJ0Zudtsw1Mr%2FP0%2BWFck75vAozTieD%2BqgZ%2BIDvJo5KoPOLw9daHsTW1Jpn36q350s7tdDS2kS9JP9x2D%2BPcnF9CEsfowfKt7RWws3GiJQ1msvfMxN%2FZWFd9WXvoiTp3GhgzfdnUJardrtjRowl4RdW1wDSryoMflwNO4GlYYeYINGZdTVgjhycZhn5AGd2ke2M9%2FwBmeapVJlVO2Np7%2BhBrNzTtaypSzjoL2YV6dhpqwAZheYGXTT8BAOIrUzogRB5iJnEiu9PhyUG0QOjZtxJ35ynBuBLf35sj7r8cmLC3Z3Yts0BGgeD8CZ57pQCY2ZBdLF%2FyxgQvnyHjM8qWPCXv30qXun1%2B4xE8guDRa82SyLlxb2bzarKT%2BHw%2FJNR%2BvfaEK4rvykEN0UOvDhZYtWJfsyKK5pdTzSNl6Dknqhnc28gNsJnqIWERFAPgUIz8g5XUNMPbL3sMGOrABkmbEXS3MLoCz563e%2BBYIRuN0COaLuYVeV4TNxalLug5L7%2BHPJfGBaFx506UrLw1pzmx5MSkG1cKArrugd%2BmXDcmYBSKcwouKftLvSanYb0uuXollih%2FyfjS6sv69Gbw8jHquAiPxrNkQeESK1429BZtHXSPr2JD45cqusce1N76e1fBcCVfQFxDVrW64eDnbqnxgZqjOAVJMnWjrgcpDKzQsjFZB3%2FgW0QAqi8GtVK8%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250716T134553Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY26XCG6PX%2F20250716%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=c91a1a6bd6eb12080c70078964facbd89d7c468605fcd1020fbf101ff66ae109&hash=e6e08097e5ea5594682b9d2bb87969307df267c94b2b83413eca06d05e7882cb&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0143816625003999&tid=spdf-b9e9a882-191a-4fcf-9e72-c594a6b08c9d&sid=c34621125a84504cc62ba9c1db3f3c1c18a8gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=120d5f5155550454550500&rr=9601f14dfa300723&cc=hk&kca=eyJrZXkiOiJLTG5RV2VwMFdjYkRESjE3cG5GZGU2Nk12L2dBZ3JUbXdmYTAxVUV6U0VqNHJiTVEwNGJoWWt4OUxKMzd6NTFvczN2NlBHcDN3T2JCSkt0U3hHUFNORjRNb3dUWTVWQU1wNnhVMnk3V01nclpCK1JzQU5vditKbU1zL1A2WUFod0JwVURTQVcvb21PRW5FclhyN2d5ZjZFS09aZGhsRU1QTG91MjFzNkFjUU51b3JNWVF3PT0iLCJpdiI6ImJmZjBhMWFiYjgwZWI4MGQ2N2Y3ZjI1OTUzMjNlOTVkIn0=_1752673568585) shows a good balance in computation cost and precision among other models
![]()
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
# Chinese Introduction
The chinese introduction is accessiable at [https://blog.csdn.net/weixin_45358930/article/details/147562104?spm=1001.2014.3001.5501](https://blog.csdn.net/weixin_45358930/article/details/147562104?spm=1001.2014.3001.5501).
# Weights

We could offer the weights for IRSTD-1K [Weight_for_IRSTD_1K](best_ckpt_for_IRSTD_1K.pth.tar) and NUDT-SIRST [weight_for_NUDT_SIRST](best_ckpt_fot_NUDT_IRSTD.pth.tar).
