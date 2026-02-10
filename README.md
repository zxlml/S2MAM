# Semi-supervised Meta Additive Model 

This repository is the official implementation for S2MAM based on Python language. 

## Requirements:

```python
Python 3.10.11
Torch 2.0.1
CUDA 117
PyYAML 6.0
Tensorboard 2.13.0
Torchvision 0.15.2
tqdm 4.65.0
```



## Training

​		In order to run the demo, please run the "main.py" file:

### 1 Run S2MAM for Regression Task

```python
# =============================================================================
#    Example for regression 
# =============================================================================
    demo_r()
```

### 2 Run S2MAM for Classification Task

```python
# =============================================================================
#     Example for classification
# =============================================================================
    demo_c()
```



## Results

#### Semi-supervised Regression & Classification

In this paper, 4 synthetic data and 12 real-world data (8 from UCI database and 1 from ADNI, 4 image datasets inlcuding COIL20, CelebA-HQ and AgeDB) are selected in our experiments for regression or classification. Indeed, these datasets have been widely used for validating additive models and semi-supervised learning models.

The empirical results can be found in the submitted manuscript.

#### Interpretability & Visualization

For simplicity, we present the first and second prediction components as well as their ground truth in above figure. The results show that our proposal can well learn the specific behavior of each input feature, which indeed provides strong interpretability to better understand the decision machinism of our model.
