# DeepIK-Source: DeepIK: an intelligent system to diagnose infectious keratitis using slit lamp photographs
# create time: 2023.10.20

# Introduction£º
This repository contains the source code for creating a unique deep learning system named DeepIK to imitate a corneal specialist for discerning a variety of corneal infections based on slit lamp photographs.
DeepIK can aid eye doctors to discern corneal infections resulting from fungi, bacteria, amoebas, and viruses early, improving the visual outcomes of patients by offering timely and precise medical interventions.

# Prerequisites
* Ubuntu: 18.04 lts
* Python 3.7.8
* Pytorch 1.6.0
* NVIDIA GPU + CUDA_10.1 CuDNN_7.5

This repository has been tested on 4 NVIDIA RTX2080Ti. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

# Installation
Other packages are as follows:
* pytorch: 1.6.0 
* wheel: 0.34.2
* timm:  0.4.12
* tourchvision: 0.7.0
* scipy:  1.5.2
* joblib: 0.16.0
* opencv-python: 4.3.0.38
* scikit-image: 0.19.2
* scikit-learn: 0.23.2
* matplotlib: 3.3.1
* efficientnet-pytorch: 0.7.1
* ipython: 7.30.1
* pandas: 1.2.3
* protobuf: 3.13.0
* h5py: 3.5.0
* numpy: 1.19.1


# Install dependencies
pip install -r requirements.txt

# Usage
* The file "DeepIK_training.py" in /DeepIK-Source is used for the DeepIK model training.
* The file "DeepIK_testing.py" in /DeepIK-Source is used for the DeepIK model testing.
* The file "Comparable_CNNs_training.py" in /DeepIK-Source is used for comparable CNNs training.
* The file "Comparable_CNNs_testing.py" in /DeepIK-Source is used for comparable CNNs testing.

## Train DeepIK model on GPU
python DeepIK_training.py -a 'densenet121'

## Train DenseNet121 on GPU
python Comparable_CNNs_training.py -a 'densenet121'

## Train InceptionResNetV2 on GPU
python Comparable_CNNs_training.py -a 'inceptionresnetv2'

## Train Transform_base on GPU
python Comparable_CNNs_training.py -a 'Transform_base'

## Evaluate DeepIK model on GPU
python DeepIK_testing.py
***

## Evaluate three comparable models of InceptionResNet-V2, DenseNet121, and Transform_base at the same time on GPU
python Comparable_CNNs_testing.py
***

The expected output: print the classification probabilities for keratitis resulting from fungi, bacteria, amoebas, viruses and other non-infectious factors.


* Please feel free to contact us for any questions or comments: Zhongwen Li, E-mail: li.zhw@qq.com or Jiewei Jiang, E-mail: jiangjw924@126.com.