# FaceGuard: Deepfake Detection using XceptionNet and GradCAM
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

An explainable deepfake detection system using a fine-tuned **XceptionNet convolutional neural network** with **Grad-CAM visualization** and full **quantitative evaluation metrics**.

---

# Project Overview

Deepfakes are AI-generated manipulated videos or images that can spread misinformation and threaten digital trust. This project develops a **deep learning based deepfake detection system** that classifies facial images as **Real** or **Fake** while also explaining model decisions using **Grad-CAM heatmaps**.

The system uses **transfer learning with XceptionNet**, trained on extracted face frames from manipulated and authentic videos.

---

# Model Architecture

The project uses **XceptionNet (Extreme Inception)** as the backbone CNN.

### Architecture Pipeline

