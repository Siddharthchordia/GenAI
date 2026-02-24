# Lab 2: Generative Adversarial Networks (GANs)

## Course
**Introduction to Generative AI**

## Objective
The objective of this lab is to understand and implement **Generative Adversarial Networks (GANs)** using PyTorch.  
The lab focuses on training a GAN on **MNIST** and **Fashion-MNIST** datasets, generating realistic images, monitoring training stability, and performing a qualitative comparison using noisy and generated images.

---

## Overview of Experiments

This lab consists of two main experiments:

1. **GAN Training**
   - Train a Generator and Discriminator on MNIST and Fashion-MNIST datasets
   - Generate synthetic images from random noise
   - Monitor training using loss values and layer-wise weight and gradient norms
   - Save generated images and trained models

2. **Image Generation & Denoising Visualization**
   - Add artificial noise to real images
   - Generate clean-looking images using a pretrained GAN Generator
   - Compare clean, noisy, and generated images visually

> **Note:** The denoising experiment demonstrates GAN-based image generation and visual comparison. It is not a true denoising autoencoder.

---

## Folder Structure
Lab_2/
├── GAN_training.ipynb
├── generator_mnist.pth
├── discriminator_mnist.pth
├── generated_samples_mnist/
│   └── epoch-wise generated images
├── final_generated_images_mnist/
│   └── 100 generated images
├── denoising_results/
│   ├── clean_images.png
│   ├── noisy_images.png
│   └── denoised_images.png
└── README.md

---

## Technologies Used

- Python 3
- PyTorch
- TorchVision
- NumPy
- Google Colab / Jupyter Notebook

---

## Model Architecture

### Generator
- Fully connected neural network
- Input: Random noise vector (100-dimensional)
- Output: 28×28 grayscale image
- Activation: LeakyReLU and Tanh

### Discriminator
- Fully connected neural network
- Input: Flattened 28×28 image
- Output: Probability of image being real
- Activation: LeakyReLU and Sigmoid

---

## Training Details

- Loss Function: Binary Cross Entropy Loss (BCELoss)
- Optimizer: Adam
- Learning Rate: 0.0002
- Batch Size: 64
- Epochs: 50
- Device: CPU / GPU (CUDA if available)

During training, the following metrics are printed for every epoch:
- Discriminator loss
- Generator loss
- Discriminator accuracy
- Generator layer-wise weight norms
- Generator layer-wise gradient norms

---

## Results

- Successfully trained GAN models on MNIST and Fashion-MNIST datasets
- Generated visually meaningful digit and clothing images
- Saved at least 100 generated images for evaluation
- Observed stable training through gradient and weight norm monitoring

---

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone <repository-url>

2.	Open the notebook:
    GAN_training.ipynb

3.	Set the dataset choice (mnist or fashion) in the configuration cell.

4.	Run all cells sequentially.

## Conclusion

This lab provided hands-on experience with GANs and helped in understanding adversarial training dynamics, image generation, and training stability monitoring.
The experiment demonstrates the effectiveness of GANs in learning data distributions and generating realistic samples.