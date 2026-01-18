# AI Cat Species Image Generation & Classification

This project demonstrates a complete end-to-end workflow for synthetic data generation and deep learning model training using Google Colab. The project focuses on generating synthetic images of cat species and then using Transfer Learning with ResNet18 to classify them.

## Project Structure

*   `image_generation.ipynb`: Notebook for generating the dataset using the `stabilityai/sd-turbo` model via the Hugging Face `diffusers` library. It generates images for 40 cat species classes.
*   `resnet_training.ipynb`: Utilizes **Transfer Learning** with a pre-trained **ResNet18** model. It fine-tunes the model for classification and saves the trained model to Google Drive.

## Google Colab Configuration

This project is optimized for the Google Colab environment. Follow these steps to set up and run the project effectively.

### 1. Enable GPU Runtime

All notebooks require a GPU for efficient execution.

1.  Open the notebook in Google Colab.
2.  Go to **Runtime** -> **Change runtime type**.
3.  Under **Hardware accelerator**, select **T4 GPU** (or better if available).
4.  Click **Save**.

### 2. Google Drive Mounting (Crucial for ResNet Training)

The `resnet_training.ipynb` notebook relies on Google Drive for loading the dataset and saving the model to ensure persistence.
*   It looks for data in: `/content/drive/MyDrive/Colab Notebooks/GenAI/dataset`
*   It saves models to: `/content/drive/MyDrive/Colab Notebooks/GenAI/models`

Ensure you mount your drive using the code provided in the notebooks or the sidebar method.

## Execution Workflow

### Step 1: Data Generation
**Run `image_generation.ipynb`**
*   This uses Stable Diffusion to generate the dataset.
*   **Important:** Ensure the output directory of the generated images matches the path expected by the training notebook (or move the generated files to your Drive folder: `Colab Notebooks/GenAI/dataset`).

### Step 2: Model Training
**Run `resnet_training.ipynb`**
*   **Architecture:** ResNet18 (Pre-trained on ImageNet).
*   **Training:** Fine-tunes the last fully connected layer for 5 epochs.
*   **Usage:** Leveraging pre-learned features allows for high accuracy (approx. 82%) with minimal training time.
*   **Output:** Saves the trained model as `resnet18_model.pth` in your Google Drive.

## Technical Details

| Feature | ResNet18 Transfer Learning |
| :--- | :--- |
| **Model** | ResNet18 (Pre-trained) |
| **Training Strategy** | Fine-tuning |
| **Epochs** | 5 |
| **Input Size** | 224x224 |
| **Normalization** | ImageNet Stats |
| **Optimizer** | Adam (lr=1e-4) |
| **Accuracy (approx)** | ~82% |

## Dependencies

Common dependencies required across notebooks:
```python
!pip install -q diffusers transformers accelerate torch torchvision pillow tqdm scikit-learn
```
(Note: `resnet_training.ipynb` relies primarily on standard `torch` and `torchvision` libraries available in Colab by default).
