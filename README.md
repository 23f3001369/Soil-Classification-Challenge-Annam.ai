# Soil-Classification-Challenge-Annam.ai

### Soil Type Classification Model

This project implements a deep learning model to classify soil images into specific soil types, helping in soil analysis and agricultural applications.

## Project Overview

- **Goal:** Classify soil images into one of the following soil types:
  - Alluvial
  - Black
  - Clay
  - Red
- **Model Backbone:** ResNet50 (with optional experiments using EfficientNet-B3a)
- **Framework:** PyTorch
- **Dataset:** Annotated soil image dataset categorized by soil types.

## Features

- Data preprocessing and augmentation pipelines
- Training and validation scripts with checkpoint saving
- Evaluation using metrics like accuracy, precision, recall, and F1-score
- Test Time Augmentation (TTA) support for robust inference
- Easily configurable hyperparameters via YAML config files

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/soil-type-classification.git
    cd soil-type-classification
    ```

2. Set up your Python environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    venv\Scripts\activate     # Windows
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

Train the model on your dataset by running preprocessing.py and postprocessing.py (present in src folder).
