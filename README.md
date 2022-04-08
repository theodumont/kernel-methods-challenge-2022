# kernel-methods-challenge-2022

- **Team:** Caramel Methods
- **Team members:** Théophile Bousquet, Théo Dumont (MVA 2021-2022)

This repository contains the code of our models for the [2022 Kernel Methods for Machine Learning Data Challenge](https://www.kaggle.com/c/mva-mash-kernel-methods-2021-2022). The goal of this year's challenge was to obtain the best accuracy on an image classification task using solely kernel-based models.

We obtained an accuracy of 59.5% using a Kernel Ridge Regression (KRR) with a log kernel, trained on the Histograms of Oriented Gradients (HOG) of the input images, with a bit of data augmentation.

| Feature extraction | Data augm. | Model                   | Kernel     | Public score | Private score |
| ------------------ | ---------- | ----------------------- | ---------- |:------------:|:-------------:|
| HOG                | Yes        | Kernel Ridge Regression | Log kernel |   57.7%      |   **59.5%**   |

## Installation and usage
In order to reproduce our results, you can:
- install the repository and its dependencies, using:
    ```bash
    git clone https://github.com/theodumont/kernel-methods-challenge.git  # clone the repo
    cd kernel-methods-challenge
    pip install -r requirements.txt               # install dependencies
    conda install pytorch torchvision -c pytorch  # install torch (used for data augmentation)
    ```
- download the data from the [challenge webpage](https://www.kaggle.com/c/mva-mash-kernel-methods-2021-2022) and put it in the `data/` directory;
- run the default training (the one of our best scoring model) using
    ```bash
    python3 main.py
    ```
    This will create a file `Yte_pred.csv` in the `data/preds/` directory. The error score may be a little different from the one submitted on challenge due to the randomness of the data augmentation process.
- in order to have some insights on the command-line parameters of the `main.py` script, you can use
    ```bash
    python3 main.py ---help
    ```

## Repository structure
- `main.py`: main script for reproducing results
- `src/`: source code
    - `models.py`: code for classifiers (KRR, KSVC) and KPCA
    - `kernels.py`: implemented kernels
    - `hog.py`: code for HOG
    - `utils.py`: utility functions
- notebooks
    - `data_exploration.ipynb`: data exploration (dataset, data augmentation)
    - `hyperparameter_tuning.ipynb`: running hyperparameter tuning functions