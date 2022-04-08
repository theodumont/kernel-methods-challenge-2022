# kernel-methods-challenge

## Results

| Feature extraction       | Data augmentation | Model                       | Kernel                | Private score |
| ------------------------ | ----------------- | --------------------------- | --------------------- |:------------:|
| HOG                      | Yes               | OneVsRestKRR (`lambd=1e-4`) | Log kernel (`p=2`)    |   todo       |

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
- in order to have some insights on the command-line parameters of the `main.py` script, you can use
    ```bash
    python3 main.py ---help
    ```
