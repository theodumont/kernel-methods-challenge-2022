import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# FILES ====================================

def get_data(data_path="./data"):
    """Return data files."""
    Xtr = np.array(pd.read_csv(os.path.join(data_path, 'Xtr.csv'), header=None, sep=',', usecols=range(3072)))
    Xte = np.array(pd.read_csv(os.path.join(data_path, 'Xte.csv'), header=None, sep=',', usecols=range(3072)))
    Ytr = np.array(pd.read_csv(os.path.join(data_path, 'Ytr.csv'),sep=',',usecols=[1])).squeeze().astype(np.int32)
    return Xtr, Xte, Ytr

def save_Yte(Yte, model_name):
    """Save Yte predictions and append 'model_name' to file name."""
    dataframe = pd.DataFrame({'Prediction' : Yte})
    dataframe.index += 1
    dataframe.to_csv(f'./data/Yte_pred_{model_name}.csv',index_label='Id')


# DISPLAY ====================================

def show_images(tensor, Ytr=None, nb_img=6):
    """Display nb_img images with their class."""
    plt.figure(figsize=(20, 20*nb_img), tight_layout=True)
    for i in range(nb_img):
        plt.subplot(1,nb_img,i+1)
        img = tensor[i]
        plt.imshow((img - img.min()) / (img.max() - img.min()))
        if Ytr is not None:
            plt.title(f"Image of class {Ytr[i]}")
    plt.show()


# RANDOM ====================================

def array_to_tensor(array):
    """Transform the 5000 x 3072 dataset into 5000 x 32 x 32 x 3."""
    N = array.shape[0]
    nb_pixels = array.shape[1] // 3
    tensor = []
    # separate each channel and reshape
    for i in range(3):
        tensor.append(array[:,i*nb_pixels:(i+1)*nb_pixels].reshape((N, 32, 32)))
    # stack
    tensor = np.stack(tensor, axis=3)
    return tensor

def accuracy(gt, pr):
    """Return proportion of well predicted samples."""
    return (gt == pr).mean()