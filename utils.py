import numpy as np
import matplotlib.pyplot as plt

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




def show_images(tensor, nb_img=6):
    plt.figure(figsize=(20, 20*nb_img), tight_layout=True)
    for i in range(nb_img):
        plt.subplot(1,nb_img,i+1)
        img = tensor[i]
        plt.imshow((img - img.min()) / (img.max() - img.min()))
    plt.show()
