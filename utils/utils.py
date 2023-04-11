import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.datasets as dset

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()


def class_label_to_img_voc(label_array):
    h, w = label_array.shape
    img_array = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Define the color map for VOC classes
    voc_color_map = {
        0: (0, 0, 0),  # Background
        1: (128, 0, 0),
        2: (0, 128, 0),
        3: (128, 128, 0),
        4: (0, 0, 128),
        5: (128, 0, 128),
        6: (0, 128, 128),
        7: (128, 128, 128),
        8: (64, 0, 0),
        9: (192, 0, 0),
        10: (64, 128, 0),
        11: (192, 128, 0),
        12: (64, 0, 128),
        13: (192, 0, 128),
        14: (64, 128, 128),
        15: (192, 128, 128),
        16: (0, 64, 0),
        17: (128, 64, 0),
        18: (0, 192, 0),
        19: (128, 192, 0),
        20: (0, 64, 128),
    }

    for label, color in voc_color_map.items():
        img_array[label_array == label] = color

    return Image.fromarray(img_array)