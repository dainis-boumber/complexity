import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# Import data and preprocess 
mnist = pd.read_csv('./data/mnist/train_src_100.csv') # Using 100 samples only for this test run
labels = mnist.as_matrix(columns=['label'])
dataset = mnist.drop('label', axis = 1).as_matrix()
dataset[dataset > 0] = 1 # Convert each pixel either 0 for white and 1 for black for better classification


def load_mnist():
    
    rows = 100
    columns = 784
    index = 1
    X = []
    for image in dataset[:rows*columns]:
        img = np.reshape(image, [28, 28])
        X.append(img)
        index += 1
    X = np.array(X).reshape(rows, -1)
    mnist = pd.DataFrame(X)
    mnist = mnist.as_matrix()
    y = labels.flatten()
    
    print("Completed with X shape: ", mnist.shape)
    print("Flattened y shape: ", y.shape)
    
    mnist, y = shuffle(X, y, random_state = 5)
    return mnist, y
  
  
def load_mnist_rotated():
    
    rows = 100
    columns = 784
    indx = 1
    X = []
    for image in dataset[:rows*columns]:
        img = np.reshape(image, [28, 28])
        rotated = ndimage.rotate(img, 90) # Rotate the images by 90 degrees
        X.append(rotated)
        indx += 1
    X = np.array(X).reshape(rows, -1)
    
    mnist_rotated = pd.DataFrame(X)
    # mnist_rotated.to_csv('./data/mnist_rotated/minst_rotated_21000.csv', index=False, header=False)
    mnist_rotated = mnist_rotated.as_matrix()
    
    y = labels.flatten()
    print("Completed with X shape: ", mnist_rotated.shape)
    print("Flattened y shape: ", y.shape)
    
    mnist_rotated, y = shuffle(X, y, random_state = 15)
    return mnist_rotated, y