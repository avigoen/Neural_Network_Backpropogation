import numpy as np

from keras.datasets import cifar10
from sklearn.decomposition import PCA



def get_raw():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)

def get_2D_normalised():
    (x_train, y_train), (x_test, y_test) = get_raw()
    
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
    
    # Calculate the mean and standard deviation of the pixel values over the training dataset
    mean = np.mean(x_train, axis=(0,1))
    std = np.std(x_train, axis=(0,1))
    
    # Normalize the pixel values by subtracting the mean and dividing by the standard deviation
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    
    return (x_train, y_train), (x_test, y_test)

def test_dimensionality_reduction(components):
    (x_train, y_train), (x_test, y_test) = get_2D_normalised()
    
    pca = PCA(n_components = components)
    # Fit the train data to the PCA and transform the train set data
    pca.fit_transform(x_train)
    
    # Print the percentage of variance explained by each component
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # Print the cumulative explained variance
    print("Cumulative explained variance ratio:", np.cumsum(pca.explained_variance_ratio_))

def get_dimensionlly_reduced(needed, components=100):
    (x_train, y_train), (x_test, y_test) = get_2D_normalised()
    
    pca = PCA(n_components = components)
    # Fit the train data to the PCA and transform the train set data
    X_train_pca = pca.fit_transform(x_train)
    # Apply the same PCA transformation to the test set data
    X_test_pca = pca.transform(x_test)
    
    return (X_train_pca[:, :needed], y_train), (X_test_pca[:, :needed], y_test)