# Import modules

import numpy as np
from scipy import misc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import imageio


def read_scene():
    data_x = imageio.imread('../../Data/umass_campus.jpg')

    return (data_x)


if __name__ == '__main__':
    ################################################
    # K-Means

    data_x = read_scene()
    data_temp = read_scene()
    print('X = ', data_x.shape)
    flattened_image = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
    flattened_temp = data_temp.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
    print('Flattened image = ', flattened_image.shape)

    print('Implement AHC here ...')

    print('Implement k-means here ...')
    for c in [2, 5, 10, 25, 50, 75, 100, 200]:
        kmeans = KMeans(n_clusters=c)
        cluster_lables = kmeans.fit_predict(flattened_image)
        for i in range(len(cluster_lables)):
            flattened_temp[i] = kmeans.cluster_centers_[cluster_lables[i]]

        reconstructed_image = flattened_temp.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
        plt.imshow(reconstructed_image)
        plt.show()
        print('Reconstructed image = ', reconstructed_image.shape)
