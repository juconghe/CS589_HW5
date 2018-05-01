# Import modules

import numpy as np
from scipy import misc
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import imageio

def read_scene():
    data_x = imageio.imread('../../Data/umass_campus_100x100x3.jpg')

    return (data_x)

def compute_mean(img, labels):
    temp_dict = {}
    for i in range(len(labels)):
        if labels[i] in temp_dict:
            temp_dict[labels[i]] = np.append(temp_dict[labels[i]], [img[i]], axis=0)
        else:
            temp_dict[labels[i]] = np.array([img[i]])
    mean_dict = {}
    for k, v in temp_dict.items():
        mean_dict[k] = [v[:,0].mean(),v[:,1].mean(),v[:,2].mean()]

    return mean_dict

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
    for a in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine' ]:
        for l in ['ward', 'complete', 'average']:
            if l == 'ward' and a != 'euclidean':
                pass
            else:
                print(a, l)
                hac = AgglomerativeClustering(affinity=a, linkage=l)
                cluster_lables = hac.fit_predict(flattened_image)
                mean_dict = compute_mean(flattened_image, cluster_lables)
                for i in range(len(cluster_lables)):
                    flattened_temp[i] = mean_dict[cluster_lables[i]]
                reconstructed_image = flattened_temp.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
                plt.imshow(reconstructed_image)
                plt.show()
    # print('Implement k-means here ...')
    # for c in [2, 5, 10, 25, 50, 75, 100, 200]:
    #     kmeans = KMeans(n_clusters=c)
    #     cluster_lables = kmeans.fit_predict(flattened_image)
    #     for i in range(len(cluster_lables)):
    #         flattened_temp[i] = kmeans.cluster_centers_[cluster_lables[i]]
    #
    #     reconstructed_image = flattened_temp.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
    #     plt.imshow(reconstructed_image)
    #     plt.savefig('c{}.jpg'.format(c))
    #     plt.close()
    #     print('Reconstructed image = ', reconstructed_image.shape)
