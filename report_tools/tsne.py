import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

raw_targetLR_DR_path = '/mnt/HDD1/phudh/super_resolution/RDSR-ReferenceDegradationSR/RDSR/dr_result/0_list_targetLR_dr.pickle'

with open(raw_targetLR_DR_path, 'rb') as file:
    targetLR_dr = pickle.load(file)
    targetLR_dr = np.array(targetLR_dr)

path2 = '/mnt/HDD1/phudh/super_resolution/RDSR-ReferenceDegradationSR/RDSR/dr_result/0_list_refLR_dr.pickle'

with open(path2, 'rb') as file:
    refLR_dr_0 = pickle.load(file)
    refLR_dr_0 = np.array(refLR_dr_0)

path3 = '/mnt/HDD1/phudh/super_resolution/RDSR-ReferenceDegradationSR/RDSR/dr_result/4_list_refLR_dr.pickle'

with open(path3, 'rb') as file:
    refLR_dr_final = pickle.load(file)
    refLR_dr_final = np.array(refLR_dr_final)


def plot_tsne_arrays(array_list):

    combined_data = np.concatenate(array_list, axis=0)

    tsne = TSNE(n_components=2, random_state=0)
    data_tsne = tsne.fit_transform(combined_data)

    start_index = 0
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] 
    for i, array in enumerate(array_list):
        array_size = array.shape[0]
        end_index = start_index + array_size
        data_tsne_subset = data_tsne[start_index:end_index, :]
        plt.scatter(data_tsne_subset[:, 0], data_tsne_subset[:, 1], color=colors[i % len(colors)], label=f'array {i+1}')
        start_index = end_index

    plt.title('T-SNE ')
    plt.xlabel('T-SNE Component 1')
    plt.ylabel('T-SNE Component 2')
    plt.legend()
    # plt.show()
    plt.savefig('test.jpg')


plot_tsne_arrays([targetLR_dr, refLR_dr_0, refLR_dr_final])
print(1)
