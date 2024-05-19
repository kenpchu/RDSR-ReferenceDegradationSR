import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import glob

def load_ref_LR_dr(list_path_refLR):
    list_ref_lr_dr = []
    
    for path in list_path_refLR:
        with open(path, 'rb') as file:
            refLR_dr = pickle.load(file)
            if isinstance(refLR_dr, list): refLR_dr = refLR_dr[-1]
            # if len (refLR_dr) >1: refLR_dr = refLR_dr[-1]
            list_ref_lr_dr.append(np.array(refLR_dr)) #.reshape(1,refLR_dr.shape[-2], refLR_dr.shape[-1])
    
    return list_ref_lr_dr

root = '/mnt/HDD1/phudh/super_resolution/RDSR-ReferenceDegradationSR/RDSR/dr_result/baseline_bz_tuneDN_target_ind_0_target_count_10'
for id_lr in [0]:
    # raw_targetLR_DR_path = f'{root}/{id_lr}_LR_0_HR_list_targetLR_dr.pickle'
    raw_targetLR_DR_path = f'{root}/{id_lr}_epoch_0_LR_0_HR_list_targetLR_dr.pickle'



    with open(raw_targetLR_DR_path, 'rb') as file:
        targetLR_dr = pickle.load(file)
        if isinstance(targetLR_dr, list): targetLR_dr = targetLR_dr[-1]
        targetLR_dr = np.array(targetLR_dr)

    # list_ref_lr_dr = load_ref_LR_dr(glob.glob(f'{root}/{id_lr}_LR_*_HR_list_refLR_dr.pickle'))

    original_string = f'{root}/{id_lr}_epoch_0_LR_*_HR_list_refLR_dr.pickle'
    # original_string = f'{root}/{id_lr}_epoch_0_LR_19_HR_list_targetLR_dr.pickle'

    # Tạo danh sách mới với các chuỗi đã thay đổi
    string_list = [original_string.replace('*', str(i)) for i in [0]]
    list_ref_lr_dr = load_ref_LR_dr(string_list)


    def plot_tsne_arrays(array_list, id_lr):
        plt.figure()
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
        plt.savefig(f'{root}/{id_lr}__.jpg')



    plot_tsne_arrays([targetLR_dr] + list_ref_lr_dr, id_lr)
    print(id_lr)
