import os
import glob
import re
from datetime import datetime
import time

def combine(paths):
    fn, ext = os.path.splitext(glob.glob(paths)[0])
    dir_name = os.path.dirname(fn)
    fn = os.path.basename(fn)
    combine_path = os.path.join(dir_name, fn.split('_')[0] + 'Combine' + ext)
    if os.path.exists(combine_path):
        os.remove(combine_path)
    time.sleep(1)
    all_content = ''
    for f in glob.glob(paths):
        with open(f, 'r') as fp:
            content = fp.read()
        all_content += content + '\n'

    with open(combine_path, 'w') as combine_fp:
        combine_fp.write(all_content)

    return combine_path


def split_eval_file(file_path, suffix='x2'):
    ind_list = []
    with open(file_path, 'r') as fp:
        line_list = fp.readlines()
    img_name_list = []
    for idx, line in enumerate(line_list):
        if suffix in line:
            ind_list.append(idx)
            img_name_list.append(line.split(',')[0])

    result_tuple_list = []

    for ind, idx in enumerate(ind_list):
        if (ind+1) < len(ind_list):
            end_ind = ind_list[ind+1]
            data_list = line_list[idx + 1:end_ind]
        else:
            data_list = line_list[idx + 1:]

        for i, data_line in enumerate(data_list):
            # psnr_list.append(data_line.split(',')[1])
            if 'max_sr_psnr' in data_line:
                score_str = data_list[i+1]
                score_list = score_str.split(',')
                result_tuple_list.append((img_name_list[ind], float(score_list[0]), float(score_list[1]), float(score_list[2]), score_str))

    result_tuple_list = sorted(result_tuple_list, key=lambda x: x[0])
    return result_tuple_list


def write_final_result(eval_path, result_tuple_list):
    csv_file_name = f'evalCombine_cmp.csv'

    min_loss_sum = 0
    max_psnr_sum = 0
    last_psnr_sum = 0

    for result_data in result_tuple_list:
        # img_name = result_data[0]
        min_loss_psnr = result_data[1]
        max_psnr = result_data[2]
        last_psnr = result_data[3]
        # max_iter = result_data[4]
        # result_str = result_data[5]

        min_loss_sum += min_loss_psnr
        max_psnr_sum += max_psnr
        last_psnr_sum += last_psnr

    min_loss_sum /= len(result_tuple_list)
    max_psnr_sum /= len(result_tuple_list)
    last_psnr_sum /= len(result_tuple_list)

    input_dir = os.path.dirname(eval_path)
    with open(os.path.join(input_dir, csv_file_name), 'w') as fp:
        for i in range(len(result_tuple_list)):
            content = f"{result_tuple_list[i][0]}, {result_tuple_list[i][-1]}"
            print(content)
            fp.write(content)
        final_result_str = f",{format(min_loss_sum, '.5f')},{format(max_psnr_sum, '.5f')},{format(last_psnr_sum, '.5f')}"
        fp.write(final_result_str)


def main_x2():
    eval_dir = '../../RDSR_dualbase/train_log/DIV2K_x4_1228/csv/eval_*.csv'

    suffix = 'im_'
    suffix = 'x2'
    suffix = 'x4'

    eval_combine_path = combine(eval_dir)

    result_tuple_list = split_eval_file(eval_combine_path, suffix=suffix)
    if suffix == 'im_':
        result_tuple_list = sorted(result_tuple_list, key=lambda x: int(x[0].split('_')[1]))
    write_final_result(eval_combine_path, result_tuple_list)



if __name__ == '__main__':
    main_x2()
