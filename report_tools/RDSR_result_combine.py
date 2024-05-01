import os
from datetime import datetime
import glob
import time


def split_eval_file(file_path, suffix='im_'):
    ind_list = []
    result_list = []
    with open(file_path, 'r') as fp:
        line_list = fp.readlines()
    img_name_list = []
    for idx, line in enumerate(line_list):
        if suffix in line:
            ind_list.append(idx)
            img_name_list.append(line.split(',')[0])

    score_tuple_list = []

    for ind, idx in enumerate(ind_list):
        if (ind+1) < len(ind_list):
            end_ind = ind_list[ind+1]
            data_list = line_list[idx + 1:end_ind]
        else:
            data_list = line_list[idx + 1:]

        for i, data_line in enumerate(data_list):
            # psnr_list.append(data_line.split(',')[1])
            if 'min_loss_sr_psnr' in data_line:
                score_str = data_list[i+1]
                result_list.append(score_str)
                score_list = score_str.split(',')
                score_tuple_list.append((float(score_list[0]), float(score_list[1]), float(score_list[2])))

    return img_name_list, result_list, score_tuple_list


def main():
    input_dir = 'RDSR_analysis'
    input_files = [
        'eval-bad_0830.csv',
        'eval-bad_0831.csv'
    ]

    input_files = glob.glob('RDSR_analysis/official_x4/eval_*.csv')

    # TODO: 1. add average
    # TODO: 2. support multi combine
    for path in input_files:
        # input_path = os.path.join(input_dir, path)
        input_dir = os.path.dirname(path)
        img_name_list, result_list, score_tuple_list = split_eval_file(path)

        f_timestamp = path.split('.')[0].split('_')[-1]
        csv_file_name = f'eval_cmp_{f_timestamp}.csv'

        rdsr_sum = 0
        rdsr_best_sum = 0
        dasr_sum = 0
        diff_1_sum = 0
        diff_2_sum = 0

        for rdsr, rdsr_best, dasr in score_tuple_list:
            rdsr_sum += rdsr
            rdsr_best_sum += rdsr_best
            dasr_sum += dasr
            diff_1_sum += rdsr - dasr
            diff_2_sum += rdsr_best - dasr

        rdsr_sum /= len(score_tuple_list)
        rdsr_best_sum /= len(score_tuple_list)
        dasr_sum /= len(score_tuple_list)
        diff_1_sum /= len(score_tuple_list)
        diff_2_sum /= len(score_tuple_list)

        with open(os.path.join(input_dir, csv_file_name), 'w') as fp:
            for i in range(len(img_name_list)):
                content = f"{img_name_list[i]}, {result_list[i]}"
                print(content)
                fp.write(content)
            final_result_str = (f",{format(rdsr_sum, '.5f')},{format(rdsr_best_sum, '.5f')},"
                                f"{format(dasr_sum, '.5f')},{format(diff_1_sum, '.5f')},{format(diff_2_sum, '.5f')}")
            fp.write(final_result_str)


def combine(paths):
    fn, ext = os.path.splitext(paths[0])
    dir_name = os.path.dirname(fn)
    fn = os.path.basename(fn)
    combine_path = os.path.join(dir_name, fn.split('_')[0] + '_combine' + ext)
    if os.path.exists(combine_path):
        os.remove(combine_path)
        time.sleep(1)
    all_content = ''
    for f in paths:
        with open(f, 'r') as fp:
            content = fp.read()
        all_content += content + '\n'

    with open(combine_path, 'w') as combine_fp:
        combine_fp.write(all_content)

    return combine_path


def main1():
    # eval_csv path
    input_files = glob.glob('../RDSR/train_loggers/aniso_dr_only_x2/eval_*.csv')

    eval_combine_path = combine(input_files)
    # time.sleep(1)
    # input_path = os.path.join(input_dir, path)
    input_dir = os.path.dirname(eval_combine_path)
    img_name_list, result_list, score_tuple_list = split_eval_file(eval_combine_path)

    f_timestamp = eval_combine_path.split('.')[0].split('_')[-1]
    csv_file_name = f'eval_cmp_{f_timestamp}.csv'

    rdsr_sum = 0
    rdsr_best_sum = 0
    dasr_sum = 0
    diff_1_sum = 0
    diff_2_sum = 0

    for rdsr, rdsr_best, dasr in score_tuple_list:
        rdsr_sum += rdsr
        rdsr_best_sum += rdsr_best
        dasr_sum += dasr
        diff_1_sum += rdsr - dasr
        diff_2_sum += rdsr_best - dasr

    rdsr_sum /= len(score_tuple_list)
    rdsr_best_sum /= len(score_tuple_list)
    dasr_sum /= len(score_tuple_list)
    diff_1_sum /= len(score_tuple_list)
    diff_2_sum /= len(score_tuple_list)

    with open(os.path.join(input_dir, csv_file_name), 'w') as fp:
        for i in range(len(img_name_list)):
            content = f"{img_name_list[i]}, {result_list[i]}"
            print(content)
            fp.write(content)
        final_result_str = (f",{format(rdsr_sum, '.5f')},{format(rdsr_best_sum, '.5f')},"
                            f"{format(dasr_sum, '.5f')},{format(diff_1_sum, '.5f')},{format(diff_2_sum, '.5f')}")
        fp.write(final_result_str)


if __name__ == '__main__':
    # main()
    main1()
