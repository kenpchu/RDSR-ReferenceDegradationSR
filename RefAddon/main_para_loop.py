# from main1_v21gt import main as main_program
from options1 import DoeOptions
import importlib
import logging
from datetime import datetime
import os


def main():
    # opt = options()
    opt = DoeOptions()
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    final_str = ''
    baseline = 0
    para_dict = opt.get_para_dict()
    main_module = importlib.import_module(para_dict["main_name"])
    main_func = getattr(main_module, "main")
    cfg_list = opt.load_config_list()

    for ind, cfg in enumerate(cfg_list):
        final_str = ''
        if ind == 0:
            # key, result = main_program(cfg, is_baseline=True)
            key, result = main_func(cfg, is_baseline=True)
            baseline = result
            final_str += f'{key}, {format(result, ".5f")}\n'
        else:
            # key, result = main_program(cfg, baseline=baseline)
            key, result = main_func(cfg, baseline=baseline)
            final_str += f'{key}, {format(result, ".5f")}, {format(result - baseline, ".5f")}\n'
        with open(os.path.join(cfg_list[0]['output_dir'], f"{para_dict['description']}_{timestamp}.csv"), 'a') as fp:
            fp.write(final_str)


if __name__ == '__main__':
    main()
