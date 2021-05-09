import argparse, time, os, json
from collections import OrderedDict
import imageio

import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description='Test Super Resolution Models')
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to options JSON file.')
    parser.add_argument(
        '-save_folder', type=str, required=True, help='Folder to save output images')
    opt = option.parse(parser.parse_args().opt)
    opt = option.dict_to_nonedict(opt)
    save_folder = parser.parse_args().save_folder


    # initial configure
    scale = opt['scale']
    degrad = opt['degradation']
    network_opt = opt['networks']
    model_name = network_opt['which_model'].upper()

    # create folders
    # util.mkdir_and_rename(opt['path']['res_root'])
    option.save(opt)

    # create test dataloader
    bm_names = []
    test_loaders = []
    for ds_name, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)
        # print('===> Test Dataset: [%s]   Number of images: [%d]' %
        #       (dataset_opt['name'], len(test_set)))
        bm_names.append(dataset_opt['name'])

    # create solver (and load model)
    solver = create_solver(opt)
    # Test phase
    # print('===> Start Test')
    # print("==================================================")
    # print("Method: %s || Scale: %d || Degradation: %s" % (model_name, scale,
    #                                                       degrad))

    for bm, test_loader in zip(bm_names, test_loaders):
        # print("Test set : [%s]" % bm)

        sr_list = []
        path_list = []


        need_HR = False if test_loader.dataset.__class__.__name__.find(
            'HR') < 0 else True

        for iter, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            solver.feed_data(batch, need_HR=need_HR, need_landmark=False)
            solver.test()

            visuals = solver.get_current_visual(need_HR=need_HR)
            sr_list.append(visuals['SR'][-1])

            path_list.append(os.path.basename(batch['LR_path'][0]))

        # # save SR results for further evaluation on MATLAB
        # save_img_path = os.path.join(opt['path']['res_root'], bm)
        #
        # print("===> Saving SR images of [%s]... Save Path: [%s]\n" %
        #       (bm, save_img_path))

        # if not os.path.exists(save_img_path): os.makedirs(save_img_path)
        for img, name in zip(sr_list, path_list):
            imageio.imwrite(os.path.join(save_folder, name), img)

    print("==================================================")
    print("===> Finished !")


if __name__ == '__main__':
    main()
