import argparse
import os.path
import logging

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_image as util


"""
# --------------------------------------------
|--model_zoo             # model_zoo
   |--ffdnet_gray        # model_name, for color images
   |--ffdnet_color
   |--ffdnet_color_clip  # for clipped uint8 color images
   |--ffdnet_gray_clip
|--testset               # testsets
   |--set12              # testset_name
   |--bsd68
   |--cbsd68
|--results               # results
   |--set12_ffdnet_gray  # result_name = testset_name + '_' + model_name
   |--set12_ffdnet_color
   |--cbsd68_ffdnet_color_clip
# --------------------------------------------
"""

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('testing')
    parser.add_argument('--test_image', type=str, required=True,
                        help='input image')
    parser.add_argument('--ID_noise', default=30, type=int,  choices=range(0,101),
                        help='image denoising noise level')
    parser.add_argument('--ID_model', type=str, default="ffdnet_color",
                        help='input image')

    return parser.parse_args()


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = 30                 # noise level for noisy image
    noise_level_model = noise_level_img  # noise level for model
    model_name = 'ffdnet_color'           # 'ffdnet_gray' | 'ffdnet_color' | 'ffdnet_color_clip' | 'ffdnet_gray_clip'
    testset_name = 'CBSD68'               # test set,  'bsd68' | 'cbsd68' | 'set12'
    need_degradation = True              # default: True
    show_img = False                     # default: False




    task_current = 'dn'       # 'dn' for denoising | 'sr' for super-resolution
    sf = 1                    # unused for denoising
    if 'color' in model_name:
        n_channels = 3        # setting for color image
        nc = 96               # setting for color image
        nb = 12               # setting for color image
    else:
        n_channels = 1        # setting for grayscale image
        nc = 64               # setting for grayscale image
        nb = 15               # setting for grayscale image
    if 'clip' in model_name:
        use_clip = True       # clip the intensities into range of [0, 1]
    else:
        use_clip = False
    model_pool = 'model_zoo'  # fixed
    testsets = 'testsets'     # fixed
    results = 'results'       # fixed
    result_name = testset_name + '_' + model_name
    border = sf if task_current == 'sr' else 0     # shave boader to calculate PSNR and SSIM
    model_path = os.path.join(model_pool, model_name+'.pth')

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    print(f"L_path: {L_path}")
    H_path = L_path                               # H_path, for High-quality images
    E_path = os.path.join(results, result_name)
    print(f"E_path: {E_path}")# E_path, for Estimated images
    util.mkdir(E_path)

    if H_path == L_path:
        need_degradation = True
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_ffdnet import FFDNet as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info('model_name:{}, model sigma:{}, image sigma:{}'.format(model_name, noise_level_img, noise_level_model))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    print(f"L_paths: {L_paths}")
    H_paths = util.get_image_paths(H_path) if need_H else None
    print(f"need_H: {need_H}")

    for idx, img in enumerate(L_paths):
        print(f"idx: {idx}   img: {img}")

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)

        if need_degradation:  # degradation process
            np.random.seed(seed=0)  # for reproducibility
            img_L += np.random.normal(0, noise_level_img/255., img_L.shape)
            if use_clip:
                img_L = util.uint2single(util.single2uint(img_L))

        util.imshow(util.single2uint(img_L), title='Noisy image with noise level {}'.format(noise_level_img)) if show_img else None

        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(device)

        sigma = torch.full((1,1,1,1), noise_level_model/255.).type_as(img_L)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        img_E = model(img_L, sigma)
        img_E = util.tensor2uint(img_E)

        if need_H:

            # --------------------------------
            # (3) img_H
            # --------------------------------
            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
            img_H = img_H.squeeze()

            # --------------------------------
            # PSNR and SSIM
            # --------------------------------

            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))
            util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None

        # ------------------------------------
        # save results
        # ------------------------------------

        util.imsave(img_E, os.path.join(E_path, img_name+ext))

    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))

if __name__ == '__main__':

    main()
