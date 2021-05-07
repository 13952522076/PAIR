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
    parser.add_argument('--ID_savepath', type=str, default="/workspace/xuma/PAIR/outputs/",
                        help='save path for Image denosing result.')

    return parser.parse_args()


def main():
    args = parse_args()
    noise_level_img = args.ID_noise                 # noise level for noisy image
    noise_level_model = noise_level_img  # noise level for model
    model_name = args.ID_model           # 'ffdnet_gray' | 'ffdnet_color' | 'ffdnet_color_clip' | 'ffdnet_gray_clip'
    testset_name = 'CBSD68'               # test set,  'bsd68' | 'cbsd68' | 'set12'
    need_degradation = True              # default: True




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
    model_path = os.path.join(model_pool, model_name+'.pth')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from models.network_ffdnet import FFDNet as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)


    img_name, ext = os.path.splitext(args.test_image)
    img_L = util.imread_uint(args.test_image, n_channels=n_channels)
    img_L = util.uint2single(img_L)

    if need_degradation:  # degradation process
        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, noise_level_img/255., img_L.shape)
        if use_clip:
            img_L = util.uint2single(util.single2uint(img_L))

    img_L = util.single2tensor4(img_L)
    img_L = img_L.to(device)

    sigma = torch.full((1,1,1,1), noise_level_model/255.).type_as(img_L)



    img_E = model(img_L, sigma)
    img_E = util.tensor2uint(img_E)

    save_path = os.path.join(args.ID_savepath, os.path.split(img_name)[1]+"_denoising"+str(args.ID_noise)+ext)
    util.imsave(img_E, save_path)
    print(f"Denoised image is saved to {save_path}")


if __name__ == '__main__':

    main()
