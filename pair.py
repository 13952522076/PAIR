import argparse
import os
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--modes', nargs='+', default=['SuperResolution', "Denoising"],
                        choices=["Deblocking", "Denoising", "CompressionArtifactRemoval", "ScratchRemoval",
                                 "FaceEnhancement", "ImageQualityEnhance", "SuperResolution"],
                        help='select image processing model(s)')

    # Super resolution parameters
    parser.add_argument('--SR_type', default="PSNR", type=str, choices=["PSNR", "perception", "balanced"],
                        help='super resolution tyle selection')
    parser.add_argument('--SR_scale', default=2, type=int, choices=[2, 3, 4],
                        help='super resolution upscale factor')


    return parser.parse_args()

def call_functions(args):
    modes = args.modes
    print(f"Ordered running {modes} functions...")
    mode_dict = {
        "SuperResolution": call_SuperResolution,
        "Deblocking": call_Deblocking,
        "Denoising": call_Denoising,
        "CompressionArtifactRemoval": call_CompressionArtifactRemoval,
        "ScratchRemoval": call_ScratchRemoval,
        "FaceEnhancement": call_FaceEnhancement,
        "ImageQualityEnhance": call_ImageQualityEnhance
    }
    # ordered call functions
    for mode in modes:
        mode_dict[mode](args)



def call_SuperResolution(args):
    print(f"Processing Super Resolution: Type:{args.SR_type} | Scale:{args.SR_scale}")
    commands = f"python3 SuperResolution/src/main.py --data_test Demo --scale {args.SR_scale} --pre_train download --test_only --save_results"
    # os.system("cd SuperResolution/src/")
    os.system(commands)

def call_Deblocking(args):
    print(f"Processing Deblocking: ")

def call_Denoising(args):
    print(f"Processing Denoising: ")

def call_CompressionArtifactRemoval(args):
    print(f"Processing Compression Artifact Removal: ")

def call_ScratchRemoval(args):
    print(f"Processing Scratch Removal: ")

def call_FaceEnhancement(args):
    print(f"Processing FaceEn hancement: ")


def call_ImageQualityEnhance(args):
    print(f"Processing Image Quality Enhance: ")

def main():
    args = parse_args()
    call_functions(args)


    # if "SuperResolution" in args.modes:





if __name__ == '__main__':
    main()
