import sys
sys.path.append('core')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.mocha_stereo import Mocha
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2
from torch.amp import autocast  # Import for mixed precision

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_image(imfile):
    img = np.array(Image.open(imfile).convert("RGB")).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(Mocha(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt, map_location=torch.device("cpu")))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            # torch.cuda.empty_cache()
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            # Using autocast for mixed precision
            with autocast(device_type='cuda', dtype=torch.float16):
                disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = disp.cpu().numpy()
            disp = padder.unpad(disp)
            file_stem = Path(imfile1).stem
            desired_stem = "_".join(file_stem.split("_")[:-1])
            output_filename = output_directory / f"{desired_stem}.png"
            # print (imfile1)
            # file_stem = imfile1.split('/')[-2]
            # filename = os.path.join(output_directory, f"{file_stem}.png")
            # plt.imsave(output_directory / f"{file_stem}.png", disp.squeeze(), cmap='jet')
            # change for test----------------------------------------------
            # plt.imsave(output_directory / imfile1, disp.squeeze(), cmap='jet')
            plt.imsave(output_filename, disp.squeeze(), cmap='jet')
            # disp = np.round(disp * 256).astype(np.uint16)
            # cv2.imwrite(filename, cv2.applyColorMap(cv2.convertScaleAbs(disp.squeeze(), alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    test_set = 'fan_7_test'
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./model/mocha-stereo.pth')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')

    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./demo-imgs/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./demo-imgs/*/im1.png")
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default=f"./rectified_img_2926_{test_set}/dev1_left/*.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default=f"./rectified_img_2926_{test_set}/dev0_right/*.png")

    parser.add_argument('--output_directory', help="directory to save output", default=f"./demo-output-7-2926-{test_set}/")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    
    args = parser.parse_args()

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    demo(args)
