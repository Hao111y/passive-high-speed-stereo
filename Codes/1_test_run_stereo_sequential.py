import numpy as np
import skimage.io as io
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import cv_util
from denoise_module import utils_image as util
import torch
from denoise_module.network_dncnn import DnCNN as net

def gamma_correction(img, gamma=0.5):
    if img.max() > 127.5:
        return np.power(img / 255., gamma) * 255.
    else:
        return np.power(img, gamma)

def imread_pipe(file_name, gamma=1.):
    img = io.imread(file_name)
    img = img / 255.
    return np.power(img, gamma) * 255.

def tile_img(img_in, tile_size=3):
    V_tile = img_in.shape[0] // tile_size
    H_tile = img_in.shape[1] // tile_size

    tiles = np.zeros((tile_size, tile_size, V_tile, H_tile))

    for i in range(tile_size):
        for j in range(tile_size):
            tiles[i, j, ...] = img_in[i*V_tile:(i+1)*V_tile, j*H_tile:(j+1)*H_tile]
    
    return tiles

def sort_img(dirs):
    for dev_id, dir_name in enumerate(dirs):
        for idx, j in enumerate(sorted(os.listdir(dir_name))):
            file_name = os.path.join(dir_name, j)
            img = io.imread(file_name)
            tiles = tile_img(img)
            for m in range(3):
                for n in range(3):
                    cv2.imwrite("./dataset/dev_{}/img_{:05}_m_{}_n{}.png".format(dev_id, idx, m, n), tiles[m, n, ...])
        dev_id += 1

def lanczos_upscale_pil(image: np.ndarray, upscale_factor: int, denoise: bool = False, h: float = 10., templateWindowSize: int = 7, searchWindowSize: int = 21) -> np.ndarray:
    pil_image = Image.fromarray(image.astype('uint8'))
    new_size = (int(image.shape[1] * upscale_factor), int(image.shape[0] * upscale_factor))
    upscaled_pil_image = pil_image.resize(new_size, Image.LANCZOS)
    if denoise:
        upscaled_pil_image = cv2.fastNlMeansDenoising(np.array(upscaled_pil_image), None, h, templateWindowSize, searchWindowSize)
    else:
        upscaled_pil_image = np.array(upscaled_pil_image)
    return upscaled_pil_image

def calibrate(img, idx=0):
    gray_img = img.astype('uint8')
    bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    ret, cam_mat, dist, rvec, tvec, corners = cv_util.calibrate(bgr_img)
    
    if not ret:
        raise Exception("[ERROR] Failed to calibrate the camera with the image provided.")

    return rvec, cam_mat, corners

def rectify(img_stored, cam_mat_dict, rot_mat_dict, corner_coords):
    # calculate homography transform matrix to rectify both images
    H_0 = np.matmul(np.linalg.inv(rot_mat_dict[0]), np.linalg.inv(cam_mat_dict[0]))
    H_0 = np.matmul(cam_mat_dict[0], H_0)
    
    H_1 = np.matmul(np.linalg.inv(rot_mat_dict[1]), np.linalg.inv(cam_mat_dict[1]))
    H_1 = np.matmul(cam_mat_dict[1], H_1)
    
    # Calculate height and width of transformed images
    h_0, w_0 = img_stored[0].shape
    homo_coords_0 = np.array([
        [0, 0, w_0, w_0],
        [0, h_0, 0, h_0], 
        [1, 1, 1, 1]
    ])
    
    h_1, w_1 = img_stored[1].shape
    homo_coords_1 = np.array([
        [0, 0, w_1, w_1],
        [0, h_1, 0, h_1], 
        [1, 1, 1, 1]
    ])
    
    transformed_0 = np.matmul(H_0, homo_coords_0)
    transformed_1 = np.matmul(H_1, homo_coords_1)
    # Scaling both to homogenous coordinates
    transformed_0[:, 0] = transformed_0[:, 0]/transformed_0[2, 0]
    transformed_0[:, 1] = transformed_0[:, 1]/transformed_0[2, 1]
    transformed_0[:, 2] = transformed_0[:, 2]/transformed_0[2, 2]
    transformed_0[:, 3] = transformed_0[:, 3]/transformed_0[2, 3]
    
    transformed_1[:, 0] = transformed_1[:, 0]/transformed_1[2, 0]
    transformed_1[:, 1] = transformed_1[:, 1]/transformed_1[2, 1]
    transformed_1[:, 2] = transformed_1[:, 2]/transformed_1[2, 2]
    transformed_1[:, 3] = transformed_1[:, 3]/transformed_1[2, 3]
    
    minx_0 = np.min(transformed_0[0, :])
    miny_0 = np.min(transformed_0[1, :])
    minx_1 = np.min(transformed_1[0, :])
    miny_1 = np.min(transformed_1[1, :]) 
    
    maxx_0 = np.max(transformed_0[0, :])
    maxy_0 = np.max(transformed_0[1, :])
    maxx_1 = np.max(transformed_1[0, :])
    maxy_1 = np.max(transformed_1[1, :]) 
    
    # translate the image to positive x and y coordinates
    translate_mat_0 = np.array(
        [[1, 0, -minx_0],
        [0, 1, -miny_0],
        [0, 0, 1]]
    )    
    translate_mat_1 = np.array(
        [[1, 0, -minx_1],
        [0, 1, -miny_1],
        [0, 0, 1]]
    )    
    
    H_0 = np.matmul(translate_mat_0, H_0)
    H_1 = np.matmul(translate_mat_1, H_1)
    
    
    # calculate new coordinates of the checkerboard corners to put both images to the same scale and fov
    block_coord_0 = np.array(
        [
            [corner_coords[0][0, 0], corner_coords[0][-1, 0]],
            [corner_coords[0][0, 1], corner_coords[0][-1, 1]],
            [1, 1] 
        ]
    )
    
    block_coord_1 = np.array(
        [
            [corner_coords[1][0, 0], corner_coords[1][-1, 0]],
            [corner_coords[1][0, 1], corner_coords[1][-1, 1]],
            [1, 1] 
        ]
    )
    
    transformed_block_coord_0 = np.matmul(H_0, block_coord_0)
    transformed_block_coord_1 = np.matmul(H_1, block_coord_1)
    transformed_block_coord_0[:, 0] = transformed_block_coord_0[:, 0]/transformed_block_coord_0[2, 0]
    transformed_block_coord_0[:, 1] = transformed_block_coord_0[:, 1]/transformed_block_coord_0[2, 1]
    transformed_block_coord_1[:, 0] = transformed_block_coord_1[:, 0]/transformed_block_coord_1[2, 0]
    transformed_block_coord_1[:, 1] = transformed_block_coord_1[:, 1]/transformed_block_coord_1[2, 1]
    
    print(transformed_block_coord_0)
    print(transformed_block_coord_1)
    
    transformed_block_len_0_x = abs(transformed_block_coord_0[0, 1] - transformed_block_coord_0[0, 0])
    transformed_block_len_0_y = abs(transformed_block_coord_0[1, 1] - transformed_block_coord_0[1, 0])
    transformed_block_len_1_x = abs(transformed_block_coord_1[0, 1] - transformed_block_coord_1[0, 0])
    transformed_block_len_1_y = abs(transformed_block_coord_1[1, 1] - transformed_block_coord_1[1, 0])
    
    transformed_scaling_x = transformed_block_len_0_x/transformed_block_len_1_x
    transformed_scaling_y = transformed_block_len_0_y/transformed_block_len_1_y
    
    # Always upscale
    # reverse = False means the scaling is applied to image[1]
    # reverse = True means the scaling is applied to image[0] 
    reverse_x = False
    reverse_y = False
    if transformed_scaling_x < 1:
        transformed_scaling_x = 1/transformed_scaling_x
        reverse_x =  True
    
    if transformed_scaling_y < 1:
        transformed_scaling_y = 1/transformed_scaling_y
        reverse_y =  True
    
    if not reverse_y:
        mincorner_y0 = transformed_block_coord_0[1, 0]
        maxcorner_y0 = transformed_block_coord_0[1, 1]
    else:
        mincorner_y0 = transformed_block_coord_0[1, 0]*transformed_scaling_y
        maxcorner_y0 = transformed_block_coord_0[1, 1]*transformed_scaling_y
    
    if reverse_y:
        mincorner_y1 = transformed_block_coord_1[1, 0]
        maxcorner_y1 = transformed_block_coord_1[1, 1]
    else:
        mincorner_y1 = transformed_block_coord_1[1, 0]*transformed_scaling_y
        maxcorner_y1 = transformed_block_coord_1[1, 1]*transformed_scaling_y

    return H_0, H_1, minx_0, miny_0, minx_1, miny_1, maxx_0, maxy_0, maxx_1, maxy_1, transformed_scaling_x, transformed_scaling_y, reverse_x, reverse_y, \
        mincorner_y0, maxcorner_y0, mincorner_y1, maxcorner_y1

def compute_depth_map(img_stored, H_0, H_1,
                      minx_0, miny_0, minx_1, miny_1,
                      maxx_0, maxy_0, maxx_1, maxy_1,
                      transformed_scaling_x, transformed_scaling_y,
                      reverse_x, reverse_y,
                      mincorner_y0, maxcorner_y0,
                      mincorner_y1, maxcorner_y1, model, img_name="img", gamma=0.5,exp_time="default_exp"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(len(img_stored)):
        img_stored[i] = np.expand_dims(img_stored[i], axis=2)  # HxWx1
        img_stored[i] = util.uint2single(img_stored[i])
        img_stored[i] = util.single2tensor4(img_stored[i])
        # img_stored[i] = img_stored[i].to(device)
        # img_stored[i] = model(img_stored[i])
        img_stored[i] = util.tensor2uint(img_stored[i])
        img_stored[i] = cv2.fastNlMeansDenoising(np.array(img_stored[i]), None, 12, 5, 100)

    # Warp using the homographies
    img_stored[0] = cv2.warpPerspective(img_stored[0], H_0, (int(maxx_0-minx_0), int(maxy_0-miny_0)))
    img_stored[1] = cv2.warpPerspective(img_stored[1], H_1, (int(maxx_1-minx_1), int(maxy_1-miny_1)))

    # Scaling if needed
    if reverse_x:
        img_stored[0] = cv2.resize(img_stored[0], (0, 0), fx=transformed_scaling_x, fy=1)
    else:
        img_stored[1] = cv2.resize(img_stored[1], (0, 0), fx=transformed_scaling_x, fy=1)
    
    if reverse_y:
        img_stored[0] = cv2.resize(img_stored[0], (0, 0), fx=1, fy=transformed_scaling_y)
    else:
        img_stored[1] = cv2.resize(img_stored[1], (0, 0), fx=1, fy=transformed_scaling_y)
    
    # Cropping overlapping FOV
    bottom_y_dist_0 = mincorner_y0
    upper_y_dist_0 = img_stored[0].shape[0] - maxcorner_y0
    
    bottom_y_dist_1 = mincorner_y1
    upper_y_dist_1 = img_stored[1].shape[0] - maxcorner_y1
    
    max_up = min(upper_y_dist_0, upper_y_dist_1)
    max_bottom = min(bottom_y_dist_0, bottom_y_dist_1)
    
    min_width = min(img_stored[0].shape[1], img_stored[1].shape[1])
    cropped_height0 = int(maxcorner_y0+max_up) - int(mincorner_y0-max_bottom)
    cropped_height1 = int(maxcorner_y1+max_up) - int(mincorner_y1-max_bottom)
    cropped_height = min(cropped_height0, cropped_height1)
    img_stored[0] = img_stored[0][int(mincorner_y0-max_bottom):int(mincorner_y0-max_bottom+cropped_height), img_stored[0].shape[1]-min_width:]
    img_stored[1] = img_stored[1][int(mincorner_y1-max_bottom):int(mincorner_y1-max_bottom+cropped_height), :min_width]
    
    # Ensure directories
    top_folder = f"stereo_img_{exp_time}"
    rectified_dir = f"{top_folder}/rectified_img"
    depth_map_dir = f"{top_folder}/depth_map"
    os.makedirs(rectified_dir, exist_ok=True)
    os.makedirs(depth_map_dir, exist_ok=True)

    for i in range(len(img_stored)):
        plt.imshow(gamma_correction(img_stored[i], gamma=gamma), cmap="gray")
        plt.savefig(os.path.join(rectified_dir, "{}_dev_{}.png".format(img_name, i)))
    
    depth = cv_util.calculateDisparity(img_stored[0], img_stored[1])
    plt.imshow(depth)
    plt.savefig(os.path.join(depth_map_dir, "{}.png".format(img_name)))

if __name__ == "__main__":
    gamma_ls = [1.3]
    h_ls = [20] 
    templateWindowSize_ls = [5]
    searchWindowSize_ls = [71]

    exp_time = "exp2926_fan_7_test"
    print(f"We are running stereo on image sequence of {exp_time}")
    device_dirs = [f"sort_img/sorted_{exp_time}/dev_0", f"sort_img/sorted_{exp_time}/dev_1"]
    for dir_name in device_dirs:
        if not os.path.isdir(dir_name):
            print(f"Directory {dir_name} does not exist. Please check the path.")
            exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    images = [os.path.splitext(f)[0] for f in os.listdir(device_dirs[0]) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
    images.sort()

    nb = 17
    n_channels=1
    model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
    model_path = "./denoise_module/dncnn_15.pth"
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    # We have dev_cnt = 2 devices and 9 calibration pairs for each device.
    dev_cnt = 2
    num_calibrations = 9

    # Store the calibration results for each of the 9 sets
    # Each element of these lists will hold a dict for that calibration index
    rot_mat_seq = []
    cam_mat_seq = []
    corner_coords_seq = []

    for gamma in gamma_ls:
        for h in h_ls:
            for templateWindowSize in templateWindowSize_ls:
                for searchWindowSize in searchWindowSize_ls:
                    # Calibrate using 9 pairs of checkerboard images for each device.
                    # After this loop, you'll have 9 sets of calibration results.
                    for cal_idx in range(num_calibrations):
                        rot_mat_dict = {}
                        cam_mat_dict = {}
                        corner_coords_dict = {}

                        for i in range(dev_cnt):
                            # Load the ith device's calibration image number cal_idx+1
                            calib_img_path = f"sort_img/checkerboard_sort/dev_{i}/checkerboard_cal_exp2926_img_0{cal_idx+1}.png"
                            img_cal = imread_pipe(calib_img_path, gamma=gamma)
                            img_cal = lanczos_upscale_pil(img_cal, 3, denoise=True, h=h, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)

                            rvec, cam_mat, corners = calibrate(img_cal, i)
                            rot_mat_dict[i] = cv2.Rodrigues(rvec[0])[0]
                            cam_mat_dict[i] = cam_mat
                            corner_coords_dict[i] = np.squeeze(corners)

                        rot_mat_seq.append(rot_mat_dict)
                        cam_mat_seq.append(cam_mat_dict)
                        corner_coords_seq.append(corner_coords_dict)

                    # Now process each image and cycle through calibration results
                    print("Images list:")
                    for image_name in images:
                        print(image_name)

                    for idx, img_name in enumerate(images):
                        # Choose which calibration set to use
                        # Cycle through 0 to 8 (9 calibration sets)
                        cal_use_idx = idx % num_calibrations

                        # Retrieve the chosen calibration dictionaries
                        rot_mat_dict = rot_mat_seq[cal_use_idx]
                        cam_mat_dict = cam_mat_seq[cal_use_idx]
                        corner_coords = corner_coords_seq[cal_use_idx]

                        img_stored = []
                        for i in range(dev_cnt):
                            img = imread_pipe(f"sort_img/sorted_{exp_time}/dev_{i}/{img_name}.png", gamma=gamma)
                            img = lanczos_upscale_pil(img, 3, denoise=True, h=h, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
                            img_stored.append(img)

                        # Rectify using the chosen calibration data
                        (H_0, H_1, 
                         minx_0, miny_0, minx_1, miny_1, 
                         maxx_0, maxy_0, maxx_1, maxy_1,
                         transformed_scaling_x, transformed_scaling_y,
                         reverse_x, reverse_y,
                         mincorner_y0, maxcorner_y0,
                         mincorner_y1, maxcorner_y1) = rectify(img_stored=img_stored,
                                                               rot_mat_dict=rot_mat_dict,
                                                               cam_mat_dict=cam_mat_dict,
                                                               corner_coords=corner_coords)

                        compute_depth_map(img_stored=img_stored,
                                          model=model,
                                          H_0=H_0,
                                          H_1=H_1,
                                          minx_0=minx_0,
                                          miny_0=miny_0,
                                          minx_1=minx_1,
                                          miny_1=miny_1,
                                          maxx_0=maxx_0,
                                          maxy_0=maxy_0,
                                          maxx_1=maxx_1,
                                          maxy_1=maxy_1,
                                          transformed_scaling_x=transformed_scaling_x,
                                          transformed_scaling_y=transformed_scaling_y,
                                          reverse_x=reverse_x,
                                          reverse_y=reverse_y,
                                          mincorner_y0=mincorner_y0,
                                          maxcorner_y0=maxcorner_y0,
                                          mincorner_y1=mincorner_y1,
                                          maxcorner_y1=maxcorner_y1,
                                          img_name=img_name,
                                          gamma=1./gamma,
                                          exp_time=exp_time)
