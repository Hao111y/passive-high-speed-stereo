import os
import numpy as np
import cv2
import skimage.io as io

def tile_img(img_in, tile_size=3):
    V_tile = img_in.shape[0] // tile_size
    H_tile = img_in.shape[1] // tile_size

    tiles = np.zeros((tile_size, tile_size, V_tile, H_tile), dtype=img_in.dtype)

    for i in range(tile_size):
        for j in range(tile_size):
            tiles[i, j, ...] = img_in[i*V_tile:(i+1)*V_tile, j*H_tile:(j+1)*H_tile]
    
    return tiles

def sort_img(dirs):
    for dev_id, dir_name in enumerate(dirs):
        # exp_time = "exp2926_fan_8_test"
        exp_time = "exp2926_fan_7_test"

        # item = "checkerboard_cal"
        item = "dynamic"
        # output_dir = f"./sort_img/sorted_{exp_time}/dev_{dev_id}"
        output_dir = f"./sort_img/checkerboard_sort/dev_{dev_id}"
        os.makedirs(output_dir, exist_ok=True)

        image_files = sorted(os.listdir(dir_name))

        for idx, filename in enumerate(image_files):
            file_path = os.path.join(dir_name, filename)
            img = io.imread(file_path)
            if img.ndim == 3 and img.shape[2] > 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            tiles = tile_img(img,3)

            for m in range(tiles.shape[0]):
                for n in range(tiles.shape[1]):
                    # output_filename = f"img_{idx:05}_m{m}_n{n}.png"
                    output_filename = f"{item}_{exp_time}_img_{idx:02}_m{m}_n{n}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, tiles[m, n, ...])
        dev_id += 1

if __name__ == "__main__":
    # dirs = ["./sort_img/to_be_sort/Device_0/", "./sort_img/to_be_sort/Device_1/"] # dirs for normal frames
    dirs = ["./sort_img/checkerboard_cal_subframe/dev_0/", "./sort_img/checkerboard_cal_subframe/dev_1/"]   # dirs for checkerboards

    for dir_name in dirs:
        if not os.path.isdir(dir_name):
            print(f"ERROR:Directory {dir_name} does not exist.")
            exit(1)

    sort_img(dirs)

    print("Image sorting complete.")
