import os
import shutil

def sort_files_by_suffix(directory):
    # Define the paths for the subfolders
    dev0_folder = os.path.join(directory, 'dev0_right')
    dev1_folder = os.path.join(directory, 'dev1_left')

    # Create the subfolders if they don't exist
    os.makedirs(dev0_folder, exist_ok=True)
    os.makedirs(dev1_folder, exist_ok=True)

    # Iterate through all the files in the given directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Move files ending with '_dev_0' to the '_dev0_files' folder
        if '_dev_0' in filename:
            shutil.move(file_path, os.path.join(dev0_folder, filename))

        # Move files ending with '_dev_1' to the '_dev1_files' folder
        elif '_dev_1' in filename:
            shutil.move(file_path, os.path.join(dev1_folder, filename))

if __name__ == "__main__":
    # Use the current directory as the target directory
    stereo_path = './stereo_img_exp2926_fan_7_test'
    path_to_be_sort = os.path.join(stereo_path, 'rectified_img')
    sort_files_by_suffix(path_to_be_sort)
    print("Files have been sorted successfully.")
