# passive-high-speed-stereo

Start from the home dictionary "*/passive-high-speed-stereo/"

## 1. Install necessary requirement with "requrements.txt"
```
pip install -r Codes/requirements.txt
```
## 2. Install torch from [official website](https://pytorch.org/get-started/locally/)
CUDA is not necessary, all the codes can run with CPU only

## 3. (OPTIONAL) Start from saved iamges captured with synced ISML multicamera setup
(RECOMMEND) You can just skip this step because we have the sorted image sequence in "IMAGES/sort_img/sorted_exp2926_fan_7_test/"\
\
Run "0_test_sort.py" with click the "Run" button in VS Code\
or run the following command at the home dictionary "*/passive-high-speed-stereo/"
```
python Codes/0_test_sort.py
```
*NOTE: you may need to change the folder path if you run this file

## 4. Run the Checkerboard Calibration and Semi-Global Block Matching (SGBM) Disparity
Run "1_test_run_stereo_sequential.py" with click the "Run" button in VS Code\
or run the following command at the home dictionary "*/passive-high-speed-stereo/"
```
python Codes/1_test_run_stereo_sequential.py
```
You can find the output images in "IMAGES/Output_Result/stereo_img_exp2926_fan_7_test" \
The disparity maps using SGBM are under "IMAGES/Output_Result/stereo_img_exp2926_fan_7_test/disparity_map" \
The rectified iamges are under "IMAGES/Output_Result/stereo_img_exp2926_fan_7_test/rectified_img"

## 5. Sort the rectigied images 
Run "2_sort_rectified_img.py" with click the "Run" button in VS Code\
or run the following command at the home dictionary "*/passive-high-speed-stereo/"
```
python Codes/2_sort_rectified_img.py
```
You can find the sorted rectigied images under "IMAGES/Output_Result/stereo_img_exp2926_fan_7_test/sort_rectified_img"

## 6. Run [MoCha-Stereo](https://github.com/ZYangChen/MoCha-Stereo) 
*You need to enter folder "Codes" before you continue\

Now the running directory becomes "*/passive-high-speed-stereo/Codes/" 

Run "3_MoCha-demo.py" with click the "Run" button in VS Code\
or run the following command
```
python Codes/3_MoCha-demo.py
```
You will get the disparity maps using MoCha-Stereo under "IMAGES/Output_Result/MoCha-disparity_map/demo-output-2926-fan_7_test"



