import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def calibrate(img, ser_num=None, square_size=0.021):
    '''
    Conduct checkerboard calibration
    '''
    # calibration starts

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.0001)
    
    num_corner_x = 9
    num_corner_y = 6

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((num_corner_x*num_corner_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:num_corner_x,0:num_corner_y].T.reshape(-1,2)

    # scale objp by size of one square on the checkerboard
    objp *= square_size

    # Arrays to store object points and image points from all the images.
    objpoints1 = [] # 3d point in real world space
    imgpoints1 = [] # 2d points in image plane.

    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()
    # ret1, corners1 = cv2.findChessboardCorners(img, (13,9), None)
    ret1, corners1 = cv2.findChessboardCorners(img, (num_corner_x,num_corner_y), None)

    # refine corner points
    if ret1:
        objpoints1.append(objp)
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners1_refined = cv2.cornerSubPix(gray_img, corners1, (num_corner_x,num_corner_y), (-1,-1), criteria)
        imgpoints1.append(corners1_refined)
        cv2.drawChessboardCorners(img, (num_corner_x,num_corner_y), corners1_refined, ret1)
        #-----------display the checkerboard calibration----------------
        # cv2.imshow("checkerboard img", img)
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.imshow(img_rgb)
        # plt.title("checkerboard img")
        # plt.axis("off")
        # plt.show()
        #-----------End of display the checkerboard calibration----------------
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints1, imgpoints1, gray_img.shape[::-1], None, None, criteria=criteria)
        print("Done calculating camera pose: ")
        print(f"Camera matrix: {mtx}")
        print(f"Rotation vector: {rvecs}")
        print(f"Translation vector: {tvecs}")
        return ret, mtx, dist, rvecs, tvecs, corners1
    else:
        if ser_num is None:
            print(f"No corner found")
        else:
            print(f"No corner found for {ser_num}")
        return False, None, None, None, None, None
    

def sort_key(e):
    return e['score']    

def computeEssential(img1, img2, cam_mat1, cam_mat2):
    '''
    Compute essential and fundamental matrix based on orb feature detector
    '''
    orb = cv2.ORB_create()
 
    # find the keypoints and descriptors with orb
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(np.float32(des1),np.float32(des2),k=2)
    pts1 = []
    pts2 = []
    score_arr = []
    
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            # pts2.append(kp2[m.trainIdx].pt)
            # pts1.append(kp1[m.queryIdx].pt)
            index_score_pair = {
                'index': 0,
                'score': 0
            }
            index_score_pair['index'] = i
            index_score_pair['score'] = m.distance
            score_arr.append(index_score_pair)
    
    print(len(score_arr))

    score_arr.sort(key=sort_key)

    for i in range(15):
        pts2.append(kp2[matches[score_arr[i]['index']][0].trainIdx].pt)
        pts1.append(kp1[matches[score_arr[i]['index']][0].queryIdx].pt)

    pts1_flt = np.array(pts1)
    pts2_flt = np.array(pts2)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    print(f"points1: {pts1}")
    print(f"points2: {pts2}")
    identity = np.eye(3)
    
    # Normalize the feature points based on https://www5.cs.fau.de/fileadmin/lectures/2014s/Lecture.2014s.IMIP/exercises/4/exercise4.pdf
    sum_x1 = 0
    sum_y1 = 0
    sum_x2 = 0
    sum_y2 = 0
    for i in range(pts1.shape[0]):
        sum_x1 += pts1[i, 0]
        sum_y1 += pts1[i, 1]
        sum_x2 += pts2[i, 0]
        sum_y2 += pts2[i, 1]
    
    centroid_x1 = sum_x1/pts1.shape[0]
    centroid_y1 = sum_y1/pts1.shape[0]
    centroid_x2 = sum_x2/pts2.shape[0]
    centroid_y2 = sum_y2/pts2.shape[0]
    
    for i in range(pts1.shape[0]):
        pts1_flt[i, :] = pts1_flt[i, :] - np.array([centroid_x1, centroid_y1])
        pts2_flt[i, :] = pts2_flt[i, :] - np.array([centroid_x2, centroid_y2])
    print(pts1_flt)
    print(pts2_flt)
    
    sum_square_dist1 = 0
    sum_square_dist2 = 0
    for i in range(pts1.shape[0]):
        sum_square_dist1 += pts1_flt[i, 0]**2 + pts1_flt[i, 1]**2
        sum_square_dist2 += pts2_flt[i, 0]**2 + pts2_flt[i, 1]**2

    scaling_factor1 = math.sqrt(2/sum_square_dist1)*pts1.shape[0]
    scaling_factor2 = math.sqrt(2/sum_square_dist2)*pts1.shape[0]

    for i in range(pts1.shape[0]):
        pts1_flt[i, :] = pts1_flt[i, :] * scaling_factor1
        pts2_flt[i, :] = pts2_flt[i, :] * scaling_factor2 

    print(f"final normalized points: {pts1_flt}")
    print(f"final normalized points: {pts2_flt}")

    # Transformation matrix to recover the original fundamental matrix
    transform_mat1 = np.array([
        [scaling_factor1,  0,                 scaling_factor1 * centroid_x1],
        [0,                scaling_factor1,   scaling_factor1 * centroid_y1],
        [0,                0,                  1]
    ])
    
    transform_mat2 = np.array([
        [scaling_factor2,  0,                 scaling_factor2 * centroid_x2],
        [0,                scaling_factor2,   scaling_factor2 * centroid_y2],
        [0,                0,                  1]
    ])

    E, mask = cv2.findEssentialMat(pts1_flt, pts2_flt, identity, method=cv2.RANSAC, prob=0.999, threshold=1, maxIters=3000)
    E_denorm = np.matmul(np.transpose(transform_mat2), E)
    E_denorm = np.matmul(E_denorm, transform_mat1)


    F, mask = cv2.findFundamentalMat(pts1_flt, pts2_flt, cv2.FM_RANSAC, 1, 0.999, maxIters=5000)
    # print(f"Fundamental matrix after normalization is: {F}")
    F = np.matmul(np.transpose(transform_mat2), F)
    F = np.matmul(F, transform_mat1)
    # print(f"Denormalized fundamental mtx: {F_calculated}")

    return E_denorm, F, pts1, pts2


def drawlines(img1,img2,lines,pts1,pts2):
    ''' 
    draw the epipolar lines
    img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines '''
    r,c,_ = img1.shape
    img1_copy = img1.copy()
    img2_copy = img2.copy()
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def computeEpiLines(pts, F, img_num):
    lines = []
    for i in range(pts.shape[0]):
        if(img_num == 2):
            lines.append(np.dot(F, np.array([pts[i, 0], pts[i, 1], 1])))
        else:
            lines.append(np.dot(np.transpose(F), np.array([pts[i, 0], pts[i, 1], 1])))
    return lines

def calculateDisparity(imgL, imgR):
    # Calculate the disparity map
    imgL[imgL < 30] = 0
    imgR[imgR < 30] = 0

    sgbm = cv2.StereoSGBM.create(minDisparity=0, numDisparities=105, blockSize=7, preFilterCap=63,
                             disp12MaxDiff=0, uniquenessRatio=0, speckleWindowSize=3,
                             speckleRange=3, mode=cv2.STEREO_SGBM_MODE_HH4)
    

    right_matcher = cv2.ximgproc.createRightMatcher(sgbm)

    lmbda = 8000
    sigma = 1.3

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)

    disparity_l = sgbm.compute(imgL, imgR)
    disparity_r = right_matcher.compute(imgR, imgL)
    disparity_l = np.int16(disparity_l)
    disparity_r = np.int16(disparity_r)
    filtered_img = wls_filter.filter(disparity_l, imgL, None, disparity_r)

    disparity = filtered_img.astype(np.float32)/16
    disparity[disparity==-1] = 0
    # disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    base = 10
    focal = 250
    disparity[disparity != 0] = base * focal / disparity[disparity!=0]
    disparity[disparity>255] = 255

    disparity = disparity.astype(np.uint8)

    # Normalize disparity for color mapping
    disparity = cv2.normalize(src=disparity, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # -- return cv2.applyColorMap(disparity, cv2.COLORMAP_JET)

    hue_values = np.linspace(0, 120, 121)
    color_disparity = np.empty((disparity.shape[0], disparity.shape[1], 3))

    bar_h = disparity.shape[0]
    bar_w = 100
    color_bar = np.empty((bar_h+20, bar_w+40, 3))
    height_space = 20
    width_space = 40
    inter_space = 15
    
    depth_color_bar = np.empty((disparity.shape[0]+height_space, disparity.shape[1]+bar_w+width_space+inter_space, 3)).astype(np.uint8)

    color_interval = bar_h//len(hue_values)

    for i in range(0, bar_h, color_interval):
        if i < color_interval*len(hue_values):
            color_bar[i+10:i+color_interval+10, :bar_w, 0] = hue_values[i//color_interval]
            color_bar[i+10:i+color_interval+10, :bar_w, 1] = 200
            color_bar[i+10:i+color_interval+10, :bar_w, 2] = 200
        else:
            color_bar[i+10:i+color_interval+10, :bar_w, 0] = hue_values[len(hue_values) - 1]
            color_bar[i+10:i+color_interval+10, :bar_w, 1] = 200
            color_bar[i+10:i+color_interval+10, :bar_w, 2] = 200
    color_bar = cv2.putText(color_bar, "0", (bar_w, 25), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    color_bar = cv2.putText(color_bar, "20", (bar_w, 25 + bar_h//6), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    color_bar = cv2.putText(color_bar, "40", (bar_w, 25 + 2 * bar_h//6), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    color_bar = cv2.putText(color_bar, "60", (bar_w, 25 + 3 * bar_h//6), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    color_bar = cv2.putText(color_bar, "80", (bar_w, 25 + 4 * bar_h//6), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    color_bar = cv2.putText(color_bar, "100", (bar_w, 25 + 5 * bar_h//6), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    color_bar = color_bar.astype(np.uint8)
    color_bar = cv2.cvtColor(color_bar, cv2.COLOR_HSV2BGR)


    for i in range(disparity.shape[0]):
        for j in range(disparity.shape[1]):
            if disparity[i, j] > 0 and disparity[i, j] < len(hue_values):
                color_disparity[i, j, 0] = hue_values[disparity[i, j]]
                color_disparity[i, j, 1] = 200
                color_disparity[i, j, 2] = 200
            elif disparity[i, j] == 0:
                color_disparity[i, j, 0] = 0
                color_disparity[i, j, 1] = 0
                color_disparity[i, j, 2] = 0
            else:
                color_disparity[i, j, 0] = hue_values[len(hue_values) - 1]
                color_disparity[i, j, 1] = 200
                color_disparity[i, j, 2] = 200
    color_disparity = color_disparity.astype(np.uint8)
    color_disparity = cv2.cvtColor(color_disparity, cv2.COLOR_HSV2BGR)
    depth_color_bar[height_space//2:height_space//2+disparity.shape[0], :disparity.shape[1], :] = color_disparity
    depth_color_bar[:, inter_space + disparity.shape[1]:, :] = color_bar

    return depth_color_bar
