import cv2 as cv
import numpy as np



def get_hr(path, nb_frames=-1):
    cap = cv.VideoCapture(path)
    hr = []
    ret = True

    while(ret and nb_frames != 0):
        ret, frame = cap.read()
        hr.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        nb_frames -= 1

    return hr

def get_bitstream(path, wize=15, blocksize=7, nb_frames=-1):
    lr = []
    residuals = []
    mv = []

    feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = blocksize)
    lk_params = dict(winSize = (wize,wize), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    cap = cv.VideoCapture(path)

    ret, first_frame = cap.read()
    first_frame = cv.resize(first_frame, (480, 270))
    prev_frame = first_frame
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

    while(ret and nb_frames != 0):
        nb_frames -= 1

        ret, frame = cap.read()

        if(not ret):
            break

        frame = cv.resize(frame, (480, 270))

        residual = cv.subtract(cv.cvtColor(frame, cv.COLOR_BGR2RGB), cv.cvtColor(prev_frame, cv.COLOR_BGR2RGB))
        residuals.append(residual)
        prev_frame = frame

        lr.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

        next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
        good_old = prev[status == 1].astype(int)
        good_new = next[status == 1].astype(int)

        mv_element = []
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            mv_element.append([a, b, c, d])

        mv.append(mv_element)
            
        prev_gray = gray.copy()
        prev = good_new.reshape(-1, 1, 2)

    return lr, residuals, mv

def reconstruct_frame(frame, mv, blocksize, residual):
    
    new_frame = frame.copy()

    blocksize = blocksize * 4
    
    for i in range(len(mv)):
        
        x = mv[i][0] * 4
        y = mv[i][1] * 4
        
        x_old = mv[i][2] * 4
        y_old = mv[i][3] * 4
        
        for j in range(blocksize):
            for k in range(blocksize):
                new_frame[y+j, x+k] = frame[y_old+j, x_old+k] + residual[y+j, x+k]
                new_frame[y_old+j, x_old+k] += residual[y_old+j, x_old+k]
    
    return new_frame

def reconstruct_video(lr, mv, residuals, blocksize=7):
    for i in range(len(lr)):
        lr[i] = reconstruct_frame(lr[i], mv[i], blocksize, residuals[i])