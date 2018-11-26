import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def crop_region(img, c_p,w,h):
    """
      This function crop the match region in the input image
      c_p: corner points
    """
    # 3 or 4 channel as the original
    # img = img
    # img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    # mask
    mask = np.zeros(img.shape, dtype=np.uint8)

    # fill the the match region
    channel_count = img.shape[2]
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, c_p, ignore_mask_color)

    # apply the mask
    matched_region = cv2.bitwise_and(img, mask)
    font = cv2.FONT_HERSHEY_COMPLEX
    if w < -10:
        x_decision = 'go right'
    elif w > 10 :
        x_decision = 'go left'
    else:
        x_decision= 'stay'

    if h < -10:
        h_decision = 'go right'
    elif h > 10 :
        h_decision = 'go left'
    else:
        h_decision= 'stay'

    cv2.putText(matched_region, x_decision , (60,60), font, 1.3, (0, 0, 255), 2)
    cv2.putText(matched_region, h_decision, (90, 90), font, 1.3, (0, 0, 255), 2)

    return matched_region

def features_matching(path_temp,path_train):
    """
          Function for Feature Matching + Perspective Transformation
    """
    img1 = cv2.imread(path_temp, 0)   # template
    img2 = cv2.imread(path_train, 0)   # input image
    img1 = cv2.resize(img1, (0, 0), fx=0.25, fy=0.25)
    img2 = cv2.resize(img2, (0, 0), fx=0.25, fy=0.25)

    min_match=10

    # SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # extract the keypoints and descriptors with SIFT

    kps1, des1 = sift.detectAndCompute(img1,None)
    kps2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches (g_matches) as per Lowe's ratio
    g_match = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            g_match.append(m)
    if len(g_match)>min_match:
        src_pts = np.float32([ kps1[m.queryIdx].pt for m in g_match ]).reshape(-1,1,2)
        dst_pts = np.float32([ kps2[m.trainIdx].pt for m in g_match ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        print('center prev: '+ str(w/2) + ' : ' + str(h/2))

        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,255,255) , 3, cv2.LINE_AA)

    else:
        print("Not enough matches have been found! - %d/%d" % (len(g_match), min_match))
        matchesMask = None

    # draw_params = dict(matchColor = (0,255,255),
    #                    singlePointColor = (0,255,0),
    #                    matchesMask = matchesMask, # only inliers
    #                    flags = 2)
    # region corners
    cpoints=np.int32(dst)
    # print(cpoints)
    a, b,c = cpoints.shape

    # reshape to standard format
    c_p=cpoints.reshape((b,a,c))
    # print(c_p)
    # crop matching region
    wid_ch = str(c_p[0][2][0] / 2 - w/2)
    heig_ch = str(c_p[0][2][1] / 2 - h/2)
    matching_region = crop_region(path_train, c_p,wid_ch,heig_ch)
    print('current : ' + str(c_p[0][2][0] / 2) + ' : ' + str(c_p[0][2][1] / 2))
    print('finc changes - width change : ' + str(c_p[0][2][0] / 2 - w/2) + ' height change : ' + str(c_p[0][2][1] / 2 - h/2) )

    # img3 = cv2.drawMatches(img1, kps1, img2, kps2, g_match, None, **draw_params)
    while 1:
        cv2.imshow('sdad',img2)
        cv2.imshow('sdaddsd', matching_region)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    return (matching_region)


def video_feauture(video_path):
    """
          Function for Feature Matching + Perspective Transformation

    """
    cap = cv2.VideoCapture(video_path)
    r,frame1 = cap.read()
    img1 = frame1
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (0, 0), fx=0.35, fy=0.35)

    while 1:
        # img1 = cv2.imread(path_temp, 0)   # template
        r,img2 = cap.read()   # input image
        # print(img2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2 = cv2.resize(img2, (0, 0), fx=0.35, fy=0.35)


        min_match=10


        sift = cv2.xfeatures2d.SIFT_create()



        kps1, des1 = sift.detectAndCompute(img1,None)
        kps2, des2 = sift.detectAndCompute(img2,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        g_match = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                g_match.append(m)
        if len(g_match)>min_match:
            src_pts = np.float32([ kps1[m.queryIdx].pt for m in g_match ]).reshape(-1,1,2)
            dst_pts = np.float32([ kps2[m.trainIdx].pt for m in g_match ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            print('center prev: '+ str(w/2) + ' : ' + str(h/2))

            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,255,255) , 3, cv2.LINE_AA)

        else:
            print("Not enough matches have been found! - %d/%d" % (len(g_match), min_match))
            matchesMask = None
        try:
            # draw_params = dict(matchColor = (0,255,255),
            #                    singlePointColor = (0,255,0),
            #                    matchesMask = matchesMask, # only inliers
            #                    flags = 2)
            # region corners
            cpoints=np.int32(dst)
            # print(cpoints)
            a, b,c = cpoints.shape

            # reshape to standard format
            c_p=cpoints.reshape((b,a,c))
            # print(c_p)
            # crop matching region
            wid_ch = c_p[0][2][0] / 2 - w/2
            heig_ch = c_p[0][2][1] / 2 - h/2
            # matching_region = crop_region(img1, c_p,wid_ch,heig_ch)
            print('current : ' + str(c_p[0][2][0] / 2) + ' : ' + str(c_p[0][2][1] / 2))
            print('finc changes - width change : ' + str(c_p[0][2][0] / 2 - w/2) + ' height change : ' + str(c_p[0][2][1] / 2 - h/2) )
            font = cv2.FONT_HERSHEY_COMPLEX
            if wid_ch < -10:
                x_decision = 'go left'
            elif wid_ch > 10:
                x_decision = 'go right'
            else:
                x_decision = 'stay'

            if heig_ch < -10:
                h_decision = 'go up'
            elif heig_ch > 10:
                h_decision = 'go down'
            else:
                h_decision = 'stay'

            cv2.putText(img2, x_decision, (120, 60), font, 1.3, (0, 0, 255), 2)
            cv2.putText(img2, h_decision, (120, 90), font, 1.3, (0, 0, 255), 2)

            # img3 = cv2.drawMatches(img1, kps1, img2, kps2, g_match, None, **draw_params)

            cv2.imshow('sdad',img2)
            # cv2.imshow('sdaddsd', matching_region)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        except UnboundLocalError :
            continue

    cap.release()
    cv2.destroyAllWindows()


video_feauture('/home/shaaran/PycharmProjects/om/pic_sample/om.mp4')