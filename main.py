# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time

import pyrealsense2 as rs
import numpy as np
import cv2

# import pyautogui as pag
# from yolo import YOLO


import copy
from pip._vendor.msgpack.fallback import xrange


def distance(x1, y1, x2, y2):
    c = np.math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return c


def centroid(max_contour):
    if max_contour is not None:
        moment = cv2.moments(max_contour)
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            return cx, cy
        else:
            return 0, 0






def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)


def calculateFingers(res, drawing, xc, yc):
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)

    if len(hull) > 20:
        defects = None
        try:
            defects = cv2.convexityDefects(res, hull)
        except Exception as ex:
            pass

        if defects is not None:
            cnt = 0
            # print(res[0])
            max = 0
            ind = ()
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = np.math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = np.math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem

                if np.math.pi >= angle >= np.math.pi * 0.6:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    dist = distance(far[0], far[1], xc, yc)

                    if dist >= max:
                        ind = far
                        max = dist
            for i in range(len(far)):
                cv2.circle(drawing, ind, 4, [211, 84, 0], -1)


def print_hi(name):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    colorizer = rs.colorizer()
    threshold_distance = 0.4
    tr1 = rs.threshold_filter(min_dist=0.15, max_dist=threshold_distance)
    config = rs.config()
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    clipping_distance_in_meters = 0.9
    clipping_distance = clipping_distance_in_meters / depth_scale
    align_to = rs.stream.color
    align = rs.align(align_to)
    pipeline.start(config)
    #thresholds for detect skins
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")


    try:
        while True:
            frames = pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

            # Render images
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))
            # cv2.namedWindow('Background Removed', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Color+depth', images)

            # images[270:, :] = [0, 0, 0]
            # images[0:240,:] = [0,0,0]
            # images[:, 0:50] = [0, 0, 0]
            # images[:, 630:] = [0, 0, 0]

            img_hsv = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2HSV)
            ## Gen lower mask (0-5) and upper mask (175-180) of RED
            lower_red = np.array([94, 80, 2],dtype="uint8")
            upper_red = np.array([126, 255, 255],dtype="uint8")
            mask = cv2.inRange(img_hsv, lower_red, upper_red)
            bluepen = cv2.bitwise_and(bg_removed, bg_removed, mask=mask)


            cv2.imshow('BluePen', bluepen)
            bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
            fgmask = bgModel.apply(bluepen)
            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.erode(fgmask, kernel, iterations=1)
            img = cv2.bitwise_and(bluepen, bluepen, mask=fgmask)

            # Skin detect and thresholding
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([94, 80, 2], dtype="uint8")
            upper = np.array([126, 255, 255], dtype="uint8")
            skinMask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((3, 3), np.uint8)
            skinMask = cv2.erode(skinMask, kernel, iterations=2)
            skinMask = cv2.dilate(skinMask, kernel, iterations=1)


            cv2.imshow('Threshold Hands', skinMask)
            contours, hierarchy = cv2.findContours(skinMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            length = len(contours)
            maxArea = -1
            drawing = np.zeros(img.shape, np.uint8)
            if length > 0:
                for i in xrange(length):
                    temp = contours[i]
                    area = cv2.contourArea(temp)
                    if area > maxArea:
                        maxArea = area
                        ci = i
                        res = contours[ci]
                hull = cv2.convexHull(res)
                # drawing = np.zeros(img.shape, np.uint8)
                cx,cy = centroid(res)
                
                print(aligned_depth_frame.get_distance(cx,cy))
                cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            else:
                drawing = np.zeros(img.shape, np.uint8)
            cv2.imshow('DRAWING', drawing)
























            # bluepen_gray = cv2.cvtColor(bluepen, cv2.COLOR_BGR2HSV)
            # contours, hierarchy = cv2.findContours(bluepen_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # length = len(contours)
            # drawing = np.zeros(bluepen_gray.shape, np.uint8)
            # maxArea = -1
            # if length > 0:
            #     for i in xrange(length):
            #         temp = contours[i]
            #         area = cv2.contourArea(temp)
            #         if area > maxArea:
            #             maxArea = area
            #             ci = i
            #             res1 = contours[ci]
            #
            #     hull = cv2.convexHull(res1)
            #
            #     cv2.drawContours(drawing, [res1], 0, (0, 255, 0), 2)
            #     cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
            # cv2.imshow('',drawing)
            # length = len(contours)
            # print(length)
            # drawing = np.zeros(res.shape, np.uint8)
            # maxArea = -1
            # if length > 0:
            #     for i in xrange(length):
            #         temp = contours[i]
            #         area = cv2.contourArea(temp)
            #         if area > maxArea:
            #             maxArea = area
            #             ci = i
            #             res = contours[ci]
            #
            #     hull = cv2.convexHull(res)
            #
            #
            #     cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            #     cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
            # cv2.imshow('Only contour for calibration', drawing)


            # cv2.imshow('BluePen', res)











            # converted = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)
            # skinMask = cv2.inRange(converted, lower, upper)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # skinMask = cv2.erode(skinMask, kernel, iterations=2)
            # skinMask = cv2.dilate(skinMask, kernel, iterations=1)
            #
            # skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)
            # skin = cv2.bitwise_and(images, images, mask=skinMask)

            cv2.waitKey(1)


            # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # ## Gen lower mask (0-5) and upper mask (175-180) of RED
            # mask1 = cv2.inRange(img_hsv, (0, 50, 20), (5, 255, 255))
            # mask2 = cv2.inRange(img_hsv, (175, 50, 20), (180, 255, 255))
            #
            # ## Merge the mask and crop the red regions
            # mask = cv2.bitwise_or(mask1, mask2)
            # croped = cv2.bitwise_and(img, img, mask=mask)

    finally:

        # Stop streaming
        pipeline.stop()



    # threshold_filter = rs.threshold_filter()
    # threshold_filter.set_option(rs.option.max_distance, 1)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
