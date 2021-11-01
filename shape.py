import math
import cv2
import numpy as np

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]

    return img_crop


def setLabel(img, pts, label):
    (x, y, w, h) = cv2.boundingRect(pts)
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    print(pt1[0], pt1[1]-3)
    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    cv2.putText(img, label, (pt1[0], pt1[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))


def imsave(img, pts, num):
    rect = cv2.minAreaRect(pts)
    print(rect)
    crop_min_rect = crop_minAreaRect(img, rect)
    cv2.imwrite('SPL_{}.png'.format(num), crop_min_rect)

    (x, y, w, h) = cv2.boundingRect(pts) # 현재 angle 값 못 넣고 있음.
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    ROI = original[y-3:y+h+3, x-3:x+w+3]
    cv2.imwrite('SPL_{}.png'.format(num), ROI)


img = cv2.imread('photos/KakaoTalk_20211007_224543118-removebg-preview.png')
original = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

num = 0
for cont in contours:
    approx = cv2.approxPolyDP(cont, cv2.arcLength(cont, True) * 0.02, True)
    vtc = len(approx)

    area = cv2.contourArea(cont)
    _, radius = cv2.minEnclosingCircle(cont)

    ratio = radius * radius * math.pi / area
    if vtc >= 8:
        setLabel(img, cont, 'NonSpl ' + str(vtc))
    elif vtc < 8:
        setLabel(img, cont, 'Spl ' + str(vtc))
        imsave(img, cont, num)
        num += 1


cv2.imshow('img', img)
cv2.imshow('binary', thr)
cv2.imwrite('output.png', img)
cv2.waitKey()
cv2.destroyAllWindows()