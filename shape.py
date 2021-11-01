import math
import cv2
import numpy as np
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
    print(vtc)
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