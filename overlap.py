import math
import cv2
import numpy as np

def setLabel(img, pts, label):
    (x, y, w, h) = cv2.boundingRect(pts)
    pt1 = (x - 3, y - 3)
    pt2 = (x + w + 3, y + h + 3)
    print(pt1[0], pt1[1]-3)
    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    cv2.putText(img, label, (pt1[0], pt1[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
    return pt1, pt2

black_img = cv2.imread('mask.png')
image = cv2.imread('SPL_1.png')

original = image.copy()

h, w, c = image.shape

roi = black_img[128:128 + h, 128:128 + w]
mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

mask[mask[:] == 255] = 0
mask[mask[:] > 0] = 255
mask_inv = cv2.bitwise_not(mask)
pill = cv2.bitwise_and(image, image, mask=mask)
back = cv2.bitwise_and(roi, roi, mask=mask_inv)
add = cv2.add(pill, back)
black_img[128:128 + h, 128:128 + w] = add

cv2.imshow("black_img_origin", black_img)
cv2.waitKey()

extend_ori = black_img.copy()

gray = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)

ret, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

num = 0

for cont in contours:
    approx = cv2.approxPolyDP(cont, cv2.arcLength(cont, True) * 0.02, True)
    vtc = len(approx)

    area = cv2.contourArea(cont)
    _, radius = cv2.minEnclosingCircle(cont)

    ratio = radius * radius * math.pi / area
    if vtc >= 7:
        setLabel(black_img, cont, 'Large' + str(vtc))
    elif vtc < 7:
        pt1, pt2 = setLabel(black_img, cont, 'small' + str(vtc))
        cv2.rectangle(extend_ori, pt1, pt2, (0, 0, 0), -1)
        num += 1



cv2.imshow('black_img_box', black_img)
cv2.imwrite('img_box.png', black_img)
cv2.imshow('output', extend_ori)
cv2.imwrite('final_output.png', extend_ori)
cv2.waitKey()
cv2.destroyAllWindows()

