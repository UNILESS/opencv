import math
import cv2
import numpy as np


def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    # rotate bounding box
    rect0 = ((rect[0]), (rect[1]), 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
               pts[1][0]:pts[2][0]]

    return img_crop


black_img = cv2.imread('mask.png')

image = cv2.imread('SPL_1.png')

original = image.copy()

"""aff = np.array([[1, 0, (image.shape[0] * 3) / 3],
                [0, 1, (image.shape[1] * 3) / 3]], dtype=np.float32)
dst = cv2.warpAffine(image, aff, (image.shape[0] * 3, image.shape[1] * 3))"""

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

cv2.imshow("", black_img)
cv2.waitKey()

k = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))

opening = cv2.morphologyEx(black_img, cv2.MORPH_OPEN, k)

# opening = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, k)

cv2.imshow("", opening)
cv2.waitKey()

gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)

ret, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

cv2.imshow("", thr)
cv2.waitKey()

contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

rect = cv2.minAreaRect(contours[0])

print(rect)
lst = list(rect)
num1, num2, num3, num4, num5 = lst[0][0], lst[0][1], lst[1][0], lst[1][1], int(lst[2])
rect_list = [[num1+7, num2+7], [num3-7, num4-7], num5]
print(rect_list)
img_cropped = crop_minAreaRect(black_img, rect_list)

cv2.imshow('img_cropped', img_cropped)
cv2.imwrite('overlap_result.png', img_cropped)
cv2.waitKey()
