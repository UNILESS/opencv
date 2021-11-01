import cv2
import numpy as np

# Read image
im_in = cv2.imread("photos/256_nobg/24bit/KakaoTalk_20211007_224543118_04-removebg-preview.png", cv2.IMREAD_GRAYSCALE)

# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255.

th, im_th = cv2.threshold(im_in, 150, 255, cv2.THRESH_BINARY)

'''cv2.imshow("th", im_th)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

# Copy the thresholded image.
im_floodfill = im_th.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 255)

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv

'''cv2.imshow("", im_out)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

contours, hierarchy = cv2.findContours(im_out.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

final_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1:
        final_contours.append(contour)


counter = 0
for c in final_contours:
    counter = counter + 1
# for c in [final_contours[0]]:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.000001 * peri, True)
    x,y,w,h = cv2.boundingRect(approx)
    print(x, y, w, h)
    aspect_ratio = w / float(h)

    if (aspect_ratio >= 0.8 and aspect_ratio <= 4):
        cv2.rectangle(im_in,(x,y),(x+w,y+h),(255,0,0),1)
        cv2.imwrite('splitted_{}.jpg'.format(counter), im_in[y:y+h, x:x+w])
cv2.imwrite('rectangled_split.jpg', im_in)