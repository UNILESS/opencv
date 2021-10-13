import cv2

# Load image, grayscale, Otsu's threshold
image = cv2.imread('photos/256_nobg/24bit/KakaoTalk_20211007_224543118_04-removebg-preview.png')
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Morph close
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

cv2.imshow('',close)
cv2.waitKey()

# Find contours and extract ROI
cnts = cv2.findContours(close, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
print(cnts)
num = 0

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    ROI = original[y:y+h, x:x+w]
    cv2.imwrite('ROI_{}.png'.format(num), ROI)
    num += 1

cv2.imshow('image', image)
cv2.imwrite('result_photo/ROI_INPUT.png', image)
cv2.waitKey()