# 열림과 닫힘 연산으로 노이즈 제거 (morph_open_close.py)

import cv2
import numpy as np

img1 = cv2.imread('photos/256_nobg/24bit/KakaoTalk_20211007_224543118-removebg-preview.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('photos/256_nobg/24bit/KakaoTalk_20211007_224543118-removebg-preview.png', cv2.IMREAD_GRAYSCALE)

# 구조화 요소 커널, 사각형 (5x5) 생성 ---①
k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# 블러 함수
k = cv2.blur(k,(5,5))
# 열림 연산 적용 ---②
opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, k)
# 닫힘 연산 적용 ---③
closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, k)

# 결과 출력
merged1 = np.hstack((img1, opening))
merged2 = np.hstack((img2, closing))
merged3 = np.vstack((merged1, merged2))
cv2.imshow('opening, closing', merged3)
cv2.imwrite('result5*5.png', merged3)
cv2.waitKey(0)
cv2.destroyAllWindows()