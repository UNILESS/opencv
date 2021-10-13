import imgaug.augmenters as iaa
import imgaug as ia
import cv2
import numpy as np

img = cv2.imread("photos/KakaoTalk_20211007_224543118-removebg-preview.png")

aug = ia.imresize_single_image(img, 0.25)

ia.imshow(aug)