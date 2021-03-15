import cv2

img = cv2.imread('img.jpg')

cv2.imshow('My Image',img)

cv2.waitKey(0)
cv2.desoyAllWindows()