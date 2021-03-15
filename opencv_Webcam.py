import cv2

cam = cv2.VideoCapture(0)

while True:
	ret,img = cam.read()
	grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	if ret == False:
		continue

	cv2.imshow("Video Frame", img)
	cv2.imshow("Gray Frame", grey_img)
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break


cam.release()
cv2.destroyAllWindows()
