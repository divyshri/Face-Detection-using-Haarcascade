import cv2

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
	ret,img = cam.read()
	

	if ret == False:
		continue

	face = face_cascade.detectMultiScale(img,1.3,5)


	
	for (x,y,w,h) in face:
		cv2.rectangle(img,(x,y),(x+w,y+h),(169,25,55),3)

	cv2.imshow("Video Frame", img)
	
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break


cam.release()
cv2.destroyAllWindows()
