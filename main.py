import cv2
import numpy as np

# Opencv DNN(deep neural network)
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")  # yolov4 loaded
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)  # kokios rezuliucijos nuotrauka bus pateikta modeliui
# -------------------------------------
# Load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
	for class_name in file_object.readlines():
		class_name = class_name.strip()
		classes.append(class_name)
print(f"Object classes = {classes}")


# ------------------------------------
# Initiliaze camera
camera_port = 0  # 0-first webcam (myPC), 1-second-webcam,2-third and so on
camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)  # cap
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # pacios naudojamos kameros rezoliucija
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#-----------------------------
button_person = False
# ----Create window-----------
def click_button(event, x, y, flags, params):
	global button_person
	if event == cv2.EVENT_LBUTTONDOWN:
		print(x, y)
		polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])
		is_inside = cv2.pointPolygonTest(polygon,(x,y),False)#tikrina ar pele patenka ant kvadrato
		if is_inside>0:
			print(f"We are clicking inside the button {x,y}")

			if button_person is False:
				button_person = True
			else:
				button_person = False
			print(f"Now button person is {button_person}")



cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)
# ----------------------------
while camera.isOpened():
	# Get frames
	ret, frame = camera.read()  # cap.read()
	# Object Detection
	(class_ids, scores, bounding_boxes) = model.detect(frame)
	# print("class_ids",class_ids)
	# print("scores", scores)
	# print("bounding_boxes", bounding_boxes)
	for class_id, score, bounding_boxe in zip(class_ids, scores,
											  bounding_boxes):  # yolov5 pats tai padaro tai klausimas ar man to reikes
		(x, y, w, h) = bounding_boxe  # coordinates
		# print(x,y,w,h)
		class_name = classes[class_id]
		if class_name == "person":
			cv2.putText(frame, str(class_name), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 50), 2)
			cv2.rectangle(frame, (x, y), (x + w, y + w), (200, 0, 50), 3)

		print(class_name)



#---------------------------
#Create the button
	# cv2.rectangle(frame,(20,20),(220,70),(0,0,200),-1)
	polygon = np.array([[(20,20),(220,20),(220,70),(20,70)]])
	cv2.fillPoly(frame,polygon,(0,0,200))
	cv2.putText(frame,"Person",(30,60),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),3)


	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
camera.release()  # release webcam
cv2.destroyAllWindows()
