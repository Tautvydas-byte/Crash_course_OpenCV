import cv2

# Opencv DNN(deep neural network)
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")  # yolov4 loaded
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)
# -------------------------------------
# Load class lists
classes = []
with open ("dnn_model/classes.txt","r") as file_object:
	for class_name in file_object.readlines():
		class_name = class_name.strip()
		classes.append(class_name)
print(f"Object classes = {classes}")
#------------------------------------
# Initiliaze camera
camera_port = 0  # 0-first webcam (myPC), 1-second-webcam,2-third and so on
camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)  # cap

while camera.isOpened():
	# Get frames
	ret, frame = camera.read()  # cap.read()
	# Object Detection
	(class_ids, scores, bounding_boxes) = model.detect(frame)
	# print("class_ids",class_ids)
	# print("scores", scores)
	# print("bounding_boxes", bounding_boxes)
	for class_id,score,bounding_boxe in zip(class_ids,scores,bounding_boxes): #yolov5 pats tai padaro tai klausimas ar man to reikes
		(x,y,w,h) = bounding_boxe #coordinates
		print(x,y,w,h)
		class_name = classes[class_id]

		cv2.putText(frame,str(class_name),(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,(200,0,50),2)
		cv2.rectangle(frame,(x,y),(x+w,y+w),(200,0,50),3)

	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
camera.release()  # release webcam
cv2.destroyAllWindows()