import numpy as np
import cv2

#Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open ("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
#colors = np.random.uniform(0, 255, size = (len(classes), 3))

#Loading image
#img = cv2.imread("fockeyimg.jpg")
#height, width, channels = img.shape
cap = cv2.VideoCapture("UConnFockey.mp4")

if cap.isOpened()== False:
    print('Error opening video file')

while(cap.isOpened()):
    ret, frame = cap.read()
    height, width, channels = frame.shape
    if frame is None:
        break;
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # showing info (bounding boxes, confidence levels)
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            #color = colors[i]
            label = classes[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()

cv2.destroyAllWindows()