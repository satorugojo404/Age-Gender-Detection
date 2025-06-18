import cv2
import numpy as np

# File paths for models
face_proto = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"
age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"
gender_proto = "gender_deploy.prototxt"
gender_model = "gender_net.caffemodel"

# Constants for age and gender models
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_buckets = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Load models
face_net = cv2.dnn.readNet(face_model, face_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

# Initialize webcam
cap = cv2.VideoCapture(0)


def highlight_face(net, frame, conf_threshold=0.7):
    """ Detects faces and draws rectangles around them. """
    frame_dnn = frame.copy()
    height, width = frame_dnn.shape[:2]
    
    blob = cv2.dnn.blobFromImage(frame_dnn, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()
    
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_dnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(height / 150)))
    
    return frame_dnn, face_boxes


# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    result_img, face_boxes = highlight_face(face_net, frame)
    if not face_boxes:
        print("No face detected")

    for face_box in face_boxes:
        x1, y1, x2, y2 = face_box
        face = frame[max(0, y1):min(y2, frame.shape[0] - 1),
                     max(0, x1):min(x2, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                     MODEL_MEAN_VALUES, swapRB=False)

        # Gender prediction
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Age prediction
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_buckets[age_preds[0].argmax()]

        label = f'{gender}, {age}'
        cv2.putText(result_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Age and Gender Prediction", result_img)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
