import cv2
import numpy as np

# Load models for face detection, age prediction, and gender prediction
face_proto = "opencv_face_detector.pbtxt"  #configuration file that describes the model's architecture
face_model = "opencv_face_detector_uint8.pb"  #TensorFlow pre-trained model file used to detect faces
age_proto = "age_deploy.prototxt"  #file that defines the architecture of the age detection network
age_model = "age_net.caffemodel"  #pre-trained Caffe model file that contains the weights
gender_proto = "gender_deploy.prototxt"  #file that defines the architecture of the gender detection network
gender_model = "gender_net.caffemodel"  #pre-trained Caffe model file that contains the weights

# Define mean values and other parameters for age and gender models


#the input image's color values are adjusted to the model's expected range
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746) #mean BGR (Blue, Green, Red)

#highest probability corresponds to one of these age groups.
age_buckets = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Load Pre-Trained networks
face_net = cv2.dnn.readNet(face_model, face_proto)  #face detection model.
age_net = cv2.dnn.readNet(age_model, age_proto)  #The age prediction model.
gender_net = cv2.dnn.readNet(gender_model, gender_proto)  #The gender prediction model.


# Set up video capture (webcam)
cap = cv2.VideoCapture(0)


#Function for Detecting Faces in the Frame
def highlight_face(net, frame, conf_threshold=0.7):
    """ Detect faces in the frame and return face rectangles. """
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]  # Height of the image
    frame_width = frame_opencv_dnn.shape[1]   # Width of the image
    #resizing, normalizing, and converting it to the required format.
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    face_boxes = []

    for i in range(detections.shape[2]):     # number of detected faces
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width) ## x-coordinate of the top-left corner
            y1 = int(detections[0, 0, i, 4] * frame_height) ## y-coordinate of the top-left corner 
            x2 = int(detections[0, 0, i, 5] * frame_width)  ## x-coordinate of the bottom-right corner
            y2 = int(detections[0, 0, i, 6] * frame_height) ## y-coordinate of the bottom-right corner
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height/150)))
    
    return frame_opencv_dnn, face_boxes

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    result_img, face_boxes = highlight_face(face_net, frame)
    if not face_boxes:
        print("No face detected")
    
    for face_box in face_boxes:
        # Extract the face region
        face = frame[max(0, face_box[1]):min(face_box[3], frame.shape[0]-1),
                     max(0, face_box[0]):min(face_box[2], frame.shape[1]-1)]

        #face_box[0]: x-coordinate of the top-left corner of the face.
        #face_box[1]: y-coordinate of the top-left corner of the face.
        #face_box[2]: x-coordinate of the bottom-right corner of the face.
        #face_box[3]: y-coordinate of the bottom-right corner of the face.

        
        # Prepare blob for age and gender prediction
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_buckets[age_preds[0].argmax()]

        # Display results
        cv2.putText(result_img, f'{gender}, {age}', (face_box[0], face_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0, 255), 2, cv2.LINE_AA)
    
    # Display the resulting frame with detected faces, age, and gender
    cv2.imshow("Age and Gender Prediction", result_img)

    # Break the loop on 'q' key press
    key = cv2.waitKey(1)
    if key == 27:
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
