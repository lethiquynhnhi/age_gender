import cv2 
import numpy as np 
from age_model import load_age_model, predict_age, load_age_model_onnx, predict_age_onnx
from gender_model import load_gender_model, predict_gender, load_gender_model_onnx, predict_gender_onnx
from face_detect_model import load_face_detect_model, face_detect

age_model = load_age_model_onnx()
gender_model = load_gender_model_onnx()
face_detect_model = load_face_detect_model()

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    faces = face_detect(face_detect_model, frame)
    
    if faces is not None and faces.shape[0] != 0:
        for face in faces:
            x_box, y_box, w_box, h_box = face[:4].astype(int)

            # get corner points of face rectangle
            (x1, y1) = x_box, y_box
            (x2, y2) = x_box+w_box, y_box+h_box

            # draw rectangle over face
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # crop the detected face region
            face_crop = np.copy(frame[y1:y2,x1:x2])

            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue

            # preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (200,200))

            age_precent, age_class = predict_age_onnx(age_model, face_crop)
            gender_precent, gender_class = predict_gender_onnx(gender_model, face_crop)

        age_info = 'Age: {}-{:.2f}%'.format(age_class, age_precent*100)
        gender_info = 'Gender: {}-{:.2f}%'.format(gender_class, gender_precent*100)
        Y = y1 - 20 if y1 - 20 > 20 else y1 + 20
        cv2.putText(frame, age_info, (x1, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, gender_info, (x1, Y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()