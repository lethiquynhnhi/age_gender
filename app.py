import streamlit as st 
import cv2 
import numpy as np 
from age_model import load_age_model, predict_age, load_age_model_onnx, predict_age_onnx
from gender_model import load_gender_model, predict_gender, load_gender_model_onnx, predict_gender_onnx
from face_detect_model import load_face_detect_model, face_detect

@st.cache()
def load_model():
    model_under_40 = load_age_model_onnx('./weight/vgg_1_39_05_0.84.onnx')
    model_over_40 = load_age_model_onnx('./weight/vgg_40_60_plus_06_0.96.onnx')
    gender_model = load_gender_model_onnx()
    face_detect_model = load_face_detect_model()

    return model_under_40, model_over_40, gender_model, face_detect_model

st.set_page_config(page_title='Face Analysis',page_icon="😄")  
st.title('Age and Gender Predictor from Face')
model_under_40, model_over_40, gender_model, face_detect_model = load_model()

age_infor = st.empty()
gender_infor = st.empty()
stframe = st.empty()
video_capture = cv2.VideoCapture(0)
i=0

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

            age_precent, age_class = predict_age_onnx(model_under_40, model_over_40, face_crop)
            gender_precent, gender_class = predict_gender_onnx(gender_model, face_crop)

            age_info = '### Age: {} - Percent: {:.2f}%'.format(age_class, age_precent*100)
            gender_info = '### Gender: {} - Percent: {:.2f}%'.format(gender_class, gender_precent*100)

            age_infor.markdown(age_info)
            gender_infor.markdown(gender_info)

    else:
        age_infor.text("")
        gender_infor.text("")
    stframe.image(frame,channels = 'BGR',use_column_width=True)
