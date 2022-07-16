import cv2

def load_face_detect_model(weight_path='./weight/yunet.onnx'):
    model = cv2.FaceDetectorYN.create(
        model=weight_path,
        config="",
        input_size=(320, 320),
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000
    )

    return model 

def face_detect(model, img):
    if img is None:
        return None 
        
    height, width = img.shape[0], img.shape[1]

    model.setInputSize(((width, height)))    
    _, face_detect = model.detect(img) 

    return face_detect  