import numpy as np 
# import tensorflow as tf 
import onnxruntime as ort

GENDER_CLASSES = ['Male', 'Female']

def load_gender_model(weight_path='./weight/vgg_gender_05_0.40.h5'):
    # baseModel = tf.keras.applications.VGG16(weights="imagenet", 
    #                         include_top=False,
    #                         input_tensor=tf.keras.Input(shape=(200, 200, 3)))

    # headModel = baseModel.output
    # headModel = tf.keras.layers.AveragePooling2D(pool_size=(3, 3))(headModel)
    # headModel = tf.keras.layers.Flatten(name="flatten")(headModel)
    # headModel = tf.keras.layers.Dense(128, activation="relu")(headModel)
    # headModel = tf.keras.layers.Dropout(0.5)(headModel)
    # headModel = tf.keras.layers.Dense(1, activation="sigmoid")(headModel)
    
    # model = tf.keras.Model(inputs=baseModel.input, outputs=headModel)
    
    # # load weight
    # model.load_weights(weight_path)
        
    # return model
    pass

def load_gender_model_onnx(weight_path='./weight/gender_model_05_0.40.onnx'):
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
        }),
        'CPUExecutionProvider',
    ]
    model = ort.InferenceSession(weight_path, providers=providers)

    return model 


def predict_gender(model, img):
    img = np.expand_dims(img, axis=0)
    predict = model.predict(img)[0][0]
    percent = predict if predict > 0.5 else 1-predict

    predict_class = GENDER_CLASSES[round(predict)]
    return percent, predict_class

def predict_gender_onnx(model, img):
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    output_names = ['dense_3']
    predict = model.run(output_names, {"input": img})[0][0][0]

    percent = predict if predict > 0.5 else 1-predict

    predict_class = GENDER_CLASSES[round(predict)]
    return percent, predict_class

if __name__=="__main__":
    import cv2 

    model = load_gender_model_onnx()

    # img = cv2.imread('/home/giabao/Documents/face/quynhnhi/crop_part1/40_0_0_20170103182925914.jpg.chip.jpg')
    img = cv2.imread('../face_age_gender_dataset/24_0_1_20170103212749284.jpg.chip.jpg')
    
    percent, result = predict_gender_onnx(model, img)
    print('Result: {} Percent: {:.2f}%'.format(result, percent*100))