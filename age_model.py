import numpy as np 
# import tensorflow as tf 
import onnxruntime as ort

AGE_CLASSES = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60+']

def load_age_model(weight_path='./weight/vgg_age_06_1.26.h5'):
    # baseModel = tf.keras.applications.VGG16(weights="imagenet", 
    #                         include_top=False,
    #                         input_tensor=tf.keras.Input(shape=(200, 200, 3)))

    # headModel = baseModel.output
    # headModel = tf.keras.layers.Flatten(name="flatten")(headModel)
    # headModel = tf.keras.layers.Dense(128, activation="relu")(headModel)
    # headModel = tf.keras.layers.Dropout(0.5)(headModel)
    # headModel = tf.keras.layers.Dense(len(AGE_CLASSES), activation="softmax")(headModel)
    
    # model = tf.keras.Model(inputs=baseModel.input, outputs=headModel)
    
    # # load weight
    # model.load_weights(weight_path)
        
    # return model
    pass

def load_age_model_onnx(weight_path='./weight/age_model_06_1.05.onnx'):
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
        }),
        'CPUExecutionProvider',
    ]
    model = ort.InferenceSession(weight_path, providers=providers)

    return model 


def predict_age(model, img):
    img = np.expand_dims(img, axis=0)
    predict = model.predict(img)[0]
    predict_index = np.argmax(predict)
    percent = predict[predict_index]

    predict_class = AGE_CLASSES[predict_index]
    return percent, predict_class

def predict_age_onnx(model1, model2, img):
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    output_names1 = ['dense_1']
    output_names2 = ['dense_3']
    predict1 = model1.run(output_names1, {"input": img})[0][0]
    predict2 = model2.run(output_names2, {"input": img})[0][0]
    predict = np.concatenate((predict1, predict2))
    predict_index = np.argmax(predict)
    percent = predict[predict_index]
    
    predict_class = AGE_CLASSES[predict_index]
    return percent, predict_class

if __name__=="__main__":
    import cv2 

    model_under_40 = load_age_model_onnx('./weight/vgg_1_39_05_0.84.onnx')
    model_over_40 = load_age_model_onnx('./weight/vgg_40_60_plus_06_0.96.onnx')

    img = cv2.imread('../face_age_gender_dataset/24_0_1_20170103212749284.jpg.chip.jpg')
    
    percent, result = predict_age_onnx(model_under_40, model_over_40, img)
    print('Result: {} Percent: {:.2f}%'.format(result, percent*100))