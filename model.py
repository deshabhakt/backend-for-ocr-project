# Importing necessary libraries
import cv2
import numpy as np
import pytesseract

import tensorflow as tf
from tensorflow.keras import Model
from keras.layers import Dense, LSTM, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
import keras.backend as K

#ignore warnings in the output
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.layers import Dense, Conv2D, MaxPool2D



def Model1(char_list):
    # input with shape of height=32 and width=128
    inputs = Input(shape=(32, 128, 1))

    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

    conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

    conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
    # poolig layer with kernel size (2,1)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(512, (3, 3), activation='relu',
                    padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

    conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(
        LSTM(256, return_sequences=True, dropout=0.25))(squeezed)
    blstm_2 = Bidirectional(
        LSTM(256, return_sequences=True, dropout=0.25))(blstm_1)

    outputs = Dense(len(char_list)+1, activation='softmax')(blstm_2)

    # model to be used at test time
    act_model = Model(inputs, outputs)

    return act_model, outputs, inputs


def process_image(img):
    """
    Converts image to shape (32, 128, 1) & normalize
    
    """
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    
    w, h = img.shape

#     _, img = cv2.threshold(img,
#                            128,
#                            255,
#                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Aspect Ratio Calculation
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape

    img = img.astype('float32')

    # Converts each to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 128 or w > 32:
        dim = (128, 32)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)

    img = np.expand_dims(img, axis=2)

    # Normalize
    img = img / 255

    return img


def predict_text(image):
    char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    # initializing model
    act_model,outputs,inputs=Model1(char_list)

    # model path
    filepath = r'./model/sgdo-25000r-200e-21143t-2348v.hdf5'
    
    # load the saved best model weights
    act_model.load_weights(filepath)

    # preprocessing uploaded image
    processed_img = process_image(image).reshape((-1, 32, 128, 1))

    prediction = act_model.predict(processed_img)

    decoded = K.ctc_decode(prediction,
                        input_length=np.ones(
                            prediction.shape[0]) * prediction.shape[1],
                        greedy=True)[0][0]

    out = K.get_value(decoded)

    # getting predicted text
    predicted_text = ""
    for _, x in enumerate(out):
        for p in x:
            if int(p) != -1:
                predicted_text += char_list[int(p)]
                
    text = pytesseract.image_to_string(image)

    return predicted_text















