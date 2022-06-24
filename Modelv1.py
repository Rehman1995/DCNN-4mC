from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D,Concatenate,concatenate, LSTM,LayerNormalization, BatchNormalization, Flatten, Dropout, Dense,SpatialDropout1D,SeparableConv1D
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from tensorflow.keras.losses import poisson

from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import keras
from keras.regularizers import l2 
import numpy as np 



import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, concatenate
    
def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * (m1**2)) + K.sum(w * (m2**2)) + smooth) # Uptill here is Dice Loss with squared
    loss = 1. - K.sum(score)  #Soft Dice Loss
    return loss
def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    #logit_y_pred = y_pred
    
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
    (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)
def Weighted_BCEnDice_loss(y_true, y_pred):
    
    
    # y_true = y_true[...,1:5]
    # y_pred = y_pred[...,1:5]
   
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    avg_pool_1d = tf.keras.layers.AveragePooling1D(pool_size=2, strides=1, padding='valid')
    averaged_mask=(y_true)
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss =  weighted_dice_loss(y_true, y_pred, weight) + weighted_bce_loss(y_true, y_pred, weight) 
    return loss



def prediction(filename,number_of_sequences,model_weights):

    def encode_seq(s):
        Encode = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
        return np.array([Encode[x] for x in s])
    
    def listToString(s):  
    
        str1 = ""  
        
        # traverse in the string   
        for ele in s:  
            str1 += ele   
        
        # return string   
        return str1   
    
    file = open(filename,"r")
    count=0
    Training=[0]*number_of_sequences
    for line in file:
      
      #Let's split the line into an array called "fields" using the ";" as a separator:
      Data = line.split(':')
      Training[count] = Data
      count=count+1
      
  
    
    X1 = {}
    
    accumulator=0
    for row in Training:
      print(row)
      row=listToString(row)
      row=row.strip('\n')
      my_hottie = encode_seq((row))
      out_final=my_hottie
     # out_final=out_final.astype(int)
      out_final = np.array(out_final)
      X1[accumulator]=out_final
      #out_final=list(out_final)
      X1[accumulator] = out_final
      accumulator += 1
      
      
    X1 = list(X1.items())
    
      
    an_array = np.array(X1)
    an_array=an_array[:,1]
    
    transpose = an_array.T
    transpose_list = transpose.tolist()
    X1=np.transpose(transpose_list)
    X1=np.transpose(X1)
      
  
      
   
    
    input_shape = (41,4) # NCP
    inputs = Input(shape = input_shape)
    
    initializer = tf.keras.initializers.RandomUniform()
    c10 = Conv1D(64,11, strides=1, activation='relu', input_shape=(41, 4), padding = 'same')(inputs)
    c1 = BatchNormalization()(c10)
    #c1 = MaxPooling1D(4,strides=2, padding = 'same')(c1)
    #c1 = Dropout(0.1)(c1)
    
    u1 = concatenate([inputs, c1])
    c2 = Conv1D(32,7, strides=1, activation='relu', padding = 'same')(u1)
    c2 = BatchNormalization()(c2)
    u6 = concatenate([u1, c2])

    #c2 = MaxPooling1D(4,strides=2)(c2)
    #c2 = Dropout(0.25)(c2)
    c2 = Conv1D(32,5, strides=1, activation='relu', padding = 'same')(u6)
    c2 = BatchNormalization()(c2)
    u7 = concatenate([u6, c2])


    c2 = MaxPooling1D(4,strides=2)(u7)
    c2 = Dropout(0.30)(c2)

    #c3 = LSTM(16, activation='relu', return_sequences=True)(c2)
    fc = Flatten()(c2)
    #fc0 = Dense(16, activation='relu')(fc)
    #R1=RandomFourierFeatures(300, kernel_initializer="gaussian")(fc)
    
    
    fc1 = Dense(16, activation='relu',kernel_initializer='glorot_uniform',
    bias_initializer='zeros')(fc)
    fc1 = Dense(8, activation='relu',kernel_initializer='glorot_uniform',
    bias_initializer='zeros')(fc1)

    fc2 = Dense(1, activation='sigmoid')(fc1)
    
    #model1 = Model(inputs =[ip], outputs = [fc2])
    
    #model1.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])
    model1 = Model(inputs =[inputs], outputs = [fc2])    
    opt=SGD(learning_rate=0.003, momentum = 0.8)

    model1.compile(loss=Weighted_BCEnDice_loss, optimizer= opt, metrics=['accuracy'])
      
    
    Predict=[]
    
    model1.load_weights(model_weights)
    
    Predict = model1.predict([X1])
    
    Predict=Predict.round();
    return Predict


model_weights='Celeg.h5'
number_of_sequences=2352


Prediction=prediction("positive.txt",number_of_sequences,model_weights)
