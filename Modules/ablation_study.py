# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:38:01 2021

@author: 55114405 (Connah Kendrick)

Module to train deep learning models for ISIC binary classification.

Possible models from https://keras.io/api/applications/ are as follows:
    '''['VGG16','VGG19','Xception','ResNet50','ResNet101','ResNet152','ResNet50V2',
                      'ResNet101V2','ResNet152V2','InceptionV3','InceptionResNetV2','DenseNet121',
                      'DenseNet169','DenseNet201','EfficientNetB0','EfficientNetB1','EfficientNetB3',
                      'EfficientNetB4','EfficientNetB5','EfficientNetB6','EfficientNetB7']'''


"""

import tensorflow as tf  
import cv2 as cv  
import numpy as np
import os
import time 
import re
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(42)
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())
print(tf.test.gpu_device_name())
print(tf.__version__)
print(np.__version__)



###############################################################################################################

# !! set save path here
save_path = r'..\\Models\\Inpaint_Telea'

# !! set data paths here
src_path_train = r"..\\Data\\train_balanced_224x224_inpainted_telea\\train"
src_path_val = r"..\\Data\\train_balanced_224x224_inpainted_telea\\val"

# !! set models to train here
models_to_train = ['EfficientNetB4']

###############################################################################################################
# Extra parameters
batch_sizes = [64]
optimisers = ['SGD']
class_count = 2
training_loops = 1
max_size = 224
num_gpus = 1
num_epochs = 200
weights = None

def getDataGens(batch_size):
    """Data pre-processing method used to prepare dataset for training and validation
    steps.
    
    Parameters
    ----------
    batch_size
        the batch size of the model to train
    
    """
    train_datagen = ImageDataGenerator(
            rescale=1 / 255.0,
            rotation_range=20,
            zoom_range=0.05,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest")

    val_datagen = ImageDataGenerator(
            rescale=1 / 255.0,
            fill_mode="nearest") 
    
    train_generator = train_datagen.flow_from_directory(
        directory=src_path_train,
        color_mode="rgb",
        batch_size=batch_size,
        target_size = (224,224),
        class_mode="categorical",
        subset='training',
        shuffle=True,
        seed=42
    )
    
    valid_generator = val_datagen.flow_from_directory(
        directory=src_path_val,
        color_mode="rgb",
        batch_size=batch_size,
        target_size = (224,224),
        class_mode="categorical",
        subset='training',
        shuffle=True,
        seed=42
    ) 
    return train_datagen, val_datagen, train_generator, valid_generator

"""Train required models"""
for model_count in range(len(models_to_train)):
    model_type = models_to_train[model_count] 
    if not os.path.exists(os.path.join(save_path,model_type)):
            os.mkdir(os.path.join(save_path,model_type))
    currSavePath = os.path.join(save_path,model_type)
    for opt_count in range(len(optimisers)):
        opt_type = optimisers[opt_count]
        if not os.path.exists(os.path.join(currSavePath,opt_type)):
            os.mkdir(os.path.join(currSavePath,opt_type))
        currSavePath = os.path.join(currSavePath,opt_type)
        model = None 
        for batchSizes in range (len(batch_sizes)): 
            tf.keras.backend.clear_session()
            if not os.path.exists(os.path.join(currSavePath,str(batch_sizes[batchSizes]))):
                os.mkdir(os.path.join(currSavePath,str(batch_sizes[batchSizes])))
            currSavePath = os.path.join(currSavePath,str(batch_sizes[batchSizes]))
            train_datagen, val_datagen, train_generator, valid_generator = getDataGens(batch_sizes[batchSizes])
            model_func = getattr(tf.keras.applications,model_type) 
            model = model_func(include_top=True,
                weights=weights,
                input_tensor=None,
                input_shape=(224,224,3),
                pooling='avg',
                classes=class_count) #(2 *4096 dense then 100 for classes)
            #x = Dense(4096, activation='relu')(model.output)
            #x = Dense(4096, activation='relu')(x)
            #predictions = Dense(class_count, activation='softmax')(x)
            
            model = Model(inputs=model.input, outputs=model.output)
            model.summary()
            #if num_gpus > 1:
                #model = multi_gpu_model(model,gpus=num_gpus)
            csv_logger =  [
                    tf.keras.callbacks.CSVLogger(
                        os.path.join(currSavePath,model_type+'_batchSize_'+str(batchSizes)+"_opt_"+opt_type+".csv"), separator=",", append=False),
                    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(currSavePath,model_type+'_batchSize_'+str(batchSizes)+"_opt_"+opt_type+'_'+'model.{epoch:02d}.h5'),
                                                       save_best_only=True,
                                                       monitor="val_accuracy",
                                                       mode='max',
                                                       save_weights_only=False),
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_accuracy",
                        min_delta=0,
                        patience=10,
                        verbose=0,
                        mode="max",
                        baseline=None,
                        restore_best_weights=False,)]
            opt_func = getattr(tf.keras.optimizers,optimisers[opt_count])
            print("using SGD due to error")
            opt =  SGD()#opt_func(opt_func())
            print("try binary_crossentropy")#keras.losses.categorical_crossentropy
            model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy', 'AUC'])   
            model.fit(train_generator, validation_data = valid_generator,epochs=num_epochs, 
                                callbacks=csv_logger, steps_per_epoch = train_generator.n//train_generator.batch_size,
                                validation_steps=valid_generator.n//valid_generator.batch_size) 
            
