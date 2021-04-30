from zipfile import ZipFile
import os
from os import listdir
   
import numpy as np
import pandas as pd 
import random
import os
import zipfile
import tensorflow.keras as keras
import keras.applications.xception as xception
import tensorflow as tf
import re

from PIL import Image
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Input, GlobalAveragePooling2D
from keras.layers.experimental.preprocessing import Normalization
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.layers import Lambda
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


   

def get_paths():

    # get a path to the main input
    INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')
    print(listdir(INPUTS_DIR))
    pathtozip = os.path.join(INPUTS_DIR,'zipfileishere')
    print(listdir(pathtozip))
    os.chdir(pathtozip)
    print(os.getcwd())
    print(listdir())
    # archive_path = os.path.join(INPUTS_DIR, archive_file)

    # unzip the folder
    pathtozip = os.path.join(INPUTS_DIR,'zipfileishere/archive.zip')
    print(os.path.isfile(pathtozip))
    with ZipFile(pathtozip, 'r') as zipObj:
    # Extract all the contents of zip file in current directory
        zipObj.extractall()
    print('Files extracted')

    # Take a path to each one of the folder+file
    print(listdir(os.path.join(INPUTS_DIR,'zipfileishere')))
    archive_path = os.path.join(INPUTS_DIR, 'zipfileishere/garbage_classification')
    paths=[]
    labels=[]
    for folder in listdir(archive_path):
        for picture in listdir(os.path.join(archive_path, folder)):
            _ = os.path.join(os.path.join(archive_path, folder), picture)
            paths.append(_)
            labels.append(folder)
        print('Completed for ',folder)

    path_to_model = os.path.join(INPUTS_DIR, 'xception-model/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')

    return paths,labels,path_to_model,archive_path


def main():
    paths,labels,path_to_model,archive_path = get_paths()
    df = pd.DataFrame({
    'filename': paths,
    'category': labels
    })

    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    print('number of elements = ' , len(df))
    print(df)

    # Modelling

    # Defining constants
    IMAGE_WIDTH = 320    
    IMAGE_HEIGHT = 320
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
    IMAGE_CHANNELS = 3

    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
    import keras.applications.xception as xception

    xception_layer = xception.Xception(include_top = False, input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS),
                        weights = path_to_model) 

    # We don't want to train the imported weights
    xception_layer.trainable = False


    # initialize object
    model = Sequential()

    # input of shape 320x320x3 
    model.add(keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

    #create a custom layer to apply the preprocessing
    def xception_preprocessing(img):
        return xception.preprocess_input(img)

    model.add(Lambda(xception_preprocessing))

    model.add(xception_layer)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(Dense(len(set(labels)), activation='softmax')) 

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    model.summary()

    #Change the categories from numbers to names <<<<< Not needed anymore since it has the categories since the beginning
    # df["category"] = df["category"].replace(categories) 

    # Splitting

    # We first split the data into two sets and then split the validate_df to two sets, test and validation
    train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
    validate_df, test_df = train_test_split(validate_df, test_size=0.5, random_state=42)

    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]

    early_stop = EarlyStopping(patience = 2, verbose = 1, monitor='val_categorical_accuracy' , mode='max', min_delta=0.001, restore_best_weights = True)
    callbacks = [early_stop]
    print('call back defined!')

    print('train size = ', total_train , 'validate size = ', total_validate, 'test size = ', test_df.shape[0])

    base_path = archive_path

    # data augmentation set up
    batch_size=64
    train_datagen = image.ImageDataGenerator(
        
        ###  Augmentation Start  ###
        
        rotation_range=30,
        shear_range=0.1,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip = True,
        width_shift_range=0.2,
        height_shift_range=0.2
        
        ##  Augmentation End  ###
    )

    # get input from train dataset
    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        base_path, 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    # augmentation is not applied to the validation dataset
    validation_datagen = image.ImageDataGenerator()

    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        base_path, 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    # fit the model
    EPOCHS = 10
    history = model.fit_generator(
        train_generator, 
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=total_validate//batch_size,
        steps_per_epoch=total_train//batch_size,
        callbacks=callbacks
    )

    # model.save_weights("model.h5") # modify this to save the output properly <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Apply model to test and calculate accuracy 
    test_datagen = image.ImageDataGenerator()

    test_generator = test_datagen.flow_from_dataframe(
        dataframe= test_df,
        directory=base_path,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=1,
        shuffle=False 
    )

    # Accuracy

    filenames = test_generator.filenames
    nb_samples = len(filenames)
    _, accuracy = model.evaluate_generator(test_generator, nb_samples)

    # We defined at the beginning of this notebook a dictionary that maps the categories number to names, but the train generator
    # generated it's own dictionary and it has assigned different numbers to our categories and the predictions made by the model 
    # will be made using the genrator's dictionary.

    gen_label_map = test_generator.class_indices
    gen_label_map = dict((v,k) for k,v in gen_label_map.items())
    print(gen_label_map)

    # get the model's predictions for the test set
    preds = model.predict_generator(test_generator, nb_samples)

    # Get the category with the highest predicted probability, the prediction is only the category's number and not name
    preds = preds.argmax(1)

    # Convert the predicted category's number to name 
    preds = [gen_label_map[item] for item in preds]

    # Convert the pandas dataframe to a numpy matrix
    labels = test_df['category'].to_numpy()

    print(classification_report(labels, preds))
    print('accuracy on test set = ',  round((accuracy * 100),2 ), '% ')

    return model

def to_lite(model):
    # Converting a tf.Keras model to a TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model

def save_keras(model):
    
    outputs_dir = os.getenv('VH_OUTPUTS_DIR', './outputs')

    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir)
    
    save_path = os.path.join (outputs_dir, "model.h5")

    model.save_weights(save_path) #model.save(outputs_dir) # save keras .h5 model

def save_lite(model):

    outputs_dir = os.getenv('VH_OUTPUTS_DIR', './outputs')

    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir)

    save_path = os.path.join (outputs_dir, model+'.tflite')

    # Save the model.
    with open(save_path, 'wb') as f:
        f.write(model)

    print('Model was saved')


if __name__ == '__main__':
    get_paths()
    keras_model = main()
    save_keras(keras_model)
    