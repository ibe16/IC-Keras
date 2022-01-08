from __future__ import print_function

import sys
sys.path.append("../evaluate")

import evaluate

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))



batch_size = 32
num_classes = 10
epochs = 20
input_shape =(28,28,1)
name = "cnn1"

def model_cnn1():
    # Modelo de red neuronal
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))

    # Compilamos el modelo
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
              metrics=['accuracy'])

    # Imprimimos el modelo
    model.summary()
    plot_model(model, to_file=name+".png", show_shapes=True, show_layer_names=True)

    return model

if __name__ == "__main__":
    #Cargamos los datasets
    x_train, y_train, x_test, y_test = evaluate.load_dataset()

    #Preprocesamiento de los datos
    x_train, x_test = evaluate.prep_dataset(x_train,x_test)
    # Definimos el modelo
    model = model_cnn1()
    #Evaluamos el modelo
    history, score_test, score_training, time= evaluate.evaluate_model(model=model,e=epochs, b=batch_size, x_data=x_train, y_data=y_train, x_test=x_test, y_test=y_test, val=False)
    #Guardar resultados
    evaluate.plot_results(history=history, score_test=score_test, score_training=score_training, time=time, name=name)