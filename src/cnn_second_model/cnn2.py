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
name = "first_cnn2"

def first_model_cnn2():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(2, 2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(rate=0.25))
    
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer='RMSprop',
              metrics=['accuracy'])


    # Imprimimos el modelo
    model.summary()
    plot_model(model, to_file=name+".png", show_shapes=True, show_layer_names=True)

    return model


def final_model_cnn2():
    # La red tiene sobre aprendizaje porque hay un 3% de difrencia entre el test loss y el training loss
    #--------
    # Vamos a añadir dropout
    # Se ha reducido el sobre aprendizaje, pero ahora la tasa de error es más alta
    # Se ha observado en el modelo que se impreme en la imagen que las capas convulutivas hacen muy pequeña la imagen, se va a cambiar de una de 64 a una de 32
    # El dropout se ha puesto más profundo, entre las capas densas 
    #---------
    # El antepenultipo dropout está mal ajustado, tiene que tener una activación relu
    # Siguiendo lo que he dicho en la memoria voy a reducir el kernel size de la última conv a 2x2 y lo voy a volver a cambiar a 64
    #---------
    #La red vuelve a tener sobre aprendizaje
    # Se cambia kernel size de la última a 1x1 y de la sengunda a 2x2 

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3),activation='relu',input_shape=input_shape, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(2, 2),activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(1, 1),activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(rate=0.25)) Primero que he puesto
    
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(rate=0.25))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer='RMSprop',
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
    model = first_model_cnn2()
    #Evaluamos el modelo
    history, score_test, score_training, time= evaluate.evaluate_model(model=model,e=epochs, b=batch_size, x_data=x_train, y_data=y_train, x_test=x_test, y_test=y_test, val=False)
    #Guardar resultados
    evaluate.plot_results(history=history, score_test=score_test, score_training=score_training, time=time, name=name)