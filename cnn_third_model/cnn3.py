from __future__ import print_function

import sys
sys.path.append("../evaluate")

import evaluate

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))



batch_size = 64
num_classes = 10
epochs = 99
input_shape =(28,28,1)
name = "cnn3_EarlyStop"

def model_cnn3():
    # Se ha comporbado que 3 convolutivas son suficientes porque si no la imagen se reduce mucho
    # Vamos a sustituir el max pooling por otra capa convolutiva de 2x2
    # Vamos a añadir kernel regularizer para reducir el overfitting
    #----------------------2
    # La red es muy lenta
    # Vamos a intentar hacerla más rápida
    # Añadimos MaxPooling (la imagen es aún grande, nos lo podemos permitir) y BatchNormalization
    # Reducción de la tasa de aprendizaje factor 0.2
    #-------------------------3
    # Factor de reducción de la tasa de aprendizaje 0.5 y lr 0.0001
    # Sobreaprendizaje, añadimos Dropout
    #-------------------------4
    # Sigo teniendo overfitting
    # Vamos a aumentar el kernel reguralizer y vamos a usar kernel initializer
    # Dropout del 20% - 35%
    #-------------------------5
    # Solucionado el overfitting, mejores resultados
    # Vamos a aumentar el tamaño de la capa densa de 128 a 256
    #-------------------------6
    # Peores resultados, vamos a disminuirla a 64
    #--------------------------7
    # Peores resultados, lo dejamos como estaba a 128
    # Añadimos ruido a las imagenes

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3),activation='relu',input_shape=input_shape, padding='same', kernel_regularizer=l2(0.005), kernel_initializer='he_normal'))
    #model.add(Conv2D(16, kernel_size=(2, 2),activation='relu', kernel_regularizer=l2(0.0005)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.20))

    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same', kernel_regularizer=l2(0.005), kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.30))

    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', kernel_regularizer=l2(0.005), kernel_initializer='he_normal'))
    model.add(Conv2D(64, kernel_size=(1, 1),activation='relu', kernel_regularizer=l2(0.005), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.35))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(rate=0.40))
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
    model = model_cnn3()
    # Ruido para las imagenes
    noise = evaluate.image_noise()
    # Reducción de la tasa de aprendizaje
    reduce_lr = evaluate.reduceLR()
    # Early Stop
    early_stop = evaluate.early_stop()

    #Evaluamos el modelo
    history, score_test, score_training, time= evaluate.evaluate_model(model=model,e=epochs, b=batch_size, x_data=x_train, y_data=y_train, x_test=x_test, y_test=y_test, noise=noise, reduce_lr=reduce_lr, early=early_stop, val=True)
    #Guardar resultados
    evaluate.plot_results(history=history, score_test=score_test, score_training=score_training, time=time, name=name)