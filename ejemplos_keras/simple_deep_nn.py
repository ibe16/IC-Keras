import sys
sys.path.append("../evaluate")

import evaluate

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D ,BatchNormalization
from keras.optimizers import RMSprop
from keras.utils.vis_utils import plot_model



batch_size = 128
num_classes = 10
epochs = 20


def model_simple():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file="simple_deep_nn.png", show_shapes=True, show_layer_names=True)

    return model


if __name__ == "__main__":
    #Cargamos los datasets
    x_train, y_train, x_test, y_test = evaluate.load_dataset()
    
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    #Preprocesamiento de los datos
    x_train, x_test = evaluate.prep_dataset(x_train,x_test)
    # Definimos el modelo
    model = model_simple()
    #Evaluamos el modelo
    history, score_test, score_training, time= evaluate.evaluate_model(model=model,e=epochs, b=batch_size, x_data=x_train, y_data=y_train, x_test=x_test, y_test=y_test, val=False)
    #Guardar resultados
    evaluate.plot_results(history=history, score_test=score_test, score_training=score_training, time=time, name="simple_deep_nn")