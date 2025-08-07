from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Load dữ liệu
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Tiền xử lý
x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255
x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255

# One-hot encoding cho nhãn
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Tăng cường dữ liệu
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)

# Xây dựng model phân loại
def build_model(hp):
    input_layer = Input(shape=(28, 28, 1))
    x = Conv2D(
        hp.Int('conv1_filters', min_value=32, max_value=128, step=32),
        (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(
        hp.Int('conv2_filters', min_value=32, max_value=128, step=32),
        (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(
        hp.Int('conv3_filters', min_value=32, max_value=128, step=32),
        (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Conv2D(
        hp.Int('conv4_filters', min_value=32, max_value=128, step=32),
        (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(hp.Float('dropout', 0.3, 0.7, step=0.1))(x)
    x = Dense(
        hp.Int('dense_units', min_value=64, max_value=256, step=64),
        activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    output_layer = Dense(10, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='mnist_tuner',
    project_name='mnist_cnn'
)

early_stop = EarlyStopping(monitor='val_loss', patience=3)

tuner.search(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stop])

best_model = tuner.get_best_models(num_models=1)[0]
best_model.save("mnist_classifier_best.keras")
