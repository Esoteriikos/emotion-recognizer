import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1, l2, l1_l2


def cnn_model_4():
    classes =7
    input_shape = (48, 48, 1)
    m1 = tf.keras.models.Sequential([
        Conv2D(8, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        # 46 46 8

        Conv2D(16, (3, 3), activation='relu', padding='same'),
        # 46 46 16
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        # 46 46 16
        BatchNormalization(),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        # 46 46 32
        Dropout(0.15),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        # 46 46 32
        MaxPooling2D(2, 2),
        # 23 23 32
        BatchNormalization(),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        # 23 23 64
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        # 23 23 64
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        # 23 23 128
        Dropout(0.2),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        # 23 23 128
        MaxPooling2D(2, 2),
        # 11 11 128
        BatchNormalization(),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        # 11 11 256
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        # 11 11 256
        BatchNormalization(),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        # 11 11 256
        Dropout(0.15),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        # 11 11 256
        AveragePooling2D(2, 2),
        # 5 5 256
        BatchNormalization(),

        Flatten(),
        # Dense(8192, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(classes, activation='softmax')

    ])
    # print(m1.summary())
    return m1


def cnn_model_2():
    classes = 7
    input_shape = (48, 48, 1)
    m1 = tf.keras.models.Sequential([
        Conv2D(8, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        # 46 46 8

        Conv2D(16, (3, 3), activation='relu', padding='same'),
        # 46 46 16
        Dropout(0.25),
        Conv2D(16, (3, 3), activation='relu'),
        # 44 44 32
        BatchNormalization(),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        # 44 44 64
        Dropout(0.25),
        Conv2D(32, (5, 5), activation='relu'),
        # 40 40 32
        MaxPooling2D(2, 2),
        # 20 20 32
        BatchNormalization(),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        # 20 20 64
        Dropout(0.3),
        Conv2D(64, (3, 3), activation='relu'),
        # 18 18 64
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        # 18 18 128
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        # 16 16 128
        MaxPooling2D(2, 2),
        # 8 8 128
        BatchNormalization(),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        # 8 8 256
        Dropout(0.35),
        Conv2D(256, (3, 3), activation='relu'),
        # 6 6 256
        AveragePooling2D(2, 2),
        # 3 3 256
        BatchNormalization(),

        Flatten(),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(classes, activation='softmax')

    ])
    # print(m1.summary())
    return m1


# if __name__ == "__main__":
#     b = cnn_model(7, (48, 48, 1))
