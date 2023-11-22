from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
from keras.models import Model


def unet(filters=8,
                img_size=(256, 256, 3),
                dropout_rate=0.1,
                kernel_size=(3, 3),
                pool_size=(2, 2),
                strides=(2, 2)):
    """
    Creates a U-Net model for image semantic segmentation.

    Args:
        filters (int, optional): The number of filters in the first layer, 8 by default.
        img_size (tuple, optional): The input image size, (256, 256, 3) by default.
        dropout_rate (float, optional): The dropout rate, 0.1 by default.
        kernel_size (tuple, optional): The kernel size for convolutional layers, (3, 3) by default.
        pool_size (tuple, optional): The pool size for max pooling layers, (2, 2) by default.
        strides (tuple, optional): The strides for transpose convolutional layers, (2, 2) by default.

    Returns:
        keras.models.Model: The implemented U-Net model.
    """
    inputs = Input(img_size)

    # Contraction path
    c1 = Conv2D(filters, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(dropout_rate)(c1)
    c1 = Conv2D(filters, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D(pool_size)(c1)

    c2 = Conv2D(filters * 2, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(dropout_rate)(c2)
    c2 = Conv2D(filters * 2, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D(pool_size)(c2)

    c3 = Conv2D(filters * 4, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(dropout_rate)(c3)
    c3 = Conv2D(filters * 4, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D(pool_size)(c3)

    c4 = Conv2D(filters * 8, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(dropout_rate)(c4)
    c4 = Conv2D(filters * 8, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size)(c4)

    # Bridge (1024)
    c5 = Conv2D(filters * 16, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(dropout_rate)(c5)
    c5 = Conv2D(filters * 16, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path
    u6 = Conv2DTranspose(filters * 8, pool_size, strides=strides, padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(filters * 8, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(dropout_rate)(c6)
    c6 = Conv2D(filters * 8, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(filters * 4, pool_size, strides=strides, padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(filters * 4, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(dropout_rate)(c7)
    c7 = Conv2D(filters * 4, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(filters * 2, pool_size, strides=strides, padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(filters * 2, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(dropout_rate)(c8)
    c8 = Conv2D(filters * 2, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(filters, pool_size, strides=strides, padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(filters, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(dropout_rate)(c9)
    c9 = Conv2D(filters, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(c9)

    model = Model(inputs=[inputs], outputs=[outputs], name='unet')
    print(model.summary())

    return model