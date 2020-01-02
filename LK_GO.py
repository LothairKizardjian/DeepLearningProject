import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.regularizers as regularizers
import numpy as np
import os
import data_utils
from tensorflow.keras import layers 

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

def build_model(input_shape, moves, blocks, filters, value_dense):
    input = layers.Input(shape=input_shape, name="board")
    x = layers.Conv2D(
        filters=filters,
        kernel_size=(3,3),
        strides=(1,1),
        padding="same",
        activation=None,
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4))(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    for i in range(blocks):
        copy = x
        x = layers.Conv2D(
            filters=filters,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same",
            activation=None,
            use_bias=False,
            kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(filters=filters,
                          kernel_size=(3,3),
                          strides=(1,1),
                          padding="same",
                          activation=None,
                          use_bias=False,
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, copy])
        x = layers.Activation("relu")(x)

    policy = layers.Conv2D(
        filters=2,
        kernel_size=(1,1),
        strides=(1,1),
        padding="same",
        activation=None,
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4))(x)
    policy = layers.BatchNormalization()(policy)
    policy = layers.Activation("relu")(policy)
    policy = layers.Flatten()(policy)
    policy = layers.Dense(moves, activation="softmax", name="policy")(policy)

    value = layers.Conv2D(
        filters=1,
        kernel_size=(1, 1),
        strides=(1,1),
        padding="same",
        activation=None,
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4))(x)
    value = layers.BatchNormalization()(value)
    value = layers.Activation("relu")(value)
    value = layers.Flatten()(value)
    value = layers.Dense(value_dense)(value)
    value = layers.Dense(1, activation="tanh", name="value")(value)

    model = keras.Model(inputs=input, outputs=[policy,value])
    model.summary()
    return model

def train(model,model_title,epochs,batch_size):
    model.compile(
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=1),
        loss={'value': 'mse', 'policy': 'categorical_crossentropy'},
        metrics=['accuracy']
    )

    checkpointer  = keras.callbacks.ModelCheckpoint(
        filepath='./models/{}.h5'.format(model_title),
        verbose=1,
        save_best_only=True)
    
    model.fit_generator(
        generator=data_utils.Datasequence(
            train_input_data,
            train_policy,
            train_value,
            batch_size=batch_size),
        validation_data=(
            test_input_data,
            {'policy': test_policy, 'value': test_value}),
        epochs=epochs,
        verbose=1,
        callbacks=[checkpointer])

planes = 8
moves = 361
dynamicBatch = True

if dynamicBatch:
    import golois
    N = 100000
    input_data = np.random.randint(2, size=(N, 19, 19, planes))
    input_data = input_data.astype ('float32')
    
    policy = np.random.randint(moves, size=(N,))
    policy = keras.utils.to_categorical (policy)
    
    value = np.random.randint(2, size=(N,))
    value = value.astype ('float32')
    
    end = np.random.randint(2, size=(N, 19, 19, 2))
    end = end.astype ('float32')

    golois.getBatch (input_data, policy, value, end)
else:
    input_data = np.load ('input_data.npy')
    policy = np.load ('policy.npy')
    value = np.load ('value.npy')
    end = np.load ('end.npy')
    
model = build_model((19,19,planes),361,10,62,64)
train(model,'LK_ResGo_v4',50,64)
    
