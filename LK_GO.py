import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.regularizers as regularizers
import numpy as np
import os
import data_utils
from tensorflow.keras import layers 

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def conv_layer(x_input,filters):
    x = layers.Conv2D(
        filters=filters,
        kernel_size=(3,3),
        strides=(1,1),
        padding="same",
        activation=None,
        kernel_regularizer=regularizers.l2(1e-4))(x_input)
    x = layers.BatchNormalization()(x)
    return x

def residual_layer(x_input,filters):
    copy = x_input

    x = conv_layer(x_input,filters)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)
    
    x = conv_layer(x_input,filters)
    x = layers.Add()([x, copy])
    x = layers.Activation("relu")(x)
    return x

def policy_head(x_input,moves):
    policy = layers.Conv2D(
        filters=2,
        kernel_size=(1,1),
        strides=(1,1),
        padding="same",
        activation=None,
        kernel_regularizer=regularizers.l2(1e-4))(x_input)
    policy = layers.BatchNormalization()(policy)
    policy = layers.Activation("relu")(policy)
    policy = layers.Flatten()(policy)
    policy = layers.Dense(moves, activation="softmax", name="policy")(policy)
    return policy

def value_head(x_input,dense_size):
    value = layers.Conv2D(
        filters=1,
        kernel_size=(1, 1),
        strides=(1,1),
        padding="same",
        activation=None,
        kernel_regularizer=regularizers.l2(1e-4))(x_input)
    value = layers.BatchNormalization()(value)
    value = layers.Activation("relu")(value)
    value = layers.Flatten()(value)
    value = layers.Dense(dense_size)(value)
    value = layers.Dense(1, activation="tanh", name="value")(value)
    return value

def build_model(input_shape, moves, blocks, filters, dense_size):
    input = layers.Input(shape=input_shape, name="board")
    
    x = layers.Conv2D(
        filters=filters,
        kernel_size=(3,3),
        strides=(1,1),
        padding="same",
        activation=None,
        kernel_regularizer=regularizers.l2(1e-4))(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    for i in range(blocks):
        x = residual_layer(x,filters)

    policy = policy_head(x,moves)
    value = value_head(x,dense_size)
    
    model = keras.Model(inputs=input, outputs=[policy,value])
    model.summary()
    return model

def train(model,model_title,epochs,batch_size):
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
        #optimizer=tf.keras.optimizers.AdaDelta(learning_rate=1),
        loss={'value': 'mse', 'policy': 'categorical_crossentropy'},
        metrics=['accuracy'],
        loss_weights=[1,1]
    )

    checkpointer  = keras.callbacks.ModelCheckpoint(
        filepath='./models/{}.h5'.format(model_title),
        verbose=1)

    model.fit(
        train_input_data,
        {'policy' : train_policy, 'value' : train_value},
        epochs = epochs,
        verbose = 1,
        callbacks=[checkpointer])
    
    """
    model.fit_generator(
        generator=data_utils.DataSequence(
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
    """
planes = 8
moves = 361
batch_size = 100000
datapasses = 20

print("Building model ...")
model = build_model((19,19,planes),moves,4,139,64)
print("Model Built.")

for i in range(datapasses):

    print("Loading train data ...")
    
    train_input_data, train_policy, train_value = data_utils.get_data(
        planes = planes,
        moves = moves,
        batch_size = batch_size)
    
    print("Train data loaded.")
    print("Training model ...")
    
    train(model,'LK_ResGo_v9',100,256)

    print("Model trained.")

