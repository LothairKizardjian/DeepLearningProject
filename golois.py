import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.regularizers as regularizers
import numpy as np
import os
from tensorflow.keras import layers 

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

def residual_layer(input, filters, filter_size):    
    res_layer = convolutional_layer(input, filters, filter_size)
    res_layer = layers.Conv2D(filters, filter_size, padding='same')(res_layer)
    res_layer = layers.BatchNormalization()(res_layer)
    res_layer = layers.add([input,res_layer])
    res_layer = layers.Activation('relu')(res_layer)
    return res_layer

def convolutional_layer(input, filters, filter_size):
    conv_layer = layers.Conv2D(filters, filter_size, padding='same')(input)
    conv_layer = layers.BatchNormalization()(conv_layer)
    conv_layer = layers.Activation('relu')(conv_layer)
    return conv_layer

def get_model():
    res_layers = 4
    filters = 256
    filter_size = 3
    input = layers.Input(shape=(19, 19, planes), name='board')    
    for i in range(res_layers):
        if i==0:
            x = residual_layer(input, filters, filter_size)
        else:
            x = residual_layer(x, filters, filter_size)
        
    policy_head = layers.Conv2D(1, 3, activation='relu', padding='same')(x)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Dense(moves, activation='softmax', name='policy')(policy_head)
    
    value_head = layers.Flatten()(x)
    value_head = layers.Dense(1, activation='sigmoid', name='value')(value_head)

    model = keras.Model(inputs=input, outputs=[policy_head, value_head])
    model.summary ()

    return model

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
    policy = layers.Dropout(0.2)(policy)
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
    value = layers.Dropout(0.4)(value)
    value = layers.Dense(value_dense)(value)
    value = layers.Dense(1, activation="tanh", name="value")(value)

    model = keras.Model(inputs=input, outputs=[policy,value])
    model.summary()
    return model

def train(model,model_title,epochs,batch_size):
    model.compile(optimizer=keras.optimizers.SGD(lr=0.1),
                  loss={'value': 'mse', 'policy': 'categorical_crossentropy'})

    checkpointer  = keras.callbacks.ModelCheckpoint(filepath='./models/{}.h5'.format(model_title),
                                                    verbose=1,
                                                    save_best_only=True)
    
    model.fit(input_data, {'policy': policy, 'value': value},
              epochs=epochs, batch_size=batch_size, validation_split=0.1,callbacks=[checkpointer])

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
    
