import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers 

def residual_layer(input, filters, filter_size):
    ident = layers.Conv2D(filters, 1, padding='same')(input)    
    res_layer = convolutional_layer(input, filters, filter_size)
    res_layer = layers.Conv2D(filters, filter_size, padding='same')(res_layer)
    res_layer = layers.BatchNormalization()(res_layer)
    res_layer = layers.add([ident,res_layer])
    res_layer = layers.Activation('relu')(res_layer)
    return res_layer

def convolutional_layer(input, filters, filter_size):
    conv_layer = layers.Conv2D(filters, filter_size, padding='same')(input)
    conv_layer = layers.BatchNormalization()(conv_layer)
    conv_layer = layers.Activation('relu')(conv_layer)
    return conv_layer

def get_model():
    res_layers = 11
    filters = 64
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
dynamicBatch = False
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

model = get_model()
train(model,'LK_ResGo_v1',100,64)
    
