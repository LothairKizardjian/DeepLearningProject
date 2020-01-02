import tensorflow.keras as keras
import numpy as np
import golois
import math

def get_data(planes,moves,batch_size):
    input_data = np.random.randint(2,size=(batch_size,19,19,planes))
    input_data = input_data.astype("float32")

    policy = np.random.randint(moves, size=(batch_size,))
    policy = keras.utils.to_categorical(policy)

    value = np.random.randint(2, size=(batch_size,))
    value = value.astype("float32")

    end = np.random.randint(2, size=(batch_size, 19, 19, 2))
    end = end.astype("float32")

    golois.getBatch(input_data, policy, value, end)
    return input_data, policy, value

class DataSequence(keras.utils.Sequence):
    def __init__(self, data, policy, value, batch_size):
        self.data = data
        self.policy = policy
        self.value = value
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, idx):
        data = np.array(self.data[idx * self.batch_size : (idx+1) * self.batch_size])
        policy = np.array(self.policy[idx * self.batch_size : (idx+1) * self.batch_size])
        value = np.array(self.value[idx * self.batch_size : (idx+1) * self.batch_size])

        return data, [policy, value]

    def on_epoch_end(self):
        self.data, self.policy, self.value = get_data(
            planes= 8,
            moves = 361,
            batch_size = self.batch_size)

        
    
