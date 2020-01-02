import tensorflow.keras as keras
import numpy as np
import golois
import math

def get_data(games,planes,moves,batch_size):
    """ Generates random batches of data with the CPP function getBatch of golois
    
    Args:
    games : data of the games
    planes : number of planes of the tensor
    moves : total number of moves you can do on a 19X19 goban
    batch_size : number of tensors to generate

    Returns:
    inputs : the (batch_size, 19, 19, planes) tensors
    policy : the policies corresponding to those inputs
    value : the values corresponding to those inputs
    """

    input = np.random.randint(2,size=(batch_size,19,19,planes))
    input = input.astype("float32")

    policy = np.random.randint(moves, size=(batch_size,))
    policy = keras.to_categorical(policy)

    value = np.random.randint(2, size=(N,))
    value = value.astype("float32")

    end = np.random.randint(2, size=(N, 19, 19, 2))
    end = end.astype("float32")

    golois.getBatch(games, input, policy, value, end)
    return input, policy, value

class DataSequence(keras.utils.Sequence):
    def __init__(self, data, policy, value, batch_size):
        self.data = data
        self.policy = policy
        self.value = value
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        data = np.array(self.data[idx * self.batch_size : (idx+1) * self.batch_size])
        policy = np.array(self.policy[idx * self.batch_size : (idx+1) * self.batch_size])
        value = np.array(self.value[idx * self.batch_size : (idx+1) * self.batch_size])

        return data, policy, value

    def on_epoch_end(self):
        self.data, self.policy, self.value = get_batch_data(
            games = "./data/games.data",
            planes= 8,
            moves = 361,
            batch_size = 137072)

        
    
