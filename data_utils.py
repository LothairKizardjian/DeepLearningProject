import numpy as np
import golois

def generate_data(games,planes,moves,batch_size):
    """ Generates random batches of data with the CPP function get_batch_data of golois
    
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

    
