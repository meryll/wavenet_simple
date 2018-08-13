import numpy as np

parameters = 256

def one_hot(x):
    return np.eye(parameters, dtype='uint8')[x.astype('uint8')]

def inverse_one_hot(data):

    inversed = np.zeros([data.shape[0], data.shape[1]])
    i = 0
    j = 0

    for row in data:
        for column in row:
            inversed[i,j] = _get_index(column)
            j+=1
        j = 0
        i+=1

    return inversed


def _get_index(row):
    for i in range(len(row)):
        if row[i] == 1.0:
            return i

# if __name__ == '__main__':
#     array = [[2,3],[1,4]]
#     array = np.array(array)
#     print(array.shape, array)
#     endo = one_hot(np.array(array))
#     print(endo.shape)
#     print(endo)
#
#     inv = inverse_one_hot_encoding(endo)
#     print("inversed")
#     print(inv)