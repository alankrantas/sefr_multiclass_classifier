"""
This is the multiclass classifier version of the SEFR algorithm for MicroPython (ESP8266)
based on my Arduino C++ version.
With some modification, this can also be used as the Python 3.4 version without using NumPy.
"""

from array import array
import math, random, time, gc


# the iris dataset
data   = ((5.1,3.5,1.4,0.2),(4.9,3.0,1.4,0.2),(4.7,3.2,1.3,0.2),(4.6,3.1,1.5,0.2),(5.0,3.6,1.4,0.2),(5.4,3.9,1.7,0.4),(4.6,3.4,1.4,0.3),(5.0,3.4,1.5,0.2),(4.4,2.9,1.4,0.2),(4.9,3.1,1.5,0.1),(5.4,3.7,1.5,0.2),(4.8,3.4,1.6,0.2),(4.8,3.0,1.4,0.1),(4.3,3.0,1.1,0.1),(5.8,4.0,1.2,0.2),(5.7,4.4,1.5,0.4),(5.4,3.9,1.3,0.4),(5.1,3.5,1.4,0.3),(5.7,3.8,1.7,0.3),(5.1,3.8,1.5,0.3),(5.4,3.4,1.7,0.2),(5.1,3.7,1.5,0.4),(4.6,3.6,1.0,0.2),(5.1,3.3,1.7,0.5),(4.8,3.4,1.9,0.2),(5.0,3.0,1.6,0.2),(5.0,3.4,1.6,0.4),(5.2,3.5,1.5,0.2),(5.2,3.4,1.4,0.2),(4.7,3.2,1.6,0.2),(4.8,3.1,1.6,0.2),(5.4,3.4,1.5,0.4),(5.2,4.1,1.5,0.1),(5.5,4.2,1.4,0.2),(4.9,3.1,1.5,0.2),(5.0,3.2,1.2,0.2),(5.5,3.5,1.3,0.2),(4.9,3.6,1.4,0.1),(4.4,3.0,1.3,0.2),(5.1,3.4,1.5,0.2),(5.0,3.5,1.3,0.3),(4.5,2.3,1.3,0.3),(4.4,3.2,1.3,0.2),(5.0,3.5,1.6,0.6),(5.1,3.8,1.9,0.4),(4.8,3.0,1.4,0.3),(5.1,3.8,1.6,0.2),(4.6,3.2,1.4,0.2),(5.3,3.7,1.5,0.2),(5.0,3.3,1.4,0.2),(7.0,3.2,4.7,1.4),(6.4,3.2,4.5,1.5),(6.9,3.1,4.9,1.5),(5.5,2.3,4.0,1.3),(6.5,2.8,4.6,1.5),(5.7,2.8,4.5,1.3),(6.3,3.3,4.7,1.6),(4.9,2.4,3.3,1.0),(6.6,2.9,4.6,1.3),(5.2,2.7,3.9,1.4),(5.0,2.0,3.5,1.0),(5.9,3.0,4.2,1.5),(6.0,2.2,4.0,1.0),(6.1,2.9,4.7,1.4),(5.6,2.9,3.6,1.3),(6.7,3.1,4.4,1.4),(5.6,3.0,4.5,1.5),(5.8,2.7,4.1,1.0),(6.2,2.2,4.5,1.5),(5.6,2.5,3.9,1.1),(5.9,3.2,4.8,1.8),(6.1,2.8,4.0,1.3),(6.3,2.5,4.9,1.5),(6.1,2.8,4.7,1.2),(6.4,2.9,4.3,1.3),(6.6,3.0,4.4,1.4),(6.8,2.8,4.8,1.4),(6.7,3.0,5.0,1.7),(6.0,2.9,4.5,1.5),(5.7,2.6,3.5,1.0),(5.5,2.4,3.8,1.1),(5.5,2.4,3.7,1.0),(5.8,2.7,3.9,1.2),(6.0,2.7,5.1,1.6),(5.4,3.0,4.5,1.5),(6.0,3.4,4.5,1.6),(6.7,3.1,4.7,1.5),(6.3,2.3,4.4,1.3),(5.6,3.0,4.1,1.3),(5.5,2.5,4.0,1.3),(5.5,2.6,4.4,1.2),(6.1,3.0,4.6,1.4),(5.8,2.6,4.0,1.2),(5.0,2.3,3.3,1.0),(5.6,2.7,4.2,1.3),(5.7,3.0,4.2,1.2),(5.7,2.9,4.2,1.3),(6.2,2.9,4.3,1.3),(5.1,2.5,3.0,1.1),(5.7,2.8,4.1,1.3),(6.3,3.3,6.0,2.5),(5.8,2.7,5.1,1.9),(7.1,3.0,5.9,2.1),(6.3,2.9,5.6,1.8),(6.5,3.0,5.8,2.2),(7.6,3.0,6.6,2.1),(4.9,2.5,4.5,1.7),(7.3,2.9,6.3,1.8),(6.7,2.5,5.8,1.8),(7.2,3.6,6.1,2.5),(6.5,3.2,5.1,2.0),(6.4,2.7,5.3,1.9),(6.8,3.0,5.5,2.1),(5.7,2.5,5.0,2.0),(5.8,2.8,5.1,2.4),(6.4,3.2,5.3,2.3),(6.5,3.0,5.5,1.8),(7.7,3.8,6.7,2.2),(7.7,2.6,6.9,2.3),(6.0,2.2,5.0,1.5),(6.9,3.2,5.7,2.3),(5.6,2.8,4.9,2.0),(7.7,2.8,6.7,2.0),(6.3,2.7,4.9,1.8),(6.7,3.3,5.7,2.1),(7.2,3.2,6.0,1.8),(6.2,2.8,4.8,1.8),(6.1,3.0,4.9,1.8),(6.4,2.8,5.6,2.1),(7.2,3.0,5.8,1.6),(7.4,2.8,6.1,1.9),(7.9,3.8,6.4,2.0),(6.4,2.8,5.6,2.2),(6.3,2.8,5.1,1.5),(6.1,2.6,5.6,1.4),(7.7,3.0,6.1,2.3),(6.3,3.4,5.6,2.4),(6.4,3.1,5.5,1.8),(6.0,3.0,4.8,1.8),(6.9,3.1,5.4,2.1),(6.7,3.1,5.6,2.4),(6.9,3.1,5.1,2.3),(5.8,2.7,5.1,1.9),(6.8,3.2,5.9,2.3),(6.7,3.3,5.7,2.5),(6.7,3.0,5.2,2.3),(6.3,2.5,5.0,1.9),(6.5,3.0,5.2,2.0),(6.2,3.4,5.4,2.3),(5.9,3.0,5.1,1.8))
target = '000000000000000000000000000000000000000000000000001111111111111111111111111111111111111111111111111122222222222222222222222222222222222222222222222222'

feature_num = len(data[0])                 # number of features
labels = tuple(sorted(tuple(set(target)))) # labels

weights = []          # model weights
bias = array('f', []) # model bias
training_time = 0     # model training time


# ================================================================================


def fit():
    """
    Train the model with dataset.
    """
    
    global weights, bias, training_time

    start_time = time.ticks_ms() # change it to time.time() on standard Python

    for _, l in enumerate(labels):
        
        weights.append(array('f', []))
        
        count_neg = str.count(target, str(l))
        count_pos = len(target)  - count_neg
        
        for f in range(feature_num):
            
            list_pos = [x[f] for x, y in zip(data, target) if y != l]
            avg_pos = sum(list_pos) / count_pos
            
            list_neg = [x[f] for x, y in zip(data, target) if y == l]
            avg_neg = sum(list_neg) / count_neg
            
            # calculate label weights
            weights[-1].append((avg_pos - avg_neg) / (avg_pos + avg_neg))


        weighted_score = array('f', [])
        
        for d in data:
            weighted_score.append(sum(map(lambda x, y: x * y, weights[-1], d)))


        list_pos_w = [x for x, y in zip(weighted_score, target) if y != l]
        avg_pos_w = sum(list_pos_w) / count_pos

        list_neg_w = [x for x, y in zip(weighted_score, target) if y == l]
        avg_neg_w = sum(list_neg_w) / count_neg
        
        # calculate label bias
        bias.append(-(count_neg * avg_pos_w + count_pos * avg_neg_w) / (count_pos + count_neg))
        
        
    training_time = time.ticks_ms() - start_time  # change it to time.time() on standard Python


def predict(new_data):
    """
    Predict label for a single new data instance.
    """
    
    score = []
    for i, _ in enumerate(labels):
        score.append(sum(map(lambda x, y: x * y, weights[i], new_data)) + bias[i])
    
    # return predicted label
    return labels[score.index(min(score))]


# ================================================================================

random.seed(42)
gc.enable()
gc.collect()

fit() # train model


while True:
    
    gc.collect()
    
    # select a random data instance and randomly +- 0~30% for each features
    
    index = -1
    while index < 0 or index >= len(target):
        index = random.getrandbits(int(math.log(len(data)) / math.log(2)))
    
    test_data = tuple(map(
        lambda x: x + (x * (random.getrandbits(3) / 10) * (1 if random.getrandbits(1) == 0 else -1)),
        data[index]))
    
    # predict label
    prediction = predict(test_data)
    
    print('Test data:', test_data)
    print('Predicted label: {} / actual label: {} / SEFR training time: {} ms\n'.format(
        prediction, target[index], training_time))

    time.sleep(1)
