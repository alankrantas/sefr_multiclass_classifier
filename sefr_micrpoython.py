"""
This is the multiclass classifier version of the SEFR algorithm for MicroPython (ESP8266/ESP32)
based on my Arduino C++ version.
With some modification, this can also be used as the Python 3.4 version without using NumPy.
"""

from machine import freq
import math, random, time, gc

gc.enable()

try:
    freq(240000000) # ESP32
except:
    freq(160000000) # ESP8266


# the iris dataset
data_factor = 10
data   = [[51, 35, 14, 2], [49, 30, 14, 2], [47, 32, 13, 2], [46, 31, 15, 2], [50, 36, 14, 2], [54, 39, 17, 4], [46, 34, 14, 3], [50, 34, 15, 2], [44, 29, 14, 2], [49, 31, 15, 1], [54, 37, 15, 2], [48, 34, 16, 2], [48, 30, 14, 1], [43, 30, 11, 1], [58, 40, 12, 2], [57, 44, 15, 4], [54, 39, 13, 4], [51, 35, 14, 3], [57, 38, 17, 3], [51, 38, 15, 3], [54, 34, 17, 2], [51, 37, 15, 4], [46, 36, 10, 2], [51, 33, 17, 5], [48, 34, 19, 2], [50, 30, 16, 2], [50, 34, 16, 4], [52, 35, 15, 2], [52, 34, 14, 2], [47, 32, 16, 2], [48, 31, 16, 2], [54, 34, 15, 4], [52, 41, 15, 1], [55, 42, 14, 2], [49, 31, 15, 2], [50, 32, 12, 2], [55, 35, 13, 2], [49, 36, 14, 1], [44, 30, 13, 2], [51, 34, 15, 2], [50, 35, 13, 3], [45, 23, 13, 3], [44, 32, 13, 2], [50, 35, 16, 6], [51, 38, 19, 4], [48, 30, 14, 3], [51, 38, 16, 2], [46, 32, 14, 2], [53, 37, 15, 2], [50, 33, 14, 2], [70, 32, 47, 14], [64, 32, 45, 15], [69, 31, 49, 15], [55, 23, 40, 13], [65, 28, 46, 15], [57, 28, 45, 13], [63, 33, 47, 16], [49, 24, 33, 10], [66, 29, 46, 13], [52, 27, 39, 14], [50, 20, 35, 10], [59, 30, 42, 15], [60, 22, 40, 10], [61, 29, 47, 14], [56, 29, 36, 13], [67, 31, 44, 14], [56, 30, 45, 15], [58, 27, 41, 10], [62, 22, 45, 15], [56, 25, 39, 11], [59, 32, 48, 18], [61, 28, 40, 13], [63, 25, 49, 15], [61, 28, 47, 12], [64, 29, 43, 13], [66, 30, 44, 14], [68, 28, 48, 14], [67, 30, 50, 17], [60, 29, 45, 15], [57, 26, 35, 10], [55, 24, 38, 11], [55, 24, 37, 10], [58, 27, 39, 12], [60, 27, 51, 16], [54, 30, 45, 15], [60, 34, 45, 16], [67, 31, 47, 15], [63, 23, 44, 13], [56, 30, 41, 13], [55, 25, 40, 13], [55, 26, 44, 12], [61, 30, 46, 14], [58, 26, 40, 12], [50, 23, 33, 10], [56, 27, 42, 13], [57, 30, 42, 12], [57, 29, 42, 13], [62, 29, 43, 13], [51, 25, 30, 11], [57, 28, 41, 13], [63, 33, 60, 25], [58, 27, 51, 19], [71, 30, 59, 21], [63, 29, 56, 18], [65, 30, 58, 22], [76, 30, 66, 21], [49, 25, 45, 17], [73, 29, 63, 18], [67, 25, 58, 18], [72, 36, 61, 25], [65, 32, 51, 20], [64, 27, 53, 19], [68, 30, 55, 21], [57, 25, 50, 20], [58, 28, 51, 24], [64, 32, 53, 23], [65, 30, 55, 18], [77, 38, 67, 22], [77, 26, 69, 23], [60, 22, 50, 15], [69, 32, 57, 23], [56, 28, 49, 20], [77, 28, 67, 20], [63, 27, 49, 18], [67, 33, 57, 21], [72, 32, 60, 18], [62, 28, 48, 18], [61, 30, 49, 18], [64, 28, 56, 21], [72, 30, 58, 16], [74, 28, 61, 19], [79, 38, 64, 20], [64, 28, 56, 22], [63, 28, 51, 15], [61, 26, 56, 14], [77, 30, 61, 23], [63, 34, 56, 24], [64, 31, 55, 18], [60, 30, 48, 18], [69, 31, 54, 21], [67, 31, 56, 24], [69, 31, 51, 23], [58, 27, 51, 19], [68, 32, 59, 23], [67, 33, 57, 25], [67, 30, 52, 23], [63, 25, 50, 19], [65, 30, 52, 20], [62, 34, 54, 23], [59, 30, 51, 18]]
target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

feature_num = len(data[0])   # number of features
labels = sorted(set(target)) # labels

weights = []      # model weights
bias    = []      # model bias
training_time = 0 # model training time


# ================================================================================

def fit():
    """
    Train the model with dataset.
    """
    
    gc.collect()
    
    global weights, bias, training_time
    
    weights = []
    bias = []
    training_time = 0

    start_time = time.ticks_ms()

    for l in labels:
        
        weights.append([])
        
        count_neg = target.count(l)
        count_pos = len(target)  - count_neg
        
        for f in range(feature_num):
            
            list_pos = [x[f] for x, y in zip(data, target) if y != l]
            avg_pos = sum(list_pos) / data_factor / count_pos
            
            list_neg = [x[f] for x, y in zip(data, target) if y == l]
            avg_neg = sum(list_neg) / data_factor / count_neg
            
            # calculate label weights
            weights[-1].append((avg_pos - avg_neg) / (avg_pos + avg_neg))

        weighted_score = []
        
        for d in data:
            weighted_score.append(sum([x * y for x, y in zip(weights[-1], d)]))

        list_pos_w = [x for x, y in zip(weighted_score, target) if y != l]
        avg_pos_w = sum(list_pos_w) / data_factor / count_pos

        list_neg_w = [x for x, y in zip(weighted_score, target) if y == l]
        avg_neg_w = sum(list_neg_w) / data_factor / count_neg
        
        # calculate label bias
        bias.append(-(count_neg * avg_pos_w + count_pos * avg_neg_w) / (count_pos + count_neg))
        
        
    training_time = time.ticks_ms() - start_time


def predict(new_data):
    """
    Predict label for a single new data instance.
    """
    
    gc.collect()
    
    score = []
    for i in range(len(labels)):
        score.append(sum([x / data_factor * y for x, y in zip(weights[i], new_data)]) + bias[i])
    
    # return predicted label
    return labels[score.index(min(score))]


# ================================================================================

random.seed(42)

fit() # train model


while True:
    
    # select a random data instance and randomly +- 0~30% for each features
    
    index = -1
    while index < 0 or index >= len(target):
        index = random.getrandbits(int(math.log(len(data)) / math.log(2)))
    
    test_data = list(map(
        lambda x: x + (x * (random.getrandbits(2) / 10) * (1 if random.getrandbits(1) == 0 else -1)),
        data[index]))
    
    # predict label
    prediction = predict(test_data)
    
    print('Test data:', list(map(lambda n: n / data_factor, test_data)))
    print('Predicted label: {} / actual label: {} / SEFR training time: {} ms\n'.format(
        prediction, target[index], training_time))

    time.sleep(1)
