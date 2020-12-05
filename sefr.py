import numpy as np, time

class SEFR:
    """
    This is the multiclass classifier version of the SEFR algorithm for Python
    based on https://github.com/sefr-classifier/sefr/blob/master/SEFR.py
    
    Also see: https://arxiv.org/abs/2006.04620
    """
    
    def __init__(self):
        """
        Initialize model class.
        """
        
        self.labels = np.array([])
        self.weights = np.array([])
        self.bias = np.array([])
        self.training_time = 0


    def fit(self, data_train, target_train):
        """
        Train the model.
        """
        
        self.labels = np.unique(target_train) # get all labels
        self.weights = []
        self.bias = []
        self.training_time = 0
        
        start_time = time.monotonic_ns()
        
        if isinstance(data_train, list):
            data_train = np.array(data_train, dtype='float32')
        
        if isinstance(target_train, list):
            target_train = np.array(target_train, dtype='int32')
        
        for label in self.labels: # train binary classifiers on each labels
            
            pos_labels = (target_train != label) # use "not the label" as positive class
            neg_labels = np.invert(pos_labels) # use the label as negative class
            
            pos_indices = data_train[pos_labels]
            neg_indices = data_train[neg_labels]
            
            avg_pos = np.mean(pos_indices, axis=0)
            avg_neg = np.mean(neg_indices, axis=0)
            
            weight = np.nan_to_num((avg_pos - avg_neg) / (avg_pos + avg_neg)) # calculate model weight of "not the label"
            weighted_scores = np.dot(data_train, weight)
            
            pos_score_avg = np.mean(weighted_scores[pos_labels])
            neg_score_avg = np.mean(weighted_scores[neg_labels])
            
            bias = -(neg_labels.size * pos_score_avg + # calculate weighted average of bias
                     pos_indices.size * neg_score_avg) / (neg_labels.size + pos_indices.size)
            
            self.weights.append(weight) # label weight
            self.bias.append(bias) # label bias
        
        self.weights = np.array(self.weights, dtype='float32')
        self.bias = np.array(self.bias, dtype='float32')
        self.training_time = time.monotonic_ns() - start_time


    def predict(self, new_data):
        """
        Predict labels of the new data.
        """
        
        if isinstance(new_data, list):
            new_data = np.array(new_data, dtype='float32')

        # calculate weighted score + bias on each labels
        weighted_score = np.add(np.dot(self.weights, new_data.T).T, self.bias)
        return self.labels[np.argmin(weighted_score, axis=1)]


    def get_params(self, deep=True): # for cross-validation
        return {}


# ================================================================================

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report

# load dataset
data, target = datasets.load_iris(return_X_y=True)

# prepare training and test dataset
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=0)

# train model and predict labels
sefr = SEFR()
sefr.fit(data_train, target_train)
predictions = sefr.predict(data_test)
cv_predictions = cross_val_predict(sefr, data_train, target_train, cv=5)

# view prediction results
print('Training time:', sefr.training_time, 'ns')
print('Training dataset prediction CV score:', accuracy_score(target_train, cv_predictions).round(3))
print('Test dataset prediction accuracy:', accuracy_score(target_test, predictions).round(3))
print('Test dataset classification report:')
print(classification_report(target_test, predictions))
