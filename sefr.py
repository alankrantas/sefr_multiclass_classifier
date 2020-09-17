import numpy as np

class SEFR:
    """
    This is the multiclass classifier version of the SEFR algorithm for Python
    based on https://github.com/sefr-classifier/sefr/blob/master/SEFR.py.
    
    Also see: https://arxiv.org/abs/2006.04620
    """
    
    def __init__(self):
        """
        Initialize model class.
        """
        
        self.weights = []
        self.bias = []


    def fit(self, data_train, target_train):
        """
        Train the model.
        """
        
        self.weights = []
        self.bias = []
        
        if isinstance(data_train, list):
            data_train = np.array(train_predictors, dtype='float32')
        
        if isinstance(target_train, list):
            target_train = np.array(train_target, dtype='int32')
        
        self.labels = np.unique(target_train) # get all labels
        
        for label in self.labels: # train binary classifiers on each labels
            
            pos_labels = (target_train != label) # use "not the label" as positive class
            neg_labels = np.invert(pos_labels) # use the label as negative class
            
            pos_indices = data_train[pos_labels]
            neg_indices = data_train[neg_labels]
            
            avg_pos = np.mean(pos_indices, axis=0)
            avg_neg = np.mean(neg_indices, axis=0)
            
            weight = (avg_pos - avg_neg) / (avg_pos + avg_neg) # calculate model weight of "not the label"
            
            weighted_scores = np.dot(data_train, weight)
            
            pos_score_avg = np.mean(weighted_scores[pos_labels])
            neg_score_avg = np.mean(weighted_scores[neg_labels])
            
            pos_label_count = pos_indices.size
            neg_label_count = neg_labels.size
            
            bias = -(neg_label_count * pos_score_avg + # calculate weighted average of bias
                     pos_label_count * neg_score_avg) / (neg_label_count + pos_label_count)
            
            self.weights.append(weight) # label weight
            self.bias.append(bias) # label bias


    def predict(self, new_data):
        """
        Predict labels of the new data.
        """
        
        probs = []
        self.preds = []
        
        if isinstance(new_data, list):
            new_data = np.array(new_data, dtype='float32')
        
        for i in self.labels: # calculate weighted score + bias of each labels
            probs.append(np.dot(new_data, self.weights[i]) + self.bias[i])
        
        probs = np.array(probs).T
        
        for prob in probs: # find the min score (least possible label of "not the label")
            self.preds.append(self.labels[np.argmin(prob)])
        
        self.preds = np.array(self.preds)
        
        return self.preds


# ================================================================================

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# load dataset
data, target = datasets.load_iris(return_X_y=True)

# prepare training and test dataset
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.2, random_state=0)

# train model and predict labels
sefr = SEFR()
sefr.fit(data_train, target_train)
predictions = sefr.predict(data_test)

# view prediction results
print('Predictions:', predictions)
print('True labels:', target_test)
print('Accuracy:', accuracy_score(target_test, predictions).round(3))
print(classification_report(target_test, predictions))
