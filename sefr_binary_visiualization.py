import numpy as np

class SEFR:
    """
    This is the original binary classifier proposed in
    'SEFR: A Fast Linear-Time Classifier for Ultra-Low Power Devices'
    (https://arxiv.org/abs/2006.04620)
    with visiualization.
    """

    def fit(self, data_train, target_train):
        """
        This is used for training the classifier on data.
        Parameters
        ----------
        data_train : float, either list or numpy array
            are the main data in DataFrame
        target_train : integer, numpy array
            labels, should consist of 0s and 1s
        """
        
        self.weights = []
        self.bias = 0

        if isinstance(data_train, list):
            data_train = np.array(data_train, dtype='float32')
            
        if isinstance(target_train, list):
            target_train = np.array(target_train, dtype='int32')
        
        # pos_labels are those records where the label is positive
        # neg_labels are those records where the label is negative
        pos_labels = (target_train > 0)   
        neg_labels = np.invert(pos_labels)

        # pos_indices are the data where the labels are positive
        # neg_indices are the data where the labels are negative
        pos_indices = data_train[pos_labels] 
        neg_indices = data_train[neg_labels]
        
        # avg_pos is the average value of each feature where the label is positive
        # avg_neg is the average value of each feature where the label is negative
        avg_pos = np.mean(pos_indices, axis=0)  # Eq. 3
        avg_neg = np.mean(neg_indices, axis=0)  # Eq. 4
        
        # weights are calculated based on Eq. 3 and Eq. 4
        self.weights = (avg_pos - avg_neg) / (avg_pos + avg_neg)  # Eq. 5
        
        # For each record, a score is calculated. If the record is positive/negative, the score will be added to posscore/negscore
        weighted_scores = np.dot(data_train, self.weights)  # Eq. 6
        
        # pos_score_avg and neg_score_avg are average values of records scores for positive and negative classes
        pos_score_avg = np.mean(weighted_scores[pos_labels])  # Eq. 7
        neg_score_avg = np.mean(weighted_scores[neg_labels])  # Eq. 8
        
        pos_label_count = pos_indices.size
        neg_label_count = neg_labels.size
        
        # bias is calculated using a weighted average
        self.bias = -(neg_label_count * pos_score_avg + pos_label_count * neg_score_avg) / (neg_label_count + pos_label_count)
    

    def predict(self, data_test):
        """
        This is for prediction. When the model is trained, it can be applied on the test data.
        Parameters
        ----------
        data_test: either list or ndarray, two dimensional
            the data without labels in
        Returns
        ----------
        predictions in numpy array
        """
        
        if isinstance(data_test, list):
            data_test = np.array(data_test, dtype='float32')

        weighted_score = np.dot(data_test, self.weights)
        
        preds = np.where(weighted_score + self.bias > 0, 1, 0)
        
        return preds


# ============================================================


from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# generate random data with 2 features
data, target = make_blobs(n_samples=5000, n_features=2, centers=2, random_state=0)
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)

# model training and predict
sefr = SEFR()
sefr.fit(data_train, target_train)
predictions = sefr.predict(data_test)

print('Accuracy:', accuracy_score(target_test, predictions))
print(classification_report(target_test, predictions))

plt.rcParams['font.size'] = 14
plt.figure(figsize=(16, 8))


# draw test data
plt.subplot(121)
plt.title('Test Data')
plt.scatter(data_test.T[0], data_test.T[1], c=target_test, cmap=plt.cm.Paired)

# draw prediction
plt.subplot(122)
plt.title('SEFR predictions')
plt.scatter(data_test.T[0], data_test.T[1], c=predictions, cmap=plt.cm.Paired)

# draw hyperplane
x1 = np.linspace(data_test.T[0].min(), data_test.T[0].max(), 2)
x2 = (-sefr.bias - sefr.weights[0] * x1) / sefr.weights[1]  # x2=(-b-w1x1)/w2
plt.plot(x1, x2, color='green')

# visiualization
plt.tight_layout()
plt.show()