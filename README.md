# SEFR: Fast Linear Multiclass Classifier (Machine Learning)

These code are based on [sefr-classifier/sefr](https://github.com/sefr-classifier/sefr) and [SEFR: A Fast Linear-Time Classifier for Ultra-Low Power Devices](https://arxiv.org/abs/2006.04620). The original version is a binary classifier, and I modified it to do multiple label classification (one-vs-all).

SEFR is indeed very fast. In the Python version (using the iris dataset), the training time (~500 nanosecond on my laptop) is only about 18% of KNN (K=3), 5.8% of Linear SVM (100 iterations) and 1.3% of logistic regression (100 iterations).

The Arduino C++ version has a built-in iris dataset ported from scikit-learn and is runnable on Arduino Uno, Arduino Micro as well as various microcontrollers. The training time is about 240 ms on AVR microcontrollers (16MHz), 95 ms on SAMD21 (48MHz) and 30 ms on ESP8266 (80 MHz).

Thanks to the simplicity of SEFR, you can also train your model on your computer and quickly copy the content of **weights** and **bias** arrays to your target device.
