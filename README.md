# SEFR: Fast Linear Multiclass Classifier (On-Device Machine Learning Training/Prediction for Low-Power Microcontrollers)

![Figure_1](https://user-images.githubusercontent.com/44191076/99478109-c7da4f00-298e-11eb-9331-0af528409ce7.png)

This is based on [sefr-classifier/sefr](https://github.com/sefr-classifier/sefr) and [SEFR: A Fast Linear-Time Classifier for Ultra-Low Power Devices](https://arxiv.org/abs/2006.04620). The original proposed version is a binary classifier; I modified it to do multi-label classification by using one-vs-rest strategy.

SEFR is indeed very fast. In the Python version (using the IRIS dataset) running on my laptop, the training time (~500 nanosecond on my laptop) is only about less than 20% of KNN (K=3), less than 6% of Linear SVM (100 iterations) and less than 2% of logistic regression (100 iterations). For the IRIS dataset, SEFR can achieve 85-95% accuracy in most scenarios. 

The microcontroller versions all have a built-in IRIS dataset ported from scikit-learn with the potential option to input new data into the training dataset (and re-train model on device). They would use the whole dataset for training and can predict one new data instance at a time (here I simply pick a data from the dataset and add random noises).

The Arduino C++ version is runnable on Arduino Uno or other 2K memory AVR microcontrollers. The training time is currently about 68 ms on AVRs (16MHz), 27 ms on SAMD21 (48MHz), 7 ms on ESP8266 (80 MHz) and merely 1 ms on ESP32 (240 MHz).

The MicroPython version can be run on ESP8266 and ESP32, and the Go\TinyGo version can be run on 32-bit microcontrollers (not possible on AVRs). Training time for both are slower than the C++ version due to their designed nature. The CircuitPython version should be able to run on CP firmwares designed for Adafruit SAMD51 (M4) and ESP32-S2 boards that have the **ulab** (MicroPython NumPy) module built-in.
