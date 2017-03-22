# Identify-Fraud-with-Machine-Learning

Small Machine Learning experience with Enron Dataset
# Investigating Enron's scandal using Machine Learning
===================================================
By: Ahmet Hamza Emra


## Introduction
> [In addition to being the largest bankruptcy reorganization in American history at that time, Enron was cited as the biggest audit failure](http://en.wikipedia.org/wiki/Enron_scandal)

From a $90 price per share, to a $1 value represents the huge value loss and scam that happened in Enron. This case has been
a point of interest for machine learning analysis because of the huge real-world impact that ML could help out and try to figure out what went wrong and how to avoid it in the future. It would be of great value to find a model that could potentially predict these types of events before much damage is done, so as to permit preventive action. Corporate governance, the stock market, and even the Government would be quite interested in a machine learning model that could signal potential fraud detections before hand.

## Enron Data
The interesting and hard part of the dataset is that the distribution of the non-POI's to POI's is very skewed, given that from the 146 there are only 11 people or data points labeled as POI's or guilty of fraud. We are interested in labeling every person in the dataset
into either a POI or a non-POI (POI stands for *Person Of Interest*). More than that, if we can assign a probability to each person to see what is the chance she is POI, it would be a much more reasonable model given that there is always some uncertainty.


## Algorithms selection and tuning
For the analysis of the data, a total of 10 classifiers were tried out, which include:

- Decision Tree Classifier
- Gaussian Naive Bayes
- KMeans


 Discussion and Conclusions
This was just a starting point analysis for classifying Enron employees. The results should not be taken too seriously and more advanced models should be used. Possibilities for future research could be to include more complex pipelines for the data, or even Neural Networks. Here we tried a basic neural network, but the SkLearn library is very limited in what it has to offer in this regard.

## References
- [Udacity - Intro to Machine Learning course](https://www.udacity.com/course/ud120)
- [Sklearn documentation](http://scikit-learn.org/stable/documentation.html)


## How to run the code

To run the code, make sure you are located in the folder `final_project`.
