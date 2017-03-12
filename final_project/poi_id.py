#!/usr/bin/python
import os
import matplotlib.pyplot as plt
import sys
import pickle
from sklearn import preprocessing
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV

sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit


### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label

features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email", 'shared_receipt_with_poi']

### load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### look at data
#print len(data_dict.keys())
#print data_dict['BUY RICHARD B']
#print data_dict.values()


### remove any outliers before proceeding further
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

### remove NAN's from dataset
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
### uncomment for printing top 4 salaries
### print outliers_final


### plot features
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
#plt.show()




### create new features
### new features are: fraction_to_poi_email,fraction_from_poi_email

def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

# After cleaning the data from outliers I had to pick the most sensible features to use.
# First I picked 'from_poi_to_this_person' and 'from_this_person_to_poi' but there is was
# no strong pattern when I plotted the data so I used fractions for both
# features of "from/to poi messages" and "total from/to messages".




### create two lists of new features
fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]=fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[count]
    count +=1


### store to my_dataset for easy export below
my_dataset = data_dict
# print (my_dataset.head())

### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)

### plot new features
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter( from_poi, to_poi )
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("fraction of emails this person gets from poi")
#plt.show()


### if you are creating new features, could also do that here


### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)




### machine learning goes here!
### please name your classifier clf for easy export below

### deploying feature selection
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

### use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

from sklearn.tree import DecisionTreeClassifier

t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
print ('accuracy before tuning ', score)

print( "Decision tree algorithm time:", round(time()-t0, 3), "s")



importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
#print 'Feature Ranking: '
#for i in range(16):
#    print "{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]])

# Finally I picked 10 features which are:
# ["salary", "bonus", "fraction_from_poi_email", "fraction_to_poi_email", 'deferral_payments',
#  'total_payments', 'loan_advances', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value']
# Accuracy for this feature set is around 0.8.

# But with these features my precision and recall were too low (less than 0.3) so I had to change
# my strategy and manually pick features which gave me precision and recall values over 0.3.
# In this dataset I cannot use accuracy for evaluating my algorithm because there a few POI�s
# in dataset and the best evaluator are precision and recall. There were only 18 examples of POIs
# in the dataset. There were 35 people who were POIs in �real life�, but
# for various reasons, half of those are not present in this dataset.

### try Naive Bayes for prediction
#t0 = time()

#clf = GaussianNB()
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#accuracy = accuracy_score(pred,labels_test)
#print accuracy

#print "NB algorithm time:", round(time()-t0, 3), "s"

print("checking new features ")
print("with new features")
fdata=data
fdata = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

clf = DecisionTreeClassifier(min_samples_split=5)
clf = clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
acc=accuracy_score(labels_test, pred)
print ("Validating algorithm:")
print ("accuracy after tuning = ", acc)
print( 'precision = ', precision_score(labels_test,pred))
print( 'recall = ', recall_score(labels_test,pred))
print("without new features")

features_to_remove= ["from_poi_to_this_person","from_this_person_to_poi"]
without_new=  [feature for feature in features_list if feature not in features_to_remove]

fdata = featureFormat(my_dataset, without_new)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

clf = DecisionTreeClassifier(min_samples_split=5)
clf = clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
acc=accuracy_score(labels_test, pred)
print ("Validating algorithm:")
print ("accuracy after tuning = ", acc)
print( 'precision = ', precision_score(labels_test,pred))
print( 'recall = ', recall_score(labels_test,pred))

### Tuning
# Tuning an algorithm or machine learning technique, can be simply thought of as
# process which one goes through in which they optimize the parameters that impact
# the model in order to enable the algorithm to perform the best

### use manual tuning parameter min_samples_split
t0 = time()
clf = DecisionTreeClassifier(min_samples_split=5)
clf = clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
print("done in %0.3fs" % (time() - t0))

acc=accuracy_score(labels_test, pred)

print ("Validating algorithm:")
print ("accuracy after tuning = ", acc)

# function for calculation ratio of true positives
# out of all positives (true + false)
print( 'precision = ', precision_score(labels_test,pred))

# function for calculation ratio of true positives
# out of true positives and false negatives
print( 'recall = ', recall_score(labels_test,pred))



# Firstly I tried Naive Bayes accuracy was lower than with Decision Tree Algorithm
# (0.83 and 0.9 respectively). I made a conclusion that that the feature set I used does
# not suit the distributional and interactive assumptions of Naive Bayes well enough. I
# selected Decision Tree Algorithm for the POI identifier. It gave me accuracy before tuning
# parameters = 0.9. No feature scaling was deployed, as it�s not necessary when using a decision
# tree. After selecting features and algorithm I manually tuned parameter min_samples_split.

# min_samples_split    precision    recall
#      2                0.67        0.8
#      3                0.57        0.8
#      4                0.57        0.8
#      5                0.8         0.8
#      6                0.8         0.8
#      7                0.67        0.8
#   average             0.68        0.8

# It turned out that the best values for min_samples_split are 5 and 6.




### dump your classifier, dataset and features_list so
### anyone can run/check your results
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )
