#!/usr/bin/python
# First run this file and then tester.py to evaluate the performance of the
# model trained in this file

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
np.set_printoptions(threshold=np.inf)
import operator
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.grid_search import GridSearchCV

### features to be used
'''see 'feature_selection.py' file'''

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
'''TOTAL is a spreadsheet quirk that added up all the data points
'THE TRAVEL AGENCY IN THE PARK' is not a person name'''

###Create new feature(s)

'''a function to create new features'''
def new_feature(new_feature_name, feature1, feature2, fun):    
    for key in data_dict:
        if data_dict[key][feature1] == 'NaN' or data_dict[key][feature2] == 'NaN':
            data_dict[key][new_feature_name] = 'NaN'
        else:
            data_dict[key][new_feature_name] = fun(data_dict[key][feature1],data_dict[key][feature2])
    

'''new feature 1 = ratio of bonus/salary'''
new_feature('bonus_salary_ratio', 'bonus', 'salary', operator.truediv)

'''new feature 2 = prop_from_this_person_to_poi (proportion of the   
messages sent to poi)'''
new_feature('prop_from_this_person_to_poi', 'from_this_person_to_poi', 'from_messages', operator.truediv)

'''new feature 3 = prop_from_poi_to_this_person  (proportion of the 
messages recieved from poi)'''
new_feature('prop_from_poi_to_this_person', 'from_poi_to_this_person', 'to_messages', operator.truediv)

'''new feature 4 = sum_prop_correspondences_with_poi  (sum of prop_from_this_person_to_poi
and prop_from_poi_to_this_person)'''
new_feature('sum_prop_correspondences_with_poi', 'prop_from_this_person_to_poi', 'prop_from_poi_to_this_person', operator.add)

features_list = ['poi', 'sum_prop_correspondences_with_poi', 
                  'bonus_salary_ratio']
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Try a varity of classifiers

from sklearn import tree
clf1 = tree.DecisionTreeClassifier(random_state=42)

from sklearn.naive_bayes import GaussianNB
clf2 = GaussianNB()

from sklearn.svm import SVC
clf3 = SVC(random_state=42)

from sklearn.ensemble import AdaBoostClassifier
clf4 = AdaBoostClassifier(random_state=42)

from sklearn.neighbors import KNeighborsClassifier
clf5 = KNeighborsClassifier()

from sklearn.ensemble import RandomForestClassifier
clf6 = RandomForestClassifier(random_state=42)

from sklearn import linear_model
clf7 = linear_model.LogisticRegression(random_state=42)

'''to try out any of the above classifiers, reset clf at line 111 to the 
name of the classifier above that you want test, run this file,
and then run 'tester.py'file'''

### parameter tuning

'''parameter tuning of DecisionTreeClassifier by GridSearchCV'''
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 3, 5],              
              "min_samples_leaf": [1, 5],
              "max_leaf_nodes": [None, 5, 10],              
              }

clf8 = GridSearchCV(tree.DecisionTreeClassifier(random_state = 42), param_grid)

'''to get the best parameters of GridSearchCV, set clf = clf8 at line 111,and 
run this file. Then write 'print clf.best_params_' at 
line 103 (inside main() function) of tester.py and run'''

'''add parameter max_depth = 2'''
clf9 = tree.DecisionTreeClassifier(random_state=42, min_samples_leaf = 5, max_depth = 2)

### dump classifier, dataset, and features_list 
clf = clf8
dump_classifier_and_data(clf, my_dataset, features_list)

'''validation - train_test_split'''
### model evaluation
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=.33, random_state=42)
clf10 = tree.DecisionTreeClassifier(random_state=42, min_samples_leaf = 5, max_depth = 2)    
clf10.fit(features_train, labels_train)
pred = clf10.predict(features_test)

print "Evaluation metrics (train_test_split)"
print "-------------------------------------"
print "Accuracy score: ", accuracy_score(pred, labels_test)
print "Precision: ", precision_score(pred, labels_test)
print "Recall: ", recall_score(pred, labels_test)
print "(Number of POIs in test set: ", sum(labels_test), ")"
print "########################################\n"

'''cross_validation.KFold'''
from sklearn.cross_validation import KFold
clf11 = tree.DecisionTreeClassifier(random_state=42, min_samples_leaf = 5, max_depth = 2)    
kf = KFold(len(labels), 10)

print "Evaluation metrics (KFold)"
print "-------------------------------------"
print "Accuracy\tPrecision\tRecall"

for train_indeces, test_indeces in kf:
    features_train = [features[i] for i in train_indeces]
    features_test = [features[i] for i in test_indeces]
    labels_train = [labels[i] for i in train_indeces]
    labels_test = [labels[i] for i in test_indeces]
    
    clf11.fit(features_train, labels_train)
    pred = clf11.predict(features_test)
    
    
    print accuracy_score(pred, labels_test), "\t\t", \
    round(precision_score(pred, labels_test), 2), "\t\t", \
    recall_score(pred, labels_test)
