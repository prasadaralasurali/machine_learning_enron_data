#!/usr/bin/python

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from tester import dump_classifier_and_data
from sklearn import tree
import numpy as np
np.set_printoptions(threshold=np.inf)

'''load my_dataset_copy.pkl (contains enron data with outliers removed and
new features added)'''
data_dict = pickle.load(open("my_dataset_cleaned.pkl", "r") )

'''create a list of all features in the dataset'''
for key in data_dict:
    feature_list = data_dict[key].keys()
    break

'''remove email_address from the feature_list, as it
 will not be used in analysis'''
feature_list.pop(feature_list.index('poi'))

'''move 'poi' to be the first element of the feature_list'''
feature_list.pop(feature_list.index('email_address'))
features_list = ['poi'] + feature_list

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=42)

'''find out scores of all the features'''
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k = 'all')
selector.fit(features_train, labels_train)
features_train = selector.transform(features_train)
features_test = selector.transform(features_test)

print "list of all features with feature scores (sorted)"
print "-------------------------------------------------"
list_of_features = np.array(features_list)[1:]
scores = selector.scores_
sorted_scores = np.sort(scores)
sorted_features = list_of_features[np.argsort(scores)]

for i in range(22, -1, -1):
    print ("%s:%.2f" % (sorted_features[i],round(sorted_scores[i], 2)))

'''fit model with different combinations of top scoring and other features, and
examine the model performance '''

'''to test different combinations of features, uncomment the specific 
'feature_list' below and uncomment the following lines and then run
 'tester.py' file '''


#features_list = ['poi', 'exercised_stock_options', 'total_stock_value']
#features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus']
#features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus',
#                'prop_from_this_person_to_poi']
#features_list = ['poi', 'sum_prop_correspondences_with_poi', 
#                  'bonus_salary_ratio']
#features_list = ['poi', 'sum_prop_correspondences_with_poi', 'bonus', 
#                 'total_payments']
#features_list = ['poi', 'exercised_stock_options', 'sum_prop_correspondences_with_poi',
#                 'total_payments']
#features_list = ['poi', 'sum_prop_correspondences_with_poi', 'salary',
#                 'shared_receipt_with_poi']


#clf = tree.DecisionTreeClassifier(random_state=42)
#my_dataset = data_dict
#dump_classifier_and_data(clf, my_dataset, features_list)



