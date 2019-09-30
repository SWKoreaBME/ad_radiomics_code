#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
from sklearn.model_selection import train_test_split

from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

from feature_classifier_utils import ensemble_voting, GridSVM
from feature_performance_utils import load_texture_feature, load_clinical_data, save_confusion_matrix, Validation, normal_porosis, save_roc_curve

np.random.seed(1010)


# ### Concatenate Clinical data to feature data

# ### Grid Search CV

# In[3]:


### feature_2018 = '../final_2018_result/2018_texture_features.xlsx'
feature_2012 = '../final_2012_result/texture_feautures.xlsx'
feature_2018 = '../final_2018_result/2018_texture_features.xlsx'
feature_2013 = '../final_2013_result/texture_features_zscore.xlsx'
feature_2015 = '../final_2015_result/texture_features.xlsx'
feature_2014 = '../final_2014_result/texture_features.xlsx'

# label_2018 = '../data/2018_label_dict.pickle'
# label_2012 = '../data/2012_label_dict.pickle'

# label_2018_class3 = '../data/2018_label_dict_class3.pickle'
# label_2012_class3 = '../data/2012_label_dict_class3.pickle'

label_2018_class1 = '../data/2018_label_dict_class1.pickle'
label_2012_class1 = '../data/2012_label_dict_class1.pickle'
label_2013_class1 = '../data/2013_label_dict_class1.pickle'
label_2014_class1 = '../data/2014_label_dict_class1.pickle'
label_2015_class1 = '../data/2015_label_dict_class1.pickle'

clinic_2018 = '../data/2018_clinical_data.pickle'
clinic_2012 = '../data/2012_clinical_data.pickle'
clinic_2013 = '../data/2013_clinical_data.pickle'
clinic_2014 = '../data/2014_clinical_data.pickle'
clinic_2015 = '../data/2015_clinical_data.pickle'

feature_data_2018, feature_name_2018, feature_index_2018, feature_label_2018_class1 = load_texture_feature(feature_2018, label_2018_class1)

feature_data_2012, feature_name_2012, feature_index_2012, feature_label_2012_class1 = load_texture_feature(feature_2012, label_2012_class1)

feature_data_2013, feature_name_2013, feature_index_2013, feature_label_2013_class1 = load_texture_feature(feature_2013, label_2013_class1)

feature_data_2014, feature_name_2014, feature_index_2014, feature_label_2014_class1 = load_texture_feature(feature_2014, label_2014_class1)

feature_data_2015, feature_name_2015, feature_index_2015, feature_label_2015_class1 = load_texture_feature(feature_2015, label_2015_class1)


clinical_data_2018, clinical_feature_names = load_clinical_data(clinic_2018, feature_index_2018)
clinical_data_2012, _ = load_clinical_data(clinic_2012, feature_index_2012)
clinical_data_2013, _ = load_clinical_data(clinic_2013, feature_index_2013)
clinical_data_2014, _ = load_clinical_data(clinic_2014, feature_index_2014)
clinical_data_2015, _ = load_clinical_data(clinic_2015, feature_index_2015)

print(feature_data_2018.shape, len(feature_label_2018_class1), len(clinical_data_2018))
print(feature_data_2012.shape, len(feature_label_2012_class1), len(clinical_data_2012))
print(feature_data_2013.shape, len(feature_label_2013_class1), len(clinical_data_2013))
print(feature_data_2014.shape, len(feature_label_2014_class1), len(clinical_data_2014))
print(feature_data_2015.shape, len(feature_label_2015_class1), len(clinical_data_2015))

# In[ ]:


num_try = 1
try_dict = dict()

# feature_data = feature_data_with_clinical
highest_acc = 0.0

internal_feature_data = np.concatenate((feature_data_2013,
                                        feature_data_2014,
                                        feature_data_2015,
                                        feature_data_2018))

internal_feature_label = np.concatenate((feature_label_2013_class1,
                                         feature_label_2014_class1,
                                         feature_label_2015_class1,
                                         feature_label_2018_class1))

internal_clinical_data = np.concatenate((clinical_data_2013,
                                         clinical_data_2014,
                                         clinical_data_2015,
                                         clinical_data_2018))

# internal_feature_data = feature_data_2013_gmm
# internal_feature_label = feature_label_2013_class1
# internal_clinical_data = clinical_data_2013

external_feature_data = feature_data_2012
external_feature_label = feature_label_2012_class1
external_clinical_data = clinical_data_2012

rus = RandomUnderSampler(random_state=42)

class_3_index = ['normal label', 'penia label', 'porosis label']
class_3_cols = ['normal prediction', 'penia prediction', 'porosis prediction']

class_2_index = ['normal label', 'porosis & penia label']
class_2_cols = ['normal prediction', 'porosis & penia prediction']

class_1_index = ['normal&penia label', 'porosis label']
class_1_cols = ['normal&penia prediction', 'porosis prediction']

if len(Counter(internal_feature_label).keys()) == 2:
    average_method = 'binary'
    conf_index = class_1_index
    conf_cols = class_1_cols
else:
    average_method = None
    conf_index = class_3_index
    conf_cols = class_3_cols
    
print('Average Method : ', average_method)

for i in range(num_try):
    
    k_dict = {}
    
    print("++++++++++")
    print(i+1 , "st try")
    print("++++++++++\n")

    for k in range(1,30):

        selector = SelectKBest(mutual_info_classif, k = k)

        # k 만큼 새로운 x 데이터를 만든다
        new_x = selector.fit_transform(internal_feature_data, internal_feature_label)
        
        # late fusion (after feature selection) clinical data
        new_x = np.hstack((np.array(new_x), internal_clinical_data))

        # feature selection
        selector_info = selector.fit(internal_feature_data, internal_feature_label)
        selected_feature = selector_info.get_support()
        
        new_x, new_y = rus.fit_resample(new_x, internal_feature_label)
#         new_y = internal_feature_label
        
        print("Train Dataset")
        print(Counter(new_y))

        # train test set divide
        X_train, X_test, y_train, y_test = train_test_split(new_x,
                                                            new_y,
                                                            test_size=0.20,
                                                            random_state=42)

        # random forest classifier

        clf = ensemble_voting(X_train, y_train, random_state=42, cv=5)
        
        # train
        clf.fit(X_train, y_train)

        # evaluation
        
        print('================================')
        print("Internal Evaluation for k = ", k)
        
        internal_result = Validation(clf, X_test, y_test)
        
        confusion = internal_result[-1]; save_confusion_matrix(confusion, conf_index, conf_cols, "Internal Validation Confusion matrix")

        k_dict[k] = [internal_result, selected_feature]
        
        print('================================')
        print("External Validation for k = ", k)
        
        new_external_x = [a[selected_feature==True] for a in np.array(external_feature_data)]
        new_external_x = np.hstack((np.array(new_external_x), external_clinical_data))
        
        new_external_x, new_y = rus.fit_resample(new_external_x, external_feature_label)
        
        print(Counter(new_y))
        
        ext_val_result = Validation(clf, new_external_x, new_y)
        
        confusion = ext_val_result[-1]; save_confusion_matrix(confusion, conf_index, conf_cols, "External Validation Confusion matrix")

        print('================================')
        
    try_dict[i] = k_dict

print('done')