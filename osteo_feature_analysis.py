import pandas as pd
import numpy as np
import os
import pickle
import argparse
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier

# custom function import
from feature_utils import dummy_labelize_swk
from feature_performance_utils import feature_selection_swk, performance_swk, performance_all_swk, return_selected_feature_names
from feature_oversampling_utils import random_oversampling, smote

np.random.seed(1010)

parser = argparse.ArgumentParser(description="feature analysis parameters")
parser.add_argument("-i", dest = "input_feature_file", help="input feature file")
parser.add_argument("--start_k", dest = "start_k", help="number of selected features", type=int, default=5)
parser.add_argument("--end_k", dest = "end_k", help="number of selected features", type=int, default=40)
parser.add_argument("-o", dest = "save_model_name", help="name of output saved model")
parser.add_argument("-l", dest = "label_file", help="input label file")
parser.add_argument("-f", dest = "excel_filename", help="name of output excel file")
parser.add_argument("-r", dest = "test_ratio", help="test set ratio", default = 0.20, type = float)
parser.add_argument("-n", dest = "number_of_tries", help="number of tries", default = 50, type = int)
parser.add_argument("-os", dest = "oversample", help="oversampling ?", default = False, type = bool)
parser.add_argument("--yes_cv", help="cross validation ?", default = False, type = bool)
parser.add_argument("--save_roc", dest = "save_roc", help="save roc curve figure", default = True, type = bool)
parser.add_argument("--save_confusion", dest = "save_confusion", help="save confusion matrix figure", default = True, type = bool)

args = parser.parse_args()

whole_feature = args.input_feature_file
whole_feature = pd.read_excel(whole_feature)
k_range = (args.start_k, args.end_k)

# label 을 아래와 같이 열면 된다.
label_file_name = args.label_file
pi = open(label_file_name, 'rb')
label_dict = pickle.load(pi)

try:
    label_dict.pop('2018_0022') # 이 코드를 작성하는 시점에서 에러가 나는 데이터를 제거한다.
except:
    pass

# label 이 없는 7명의 환자에 대해 데이터를 제거하여 준다
no_label = [a for a in whole_feature.to_dict('split')['index'] if a not in label_dict.keys()]
whole_feature = whole_feature.drop(no_label)

feature_data = whole_feature.to_dict('split')['data'] # This is input x
feature_names = whole_feature.to_dict('split')['columns']
feature_index = whole_feature.to_dict('split')['index']

feature_label = list(label_dict.values()) # This is original x label

# feature_label 을 3 class 가 아닌 binary class 로 바꿔준다
# config1 : osteoporosis vs. osteopenia & normal
# config2 : osteoporosis & osteopenia vs. normal
# config3 : osteoporosis vs. osteopenia vs. normal ( 3 class classification )

binary_feature_label = np.array(feature_label)

# if config 1
# binary_feature_label[binary_feature_label == 1] = 0
# binary_feature_label[binary_feature_label == 2] = 1
# binary_target_names = ['normal and osteopenia', 'osteoporosis']

# if config 2
binary_feature_label[binary_feature_label == 2] = 1
binary_target_names = ['normal', 'osteopenia and osteoporosis']

# if config 3
# target_names = ['normal', 'osteopenia', 'osteoporosis']

print('Whole input data : %d' % len(feature_index))
print('Oversampling : ', args.oversample)
print('cross validation : ', args.yes_cv)

# Feature analysis

whole_results = []
whole_k_selected_features = pd.DataFrame()

# feature selection methods

test_set_ratio = args.test_ratio
feature_label = binary_feature_label

highest_acc = 0.0

for n in range(args.number_of_tries):

    print('( %d / %d )' %(n + 1, args.number_of_tries))

    temp_results = []
        
    for k in range(*k_range):

        # k 만큼 새로운 x 데이터를 만든다
        new_x, selected_feature_index = feature_selection_swk(feature_data = feature_data, feature_label = feature_label, k = k, return_names = True)

        if n == 0 :
            temp = {k : return_selected_feature_names(selected_feature_index, feature_names)}
            temp_pd = pd.DataFrame(temp, index=['feature'+str(i+1) for i in range(k)]).T
            whole_k_selected_features = pd.concat([whole_k_selected_features, temp_pd], sort=False)

        if args.oversample : 
            X_train, X_test, y_train, y_test = train_test_split(new_x, feature_label, test_size=test_set_ratio, random_state=np.random.randint(1000)) # train test set divide

            # X_train, y_train = random_oversampling(X_train, y_train, random_state = np.random.randint(1000))
            X_train, y_train = smote(X_train, y_train, random_state = np.random.randint(1000))

        else:
            X_train, X_test, y_train, y_test = train_test_split(new_x, feature_label, test_size=test_set_ratio, random_state=np.random.randint(1000)) # train test set divide

        # random forest classifier
        clf = RandomForestClassifier(n_estimators=16, max_depth=8, random_state=10)

        if args.yes_cv:

            # 5 cross validation

            cv_results = cross_validate(clf, X_train, y_train, cv=5, return_train_score=True, return_estimator=True)

            best_model = cv_results['estimator'][np.argmax(cv_results['test_score'])]

            clf = best_model

        else : 

            # train
            clf.fit(X_train, y_train)

        # performance check
        # train_performance, test_performance = performance_swk(clf, X_train, X_test, y_train, y_test)

        # print("Training dataset Accuracy : %.3f" % train_performance)
        # print("Test dataset Accuracy : %.3f" % test_performance)

        # performance all
        result = performance_all_swk(clf, X_test, y_test, roc_figure_save = args.save_roc, confusion_matrix_save = args.save_confusion)

        # save with best accuracy

        acc = result[3] # accuracy

        if acc > highest_acc :

            print("New Record !! : ", acc)

            highest_acc = acc

            # save model
            filename = args.save_model_name
            pickle.dump(clf, open(filename, 'wb'))

            np.save('./selected_features.npy', selected_feature_index)

        temp_results.append(result)

    whole_results.append(temp_results)

# get average of each results
whole_mean_results = np.mean(whole_results, axis=0)
whole_std_results = np.std(whole_results, axis=0)

# save results
whole_result_df = pd.DataFrame(np.hstack((whole_mean_results, whole_std_results)), index = np.arange(*k_range), \
            columns = ['auc-roc mean', 'precision mean', 'f1 score mean',  'accuracy mean'] + ['auc-roc std', 'precision std', 'f1 score std', 'accuracy std'])
whole_result_df.to_excel(args.excel_filename)

whole_k_selected_features.to_excel(args.excel_filename.split('.xlsx')[0] + '_selected_features.xlsx')

# save model
# filename = args.save_model_name
# pickle.dump(clf, open(filename, 'wb'))
print('Radiomics classification model saved')