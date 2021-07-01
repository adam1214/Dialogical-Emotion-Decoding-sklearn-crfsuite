import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import nltk
from numpy.lib.function_base import average

import sklearn
import scipy.stats
import joblib
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import PredefinedSplit
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter
import utils
import random
from time import time
import argparse
from argparse import RawTextHelpFormatter

plt.style.use('ggplot')\

np.random.seed(1)
random.seed(1)

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
        
def utt2features(dialog, i):
    current_ses = dialog[i][0][:5] #Ses01
    features = {        
        'pretrained_a':out_dict[dialog[i][0]][0],
        'pretrained_h':out_dict[dialog[i][0]][1],
        'pretrained_n':out_dict[dialog[i][0]][2],
        'pretrained_s':out_dict[dialog[i][0]][3]
    }
    return features

def dialog2features(dialog):
    return [utt2features(dialog, i) for i in range(len(dialog))]

def dialog2labels(dialog):
    return [emo for utt, spk, emo in dialog]

def construct_train_test(emo_dict, dialogs_edit):
    Ses01_list = []
    Ses02_list = []
    Ses03_list = []
    Ses04_list = []
    Ses05_list = []
    
    for dialog in dialogs_edit.values():
        Ses_num = dialog[0][:5]
        if Ses_num == 'Ses01':
            Ses01_list.append([])
        elif Ses_num == 'Ses02':
            Ses02_list.append([])
        elif Ses_num == 'Ses03':
            Ses03_list.append([])
        elif Ses_num == 'Ses04':
            Ses04_list.append([])
        elif Ses_num == 'Ses05':
            Ses05_list.append([])
        for utt in dialog:
            spk = utt[-4]
            emo = emo_dict[utt]
            if Ses_num == 'Ses01':
                Ses01_list[len(Ses01_list)-1].append((utt, spk, emo))
            elif Ses_num == 'Ses02':
                Ses02_list[len(Ses02_list)-1].append((utt, spk, emo))
            elif Ses_num == 'Ses03':
                Ses03_list[len(Ses03_list)-1].append((utt, spk, emo))
            elif Ses_num == 'Ses04':
                Ses04_list[len(Ses04_list)-1].append((utt, spk, emo))
            elif Ses_num == 'Ses05':
                Ses05_list[len(Ses05_list)-1].append((utt, spk, emo))
    train_val_dialogs1 = Ses02_list + Ses03_list + Ses04_list + Ses05_list
    test_dialogs1 = Ses01_list

    train_val_dialogs2 = Ses01_list + Ses03_list + Ses04_list + Ses05_list
    test_dialogs2 = Ses02_list

    train_val_dialogs3 = Ses01_list + Ses02_list + Ses04_list + Ses05_list
    test_dialogs3 = Ses03_list

    train_val_dialogs4 = Ses01_list + Ses02_list + Ses03_list + Ses05_list
    test_dialogs4 = Ses04_list

    train_val_dialogs5 = Ses01_list + Ses02_list + Ses03_list + Ses04_list
    test_dialogs5 = Ses05_list

    return train_val_dialogs1, train_val_dialogs2, train_val_dialogs3, train_val_dialogs4, train_val_dialogs5, test_dialogs1, test_dialogs2, test_dialogs3, test_dialogs4, test_dialogs5, Ses01_list, Ses02_list, Ses03_list, Ses04_list, Ses05_list

def my_custom_score(y_true, y_pred):
    predict_val = []
    true_val = []
    for sub_list in y_pred:
        predict_val += sub_list
    
    for sub_list in y_true:
        true_val += sub_list
    
    for i in range(0, len(predict_val), 1):
        if predict_val[i] == 'ang':
            predict_val[i] = 0
        elif predict_val[i] == 'hap':
            predict_val[i] = 1
        elif predict_val[i] == 'neu':
            predict_val[i] = 2
        elif predict_val[i] == 'sad':
            predict_val[i] = 3
        
        if true_val[i] == 'ang':
            true_val[i] = 0
        elif true_val[i] == 'hap':
            true_val[i] = 1
        elif true_val[i] == 'neu':
            true_val[i] = 2
        elif true_val[i] == 'sad':
            true_val[i] = 3
        else:
            true_val[i] = -1
        
    uar, acc, conf = utils.evaluate(predict_val, true_val)
    return uar
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-n", "--model_num", type=int, help="which model number you want to train?", default=1)
    args = parser.parse_args()

    start_t = time()
    emo_mapping_dict = {'ang':'a', 'hap':'h', 'neu':'n', 'sad':'s', 'Start':'Start', 'End':'End', 'pre-trained':'p', 0:'ang', 1:'hap', 2:'neu', 3:'sad'}
    emo_dict = joblib.load('../../data/emo_all_iemocap.pkl')
    dialogs = joblib.load('../../data/dialog_iemocap.pkl')
    dialogs_edit = joblib.load('../../data/dialog_4emo_iemocap.pkl')
    out_dict = joblib.load('../../data/outputs.pkl')

    train_val_dialogs1, train_val_dialogs2, train_val_dialogs3, train_val_dialogs4, train_val_dialogs5, test_dialogs1, test_dialogs2, test_dialogs3, test_dialogs4, test_dialogs5, Ses01_list, Ses02_list, Ses03_list, Ses04_list, Ses05_list = construct_train_test(emo_dict, dialogs_edit)
    if args.model_num == 1:
        X_train_val = [dialog2features(s) for s in train_val_dialogs1]
        y_train_val = [dialog2labels(s) for s in train_val_dialogs1]
        
        # Create a list where train data indices are -1 and validation data indices are 0
        # train + val: 2, 3, 4, 5
        split_index = np.zeros(len(X_train_val))
        split_index[len(Ses02_list):len(Ses02_list+Ses03_list)] = 1
        split_index[len(Ses02_list+Ses03_list):len(Ses02_list+Ses03_list+Ses04_list)] = 2
        split_index[len(Ses02_list+Ses03_list+Ses04_list):] = 3
    elif args.model_num == 2:
        X_train_val = [dialog2features(s) for s in train_val_dialogs2]
        y_train_val = [dialog2labels(s) for s in train_val_dialogs2]
        
        # Create a list where train data indices are -1 and validation data indices are 0
        # train + val: 1, 3, 4, 5
        split_index = np.zeros(len(X_train_val))
        split_index[len(Ses01_list):len(Ses01_list+Ses03_list)] = 1
        split_index[len(Ses01_list+Ses03_list):len(Ses01_list+Ses03_list+Ses04_list)] = 2
        split_index[len(Ses01_list+Ses03_list+Ses04_list):] = 3
    elif args.model_num == 3:
        X_train_val = [dialog2features(s) for s in train_val_dialogs3]
        y_train_val = [dialog2labels(s) for s in train_val_dialogs3]
        
        # Create a list where train data indices are -1 and validation data indices are 0
        # train + val: 1, 2, 4, 5
        split_index = np.zeros(len(X_train_val))
        split_index[len(Ses01_list):len(Ses01_list+Ses02_list)] = 1
        split_index[len(Ses01_list+Ses02_list):len(Ses01_list+Ses02_list+Ses04_list)] = 2
        split_index[len(Ses01_list+Ses02_list+Ses04_list):] = 3
    elif args.model_num == 4:
        X_train_val = [dialog2features(s) for s in train_val_dialogs4]
        y_train_val = [dialog2labels(s) for s in train_val_dialogs4]
        
        # Create a list where train data indices are -1 and validation data indices are 0
        # train + val: 1, 2, 3, 5
        split_index = np.zeros(len(X_train_val))
        split_index[len(Ses01_list):len(Ses01_list+Ses02_list)] = 1
        split_index[len(Ses01_list+Ses02_list):len(Ses01_list+Ses02_list+Ses03_list)] = 2
        split_index[len(Ses01_list+Ses02_list+Ses03_list):] = 3
    elif args.model_num == 5:
        X_train_val = [dialog2features(s) for s in train_val_dialogs5]
        y_train_val = [dialog2labels(s) for s in train_val_dialogs5]
        
        # Create a list where train data indices are -1 and validation data indices are 0
        # train + val: 1, 2, 3, 4
        split_index = np.zeros(len(X_train_val))
        split_index[len(Ses01_list):len(Ses01_list+Ses02_list)] = 1
        split_index[len(Ses01_list+Ses02_list):len(Ses01_list+Ses02_list+Ses03_list)] = 2
        split_index[len(Ses01_list+Ses02_list+Ses03_list):] = 3
    # parameters
    # c1: default = 0.0
    # c2: default = 1.0
    # max_iterations: default=unlimited
    # num_memories: default=6
    # epsilon: default=1e-5
    # period: default=10
    # delta: default=1e-5
    # linesearch: default='MoreThuente'
    # max_linesearch: default=20
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs') 
    params_space = {
        'c1': np.linspace(start = 0.01, stop = 50,  endpoint=True, num = 25),
        'c2': np.linspace(start = 0.01, stop = 50,  endpoint=True, num = 25),
        'num_memories': [1,2,3,4,5,6,7,8,9,10],
        'epsilon': np.linspace(start = 1e-6, stop = 1e-3, endpoint=True, num = 25),
        'period': range(1, 21, 3),
        'delta': np.linspace(start = 1e-6, stop = 1e-4,  endpoint=True, num = 20),
        'linesearch': ['MoreThuente', 'Backtracking', 'StrongBacktracking'],
        'max_linesearch': range(10, 101, 3)
    }
    # use the same metric for evaluation
    scorer = make_scorer(my_custom_score, greater_is_better=True)
    ps = PredefinedSplit(test_fold=split_index)
    
    s_CV = RandomizedSearchCV(crf, params_space, cv=ps, verbose=1, n_jobs=-1, n_iter=10, scoring=scorer, refit=True, random_state=0)
    #s_CV = GridSearchCV(crf, params_space, cv=5, verbose=1, n_jobs=-1, scoring=scorer)
    s_CV.fit(X_train_val, y_train_val)
    crf = s_CV.best_estimator_

    joblib.dump(crf, 'model/Ses0'+str(args.model_num)+'.model')
    end_t = time()
    dur = (end_t-start_t)/60.
    print('Process time:', dur, 'min')

    print('Best score for training data:', s_CV.best_score_)
    
    '''
    print('===================================================') 
    print("Top likely transitions:") # transition feature coefficients {(label_from, label_to) -- coef}
    print_transitions(Counter(crf.transition_features_).most_common(len(crf.transition_features_)))
    print('===================================================')
    print("Top positive:") # state feature coefficients {(attr_name, label) -- coef}
    print_state_features(Counter(crf.state_features_).most_common(len(crf.state_features_)))
    '''