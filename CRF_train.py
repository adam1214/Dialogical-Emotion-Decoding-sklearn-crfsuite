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
from time import time
import argparse
from argparse import RawTextHelpFormatter

plt.style.use('ggplot')\

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-n", "--model_num", type=int, help="which model number you want to train?", default=1)
    args = parser.parse_args()

    start_t = time()
    emo_mapping_dict = {'ang':'a', 'hap':'h', 'neu':'n', 'sad':'s', 'Start':'Start', 'End':'End', 'pre-trained':'p', 0:'ang', 1:'hap', 2:'neu', 3:'sad'}
    emo_dict = joblib.load('data/emo_all_iemocap.pkl')
    dialogs = joblib.load('data/dialog_iemocap.pkl')
    dialogs_edit = joblib.load('data/dialog_4emo_iemocap.pkl')
    out_dict = joblib.load('data/outputs.pkl')

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
    # c2: default = 1.0
    # max_iterations: default=1000
    # period:The duration of iterations to test the stopping criterion. default = 10
    # delta:The threshold for the stopping criterion; an iteration stops when the improvement of the log likelihood over the last period iterations is no greater than this threshold. default = 1e-5
    # calibration_eta:The initial value of learning rate (eta) used for calibration. default=0.1
    # calibration_rate:The rate of increase/decrease of learning rate for calibration. default=2.0
    # calibration_samples:The number of instances used for calibration. The calibration routine randomly chooses instances no larger than calibration_samples. default=1000
    # calibration_candidates:The number of candidates of learning rate. The calibration routine terminates after finding calibration_samples candidates of learning rates that can increase log-likelihood. default=10
    # calibration_max_trials:The maximum number of trials of learning rates for calibration. The calibration routine terminates after trying calibration_max_trials candidate values of learning rates. default=20
    
    crf = sklearn_crfsuite.CRF(algorithm='l2sgd') 
    params_space = {
        'c2': np.linspace(start = 0.01, stop = 1,  endpoint=True, num = 10),
        'max_iterations': range(800, 1201, 100),
        'period': range(8, 13, 1),
        'delta': np.linspace(start = 0.000001, stop = 0.0001,  endpoint = True, num = 20),
        'calibration_eta': np.linspace(start = 0.001, stop = 0.1,  endpoint = True, num = 20),
        'calibration_rate': np.linspace(start = 1.0, stop = 5.0,  endpoint = True, num = 5),
        'calibration_samples': [800, 1000, 1200, 1400],
        'calibration_candidates': [5, 10, 15, 20, 25],
        'calibration_max_trials': [10, 15, 20, 25]
    }
    # use the same metric for evaluation
    scorer = make_scorer(metrics.flat_recall_score, average='macro', labels=['ang', 'hap', 'neu', 'sad'])
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
    print('Best c2:', s_CV.best_estimator_.c2)
    print('Best max_iterations', s_CV.best_estimator_.max_iterations)
    print('Best period:', s_CV.best_estimator_.period) 
    print('Best delta:', s_CV.best_estimator_.delta)
    print('Best calibration_eta', s_CV.best_estimator_.calibration_eta)
    print('Best calibration_rate', s_CV.best_estimator_.calibration_rate)
    print('Best calibration_samples', s_CV.best_estimator_.calibration_samples)
    print('Best calibration_candidates', s_CV.best_estimator_.calibration_candidates)
    print('Best calibration_max_trials', s_CV.best_estimator_.calibration_max_trials)
    
    '''
    print('===================================================') 
    print("Top likely transitions:") # transition feature coefficients {(label_from, label_to) -- coef}
    print_transitions(Counter(crf.transition_features_).most_common(len(crf.transition_features_)))
    print('===================================================')
    print("Top positive:") # state feature coefficients {(attr_name, label) -- coef}
    print_state_features(Counter(crf.state_features_).most_common(len(crf.state_features_)))
    
    '''