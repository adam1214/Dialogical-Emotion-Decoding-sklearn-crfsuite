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

def fix_len(Ses01_list, Ses02_list, Ses03_list, Ses04_list, Ses05_list):
    Ses01_list_fix = []
    Ses02_list_fix = []
    Ses03_list_fix = []
    Ses04_list_fix = []
    Ses05_list_fix = []
    
    for utts_list in Ses01_list:
        Ses01_list_fix.append([])
        for utt in utts_list:
            if args.data_len == len(Ses01_list_fix[len(Ses01_list_fix)-1]):
                Ses01_list_fix.append([])
            Ses01_list_fix[len(Ses01_list_fix)-1].append(utt)
        if len(Ses01_list_fix[len(Ses01_list_fix)-1]) < args.data_len:
            padding_cnt = args.data_len - len(Ses01_list_fix[len(Ses01_list_fix)-1])
            for i in range(0, padding_cnt, 1):
                Ses01_list_fix[len(Ses01_list_fix)-1].append(Ses01_list_fix[len(Ses01_list_fix)-1][-1])
    for utts_list in Ses02_list:
        Ses02_list_fix.append([])
        for utt in utts_list:
            if args.data_len == len(Ses02_list_fix[len(Ses02_list_fix)-1]):
                Ses02_list_fix.append([])
            Ses02_list_fix[len(Ses02_list_fix)-1].append(utt)
        if len(Ses02_list_fix[len(Ses02_list_fix)-1]) < args.data_len:
            padding_cnt = args.data_len - len(Ses02_list_fix[len(Ses02_list_fix)-1])
            for i in range(0, padding_cnt, 1):
                Ses02_list_fix[len(Ses02_list_fix)-1].append(Ses02_list_fix[len(Ses02_list_fix)-1][-1])
    for utts_list in Ses03_list:
        Ses03_list_fix.append([])
        for utt in utts_list:
            if args.data_len == len(Ses03_list_fix[len(Ses03_list_fix)-1]):
                Ses03_list_fix.append([])
            Ses03_list_fix[len(Ses03_list_fix)-1].append(utt)
        if len(Ses03_list_fix[len(Ses03_list_fix)-1]) < args.data_len:
            padding_cnt = args.data_len - len(Ses03_list_fix[len(Ses03_list_fix)-1])
            for i in range(0, padding_cnt, 1):
                Ses03_list_fix[len(Ses03_list_fix)-1].append(Ses03_list_fix[len(Ses03_list_fix)-1][-1])
    for utts_list in Ses04_list:
        Ses04_list_fix.append([])
        for utt in utts_list:
            if args.data_len == len(Ses04_list_fix[len(Ses04_list_fix)-1]):
                Ses04_list_fix.append([])
            Ses04_list_fix[len(Ses04_list_fix)-1].append(utt)
        if len(Ses04_list_fix[len(Ses04_list_fix)-1]) < args.data_len:
            padding_cnt = args.data_len - len(Ses04_list_fix[len(Ses04_list_fix)-1])
            for i in range(0, padding_cnt, 1):
                Ses04_list_fix[len(Ses04_list_fix)-1].append(Ses04_list_fix[len(Ses04_list_fix)-1][-1])
    for utts_list in Ses05_list:
        Ses05_list_fix.append([])
        for utt in utts_list:
            if args.data_len == len(Ses05_list_fix[len(Ses05_list_fix)-1]):
                Ses05_list_fix.append([])
            Ses05_list_fix[len(Ses05_list_fix)-1].append(utt)
        if len(Ses05_list_fix[len(Ses05_list_fix)-1]) < args.data_len:
            padding_cnt = args.data_len - len(Ses05_list_fix[len(Ses05_list_fix)-1])
            for i in range(0, padding_cnt, 1):
                Ses05_list_fix[len(Ses05_list_fix)-1].append(Ses05_list_fix[len(Ses05_list_fix)-1][-1])
    
    return Ses01_list_fix, Ses02_list_fix, Ses03_list_fix, Ses04_list_fix, Ses05_list_fix

def construct_train_test(emo_dict, dias):
    Ses01_list = []
    Ses02_list = []
    Ses03_list = []
    Ses04_list = []
    Ses05_list = []
    
    for dialog in dias.values():
        Ses_num = dialog[0][:5]
        if Ses_num == 'Ses01':
            Ses01_list.append([])
            Ses01_list.append([])
        elif Ses_num == 'Ses02':
            Ses02_list.append([])
            Ses02_list.append([])
        elif Ses_num == 'Ses03':
            Ses03_list.append([])
            Ses03_list.append([])
        elif Ses_num == 'Ses04':
            Ses04_list.append([])
            Ses04_list.append([])
        elif Ses_num == 'Ses05':
            Ses05_list.append([])
            Ses05_list.append([])
        for utt in dialog:
            spk = utt[-4]
            emo = emo_dict[utt]
            if Ses_num == 'Ses01':
                if spk == 'F':
                    Ses01_list[len(Ses01_list)-2].append((utt, spk, emo))
                elif spk == 'M':
                    Ses01_list[len(Ses01_list)-1].append((utt, spk, emo))
            elif Ses_num == 'Ses02':
                if spk == 'F':
                    Ses02_list[len(Ses02_list)-2].append((utt, spk, emo))
                elif spk == 'M':
                    Ses02_list[len(Ses02_list)-1].append((utt, spk, emo))
            elif Ses_num == 'Ses03':
                if spk == 'F':
                    Ses03_list[len(Ses03_list)-2].append((utt, spk, emo))
                elif spk == 'M':
                    Ses03_list[len(Ses03_list)-1].append((utt, spk, emo))
            elif Ses_num == 'Ses04':
                if spk == 'F':
                    Ses04_list[len(Ses04_list)-2].append((utt, spk, emo))
                elif spk == 'M':
                    Ses04_list[len(Ses04_list)-1].append((utt, spk, emo))
            elif Ses_num == 'Ses05':
                if spk == 'F':
                    Ses05_list[len(Ses05_list)-2].append((utt, spk, emo))
                elif spk == 'M':
                    Ses05_list[len(Ses05_list)-1].append((utt, spk, emo))
        if Ses_num == 'Ses01':
            if len(Ses01_list[-1]) == 0:
                del Ses01_list[-1]
            if len(Ses01_list[-2]) == 0:
                del Ses01_list[-2]
        elif Ses_num == 'Ses02':
            if len(Ses02_list[-1]) == 0:
                del Ses02_list[-1]
            if len(Ses02_list[-2]) == 0:
                del Ses02_list[-2]
        elif Ses_num == 'Ses03':
            if len(Ses03_list[-1]) == 0:
                del Ses03_list[-1]
            if len(Ses03_list[-2]) == 0:
                del Ses03_list[-2]
        elif Ses_num == 'Ses04':
            if len(Ses04_list[-1]) == 0:
                del Ses04_list[-1]
            if len(Ses04_list[-2]) == 0:
                del Ses04_list[-2]
        elif Ses_num == 'Ses05':
            if len(Ses05_list[-1]) == 0:
                del Ses05_list[-1]
            if len(Ses05_list[-2]) == 0:
                del Ses05_list[-2]
    
    Ses01_list_fix, Ses02_list_fix, Ses03_list_fix, Ses04_list_fix, Ses05_list_fix = fix_len(Ses01_list, Ses02_list, Ses03_list, Ses04_list, Ses05_list)
    
    train_dialogs1 = Ses02_list_fix + Ses03_list_fix + Ses04_list_fix + Ses05_list_fix
    test_dialogs1 = Ses01_list_fix

    train_dialogs2 = Ses01_list_fix + Ses03_list_fix + Ses04_list_fix + Ses05_list_fix
    test_dialogs2 = Ses02_list_fix

    train_dialogs3 = Ses01_list_fix + Ses02_list_fix + Ses04_list_fix + Ses05_list_fix
    test_dialogs3 = Ses03_list_fix

    train_dialogs4 = Ses01_list_fix + Ses02_list_fix + Ses03_list_fix + Ses05_list_fix
    test_dialogs4 = Ses04_list_fix

    train_dialogs5 = Ses01_list_fix + Ses02_list_fix + Ses03_list_fix + Ses04_list_fix
    test_dialogs5 = Ses05_list_fix

    return train_dialogs1, train_dialogs2, train_dialogs3, train_dialogs4, train_dialogs5, test_dialogs1, test_dialogs2, test_dialogs3, test_dialogs4, test_dialogs5, Ses01_list_fix, Ses02_list_fix, Ses03_list_fix, Ses04_list_fix, Ses05_list_fix

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
    parser.add_argument("-l", "--data_len", type=int, help="how many utts per data?", default = 5)
    parser.add_argument("-d", "--dataset", type=str, help="which dataset to use? original or C2C or U2U", default = 'original')
    args = parser.parse_args()
    
    start_t = time()
    emo_mapping_dict = {'ang':'a', 'hap':'h', 'neu':'n', 'sad':'s', 'Start':'Start', 'End':'End', 'pre-trained':'p', 0:'ang', 1:'hap', 2:'neu', 3:'sad'}
    map_emo = {'ang':0, 'hap':1, 'neu':2, 'sad':3}
    emo_dict = joblib.load('../../data/emo_all_iemocap.pkl')
    
    if args.dataset == 'original':
        emo_dict = joblib.load('../../data/emo_all_iemocap.pkl')
    elif args.dataset == 'C2C':
        emo_dict = joblib.load('../../data/C2C_4emo_all_iemocap.pkl')
    elif args.dataset == 'U2U':
        emo_dict = joblib.load('../../data/U2U_4emo_all_iemocap.pkl')

    dialogs = joblib.load('../../data/dialog_iemocap.pkl')
    dialogs_edit = joblib.load('../../data/dialog_4emo_iemocap.pkl')
    out_dict = joblib.load('../../data/outputs.pkl')

    #intra_emo_trans_prob_dict = utils.get_val_emo_trans_prob(emo_dict, dialogs_edit)
    
    if args.dataset == 'original':
        train_val_dialogs1, train_val_dialogs2, train_val_dialogs3, train_val_dialogs4, train_val_dialogs5, test_dialogs1, test_dialogs2, test_dialogs3, test_dialogs4, test_dialogs5, Ses01_list, Ses02_list, Ses03_list, Ses04_list, Ses05_list = construct_train_test(emo_dict, dialogs_edit)
    else:
        train_val_dialogs1, train_val_dialogs2, train_val_dialogs3, train_val_dialogs4, train_val_dialogs5, test_dialogs1, test_dialogs2, test_dialogs3, test_dialogs4, test_dialogs5, Ses01_list, Ses02_list, Ses03_list, Ses04_list, Ses05_list = construct_train_test(emo_dict, dialogs)
    
    if args.model_num == 1:
        X_train_val = [dialog2features(s) for s in train_val_dialogs1]
        y_train_val = [dialog2labels(s) for s in train_val_dialogs1]
        
        # Create a list where train data indices are -1 and validation data indices are 0
        # train: 2, 3, 4
        # val: 5
        split_index = np.ones(len(X_train_val)) * (-1)
        split_index[len(Ses02_list+Ses03_list+Ses04_list):] = 0
    elif args.model_num == 2:
        X_train_val = [dialog2features(s) for s in train_val_dialogs2]
        y_train_val = [dialog2labels(s) for s in train_val_dialogs2]
        
        # Create a list where train data indices are -1 and validation data indices are 0
        # train: 1, 3, 5
        # val: 4
        split_index = np.ones(len(X_train_val)) * (-1)
        split_index[len(Ses01_list+Ses03_list):len(Ses01_list+Ses03_list+Ses04_list)] = 0
    elif args.model_num == 3:
        X_train_val = [dialog2features(s) for s in train_val_dialogs3]
        y_train_val = [dialog2labels(s) for s in train_val_dialogs3]
        
        # Create a list where train data indices are -1 and validation data indices are 0
        # train: 1, 4, 5
        # val: 2
        split_index = np.ones(len(X_train_val)) * (-1)
        split_index[len(Ses01_list):len(Ses01_list+Ses02_list)] = 0
    elif args.model_num == 4:
        X_train_val = [dialog2features(s) for s in train_val_dialogs4]
        y_train_val = [dialog2labels(s) for s in train_val_dialogs4]
        
        # Create a list where train data indices are -1 and validation data indices are 0
        # train: 2, 3, 5
        # val: 1
        split_index = np.ones(len(X_train_val)) * (-1)
        split_index[:len(Ses01_list)] = 0
    elif args.model_num == 5:
        X_train_val = [dialog2features(s) for s in train_val_dialogs5]
        y_train_val = [dialog2labels(s) for s in train_val_dialogs5]
        
        # Create a list where train data indices are -1 and validation data indices are 0
        # train: 1, 2, 4
        # val: 3
        split_index = np.ones(len(X_train_val)) * (-1)
        split_index[len(Ses01_list+Ses02_list):len(Ses01_list+Ses02_list+Ses03_list)] = 0
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
    best_params = s_CV.best_params_
    print('c1:', best_params['c1'])
    print('c2:', best_params['c2'])
    print('num_memories:', best_params['num_memories'])
    print('epsilon:', best_params['epsilon'])
    print('period:', best_params['period'])
    print('delta:', best_params['delta'])
    print('linesearch:', best_params['linesearch'])
    print('max_linesearch:', best_params['max_linesearch'])

    joblib.dump(crf, 'model/len_' + str(args.data_len) + '/' + args.dataset + '/Ses0'+str(args.model_num)+'.model')
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