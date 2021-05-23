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
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter
import utils
from time import time

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
    train_dialogs1 = Ses02_list + Ses03_list + Ses04_list + Ses05_list
    test_dialogs1 = Ses01_list

    train_dialogs2 = Ses01_list + Ses03_list + Ses04_list + Ses05_list
    test_dialogs2 = Ses02_list

    train_dialogs3 = Ses01_list + Ses02_list + Ses04_list + Ses05_list
    test_dialogs3 = Ses03_list

    train_dialogs4 = Ses01_list + Ses02_list + Ses03_list + Ses05_list
    test_dialogs4 = Ses04_list

    train_dialogs5 = Ses01_list + Ses02_list + Ses03_list + Ses04_list
    test_dialogs5 = Ses05_list

    return train_dialogs1, train_dialogs2, train_dialogs3, train_dialogs4, train_dialogs5, test_dialogs1, test_dialogs2, test_dialogs3, test_dialogs4, test_dialogs5

if __name__ == "__main__":
    start_t = time()
    emo_mapping_dict = {'ang':'a', 'hap':'h', 'neu':'n', 'sad':'s', 'Start':'Start', 'End':'End', 'pre-trained':'p', 0:'ang', 1:'hap', 2:'neu', 3:'sad'}
    emo_dict = joblib.load('data/emo_all_iemocap.pkl')
    dialogs = joblib.load('data/dialog_iemocap.pkl')
    dialogs_edit = joblib.load('data/dialog_4emo_iemocap.pkl')
    out_dict = joblib.load('data/outputs.pkl')

    train_dialogs1, train_dialogs2, train_dialogs3, train_dialogs4, train_dialogs5, test_dialogs1, test_dialogs2, test_dialogs3, test_dialogs4, test_dialogs5 = construct_train_test(emo_dict, dialogs_edit)

    X1_train = [dialog2features(s) for s in train_dialogs1]
    X2_train = [dialog2features(s) for s in train_dialogs2]
    X3_train = [dialog2features(s) for s in train_dialogs3]
    X4_train = [dialog2features(s) for s in train_dialogs4]
    X5_train = [dialog2features(s) for s in train_dialogs5]

    y1_train = [dialog2labels(s) for s in train_dialogs1]
    y2_train = [dialog2labels(s) for s in train_dialogs2]
    y3_train = [dialog2labels(s) for s in train_dialogs3]
    y4_train = [dialog2labels(s) for s in train_dialogs4]
    y5_train = [dialog2labels(s) for s in train_dialogs5]

    X1_test = [dialog2features(s) for s in test_dialogs1]
    X2_test = [dialog2features(s) for s in test_dialogs2]
    X3_test = [dialog2features(s) for s in test_dialogs3]
    X4_test = [dialog2features(s) for s in test_dialogs4]
    X5_test = [dialog2features(s) for s in test_dialogs5]

    y1_test = [dialog2labels(s) for s in test_dialogs1]
    y2_test = [dialog2labels(s) for s in test_dialogs2]
    y3_test = [dialog2labels(s) for s in test_dialogs3]
    y4_test = [dialog2labels(s) for s in test_dialogs4]
    y5_test = [dialog2labels(s) for s in test_dialogs5]

    predict = []
    '''
    parameters
    c2: default = 1.0
    max_iterations: default=1000
    period:The duration of iterations to test the stopping criterion. default = 10
    delta:The threshold for the stopping criterion; an iteration stops when the improvement of the log likelihood over the last period iterations is no greater than this threshold. default = 1e-5
    calibration_eta:The initial value of learning rate (eta) used for calibration. default=0.1
    calibration_rate:The rate of increase/decrease of learning rate for calibration. default=2.0
    calibration_samples:The number of instances used for calibration. The calibration routine randomly chooses instances no larger than calibration_samples. default=1000
    calibration_candidates:The number of candidates of learning rate. The calibration routine terminates after finding calibration_samples candidates of learning rates that can increase log-likelihood. default=10
    calibration_max_trials:The maximum number of trials of learning rates for calibration. The calibration routine terminates after trying calibration_max_trials candidate values of learning rates. default=20
    '''
    crf = sklearn_crfsuite.CRF(algorithm='l2sgd') 
    params_space = {
        'c2': np.linspace(start = 0.01, stop = 1,  endpoint=True, num = 10),
        'max_iterations': range(800, 1201, 100),
        'period': range(8, 13, 1),
        'delta': np.linspace(start = 0.000001, stop = 0.0001,  endpoint = True, num = 20),
        'calibration_eta': np.linspace(start = 0.001, stop = 0.1,  endpoint = True, num = 20),
        'calibration_rate': np.linspace(start = 1.0, stop = 5.0,  endpoint = True, num = 5),
        'calibration_samples': range(1000, 1501, 250),
        'calibration_candidates': [5, 10 , 15],
        'calibration_max_trials': [15, 20, 25]
    }
    # use the same metric for evaluation
    scorer = make_scorer(metrics.flat_accuracy_score)
    #s_CV = RandomizedSearchCV(crf, params_space, cv=5, verbose=1, n_jobs=-1, n_iter=200, scoring=recall_scorer, random_state=0)
    s_CV = GridSearchCV(crf, params_space, cv=5, verbose=1, n_jobs=-1, scoring=scorer)
    s_CV.fit(X1_train, y1_train)
    crf = s_CV.best_estimator_
    y1_pred = crf.predict(X1_test)
    for sub_list in y1_pred:
        predict += sub_list
    joblib.dump(crf, 'model/Ses01.sav')
    end_t = time()
    dur = (end_t-start_t)/60.
    print('Process time(1):', dur, 'min')
    
    s_CV.fit(X2_train, y2_train)
    crf = s_CV.best_estimator_
    y2_pred = crf.predict(X2_test)
    for sub_list in y2_pred:
        predict += sub_list
    joblib.dump(crf, 'model/Ses02.sav')
    end_t = time()
    dur = (end_t-start_t)/60.
    print('Process time(2):', dur, 'min')

    s_CV.fit(X3_train, y3_train)
    crf = s_CV.best_estimator_
    y3_pred = crf.predict(X3_test)
    for sub_list in y3_pred:
        predict += sub_list
    joblib.dump(crf, 'model/Ses03.sav')
    end_t = time()
    dur = (end_t-start_t)/60.
    print('Process time(3):', dur, 'min')

    s_CV.fit(X4_train, y4_train)
    crf = s_CV.best_estimator_
    y4_pred = crf.predict(X4_test)
    for sub_list in y4_pred:
        predict += sub_list
    joblib.dump(crf, 'model/Ses04.sav')
    end_t = time()
    dur = (end_t-start_t)/60.
    print('Process time(4):', dur, 'min')

    s_CV.fit(X5_train, y5_train)
    crf = s_CV.best_estimator_
    y5_pred = crf.predict(X5_test)
    for sub_list in y5_pred:
        predict += sub_list
    joblib.dump(crf, 'model/Ses05.sav')
    end_t = time()
    dur = (end_t-start_t)/60.
    print('Process time(5):', dur, 'min')

    label = []
    for _, dia in enumerate(dialogs):
        label += [utils.convert_to_index(emo_dict[utt]) for utt in dialogs[dia]]
    
    for i in range(0, len(predict), 1):
        if predict[i] == 'ang':
            predict[i] = 0
        elif predict[i] == 'hap':
            predict[i] = 1
        elif predict[i] == 'neu':
            predict[i] = 2
        elif predict[i] == 'sad':
            predict[i] = 3
    
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
    
    uar, acc, conf = utils.evaluate(predict, label)
    print('UAR:', uar)
    print('ACC:', acc)
    print(conf)
    '''
    print('===================================================') 
    print("Top likely transitions:") # transition feature coefficients {(label_from, label_to) -- coef}
    print_transitions(Counter(crf.transition_features_).most_common(len(crf.transition_features_)))
    print('===================================================')
    print("Top positive:") # state feature coefficients {(attr_name, label) -- coef}
    print_state_features(Counter(crf.state_features_).most_common(len(crf.state_features_)))
    '''
    