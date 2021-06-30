import numpy as np
import matplotlib.pyplot as plt

from itertools import chain

import sklearn
import scipy.stats
import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter

import utils
import random

from argparse import RawTextHelpFormatter
import argparse

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
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-d", "--dataset", type=str, help="which dataset to use? original or C2C or U2U", default = 'U2U')
    args = parser.parse_args()

    emo_mapping_dict = {'ang':'a', 'hap':'h', 'neu':'n', 'sad':'s', 'Start':'Start', 'End':'End', 'pre-trained':'p', 0:'ang', 1:'hap', 2:'neu', 3:'sad'}
    
    if args.dataset == 'original':
        emo_dict = joblib.load('../data/emo_all_iemocap.pkl')
    elif args.dataset == 'C2C':
        emo_dict = joblib.load('../data/C2C_4emo_all_iemocap.pkl')
    elif args.dataset == 'U2U':
        emo_dict = joblib.load('../data/U2U_4emo_all_iemocap.pkl')
    
    dialogs = joblib.load('../data/dialog_iemocap.pkl')
    dialogs_edit = joblib.load('../data/dialog_4emo_iemocap.pkl')
    out_dict = joblib.load('../data/outputs.pkl')

    #intra_emo_trans_prob_dict = utils.get_val_emo_trans_prob(emo_dict, dialogs_edit)

    if args.dataset == 'original':
        train_dialogs1, train_dialogs2, train_dialogs3, train_dialogs4, train_dialogs5, test_dialogs1, test_dialogs2, test_dialogs3, test_dialogs4, test_dialogs5 = construct_train_test(emo_dict, dialogs_edit)
    else:
        train_dialogs1, train_dialogs2, train_dialogs3, train_dialogs4, train_dialogs5, test_dialogs1, test_dialogs2, test_dialogs3, test_dialogs4, test_dialogs5 = construct_train_test(emo_dict, dialogs)

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
    
    predict_dict = {}
    
    crf1 = sklearn_crfsuite.CRF(algorithm='l2sgd', c2=50)
    crf1.fit(X1_train, y1_train)
    y1_pred = crf1.predict(X1_test)
    for i in range(0, len(y1_pred), 1):
        for j in range(0, len(y1_pred[i]), 1):
            predict_dict[test_dialogs1[i][j][0]] = y1_pred[i][j]

    crf2 = sklearn_crfsuite.CRF(algorithm='l2sgd', c2=50)
    crf2.fit(X2_train, y2_train)
    y2_pred = crf2.predict(X2_test)
    for i in range(0, len(y2_pred), 1):
        for j in range(0, len(y2_pred[i]), 1):
            predict_dict[test_dialogs2[i][j][0]] = y2_pred[i][j]

    crf3 = sklearn_crfsuite.CRF(algorithm='l2sgd', c2=50)
    crf3.fit(X3_train, y3_train)
    y3_pred = crf3.predict(X3_test)
    for i in range(0, len(y3_pred), 1):
        for j in range(0, len(y3_pred[i]), 1):
            predict_dict[test_dialogs3[i][j][0]] = y3_pred[i][j]

    crf4 = sklearn_crfsuite.CRF(algorithm='l2sgd', c2=50)
    crf4.fit(X4_train, y4_train)
    y4_pred = crf4.predict(X4_test)
    for i in range(0, len(y4_pred), 1):
        for j in range(0, len(y4_pred[i]), 1):
            predict_dict[test_dialogs4[i][j][0]] = y4_pred[i][j]

    crf5 = sklearn_crfsuite.CRF(algorithm='l2sgd', c2=50)
    crf5.fit(X5_train, y5_train)
    y5_pred = crf5.predict(X5_test)
    for i in range(0, len(y5_pred), 1):
        for j in range(0, len(y5_pred[i]), 1):
            predict_dict[test_dialogs5[i][j][0]] = y5_pred[i][j]
    
    ori_emo_dict = joblib.load('../data/emo_all_iemocap.pkl')
    label = []
    predict = []
    for utt_name in predict_dict:
        label.append(ori_emo_dict[utt_name])
        predict.append(predict_dict[utt_name])

    for i in range(0, len(predict), 1):
        if predict[i] == 'ang':
            predict[i] = 0
        elif predict[i] == 'hap':
            predict[i] = 1
        elif predict[i] == 'neu':
            predict[i] = 2
        elif predict[i] == 'sad':
            predict[i] = 3
        
        if label[i] == 'ang':
            label[i] = 0
        elif label[i] == 'hap':
            label[i] = 1
        elif label[i] == 'neu':
            label[i] = 2
        elif label[i] == 'sad':
            label[i] = 3
        else:
            label[i] = -1
    
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