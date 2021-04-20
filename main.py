import numpy as np
import matplotlib.pyplot as plt

from itertools import chain

import nltk
import sklearn
import scipy.stats
import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import utils

plt.style.use('ggplot')\

def utt2features(dialog, i):
    current_ses = dialog[i][0][:5] #Ses01
    current_spk = dialog[i][1]
    current_emo = dialog[i][2]
    previous_emo = ""

    # find this speaker's emotion in this dialog
    for index in range(i, -1, -1):
        previous_spk = dialog[index][1]
        if previous_spk == current_spk:
            previous_emo = dialog[index][2]
            break
    if previous_emo == "":
        previous_emo = "Start"
    features = {
        'emo_transition_prob_intra': intra_emo_trans_prob_dict[current_ses][emo_mapping_dict[previous_emo] + '2' + emo_mapping_dict[current_emo]],
        'pre-trained classifier predicted emo': emo_mapping_dict[np.argmax(out_dict[dialog[i][0]])],
        'pre-trained classifier predicted emo val': np.max(out_dict[dialog[i][0]])
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
    emo_mapping_dict = {'ang':'a', 'hap':'h', 'neu':'n', 'sad':'s', 'Start':'Start', 'End':'End', 'pre-trained':'p', 0:'ang', 1:'hap', 2:'neu', 3:'sad'}
    emo_dict = joblib.load('data/emo_all_iemocap.pkl')
    dialogs = joblib.load('data/dialog_iemocap.pkl')
    dialogs_edit = joblib.load('data/dialog_4emo_iemocap.pkl')
    out_dict = joblib.load('data/outputs.pkl')

    intra_emo_trans_prob_dict = utils.get_val_emo_trans_prob(emo_dict, dialogs_edit)

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
    
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    crf.fit(X1_train, y1_train)
    y1_pred = crf.predict(X1_test)
    for sub_list in y1_pred:
        predict += sub_list
    joblib.dump(crf, 'model/Ses01.sav')

    crf.fit(X2_train, y2_train)
    y2_pred = crf.predict(X2_test)
    for sub_list in y2_pred:
        predict += sub_list
    joblib.dump(crf, 'model/Ses02.sav')

    crf.fit(X3_train, y3_train)
    y3_pred = crf.predict(X3_test)
    for sub_list in y3_pred:
        predict += sub_list
    joblib.dump(crf, 'model/Ses03.sav')

    crf.fit(X4_train, y4_train)
    y4_pred = crf.predict(X4_test)
    for sub_list in y4_pred:
        predict += sub_list
    joblib.dump(crf, 'model/Ses04.sav')

    crf.fit(X5_train, y5_train)
    y5_pred = crf.predict(X5_test)
    for sub_list in y5_pred:
        predict += sub_list
    joblib.dump(crf, 'model/Ses05.sav')

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

    uar, acc, conf = utils.evaluate(predict, label)
    print('UAR:', uar)
    print('ACC:', acc)
    print(conf)