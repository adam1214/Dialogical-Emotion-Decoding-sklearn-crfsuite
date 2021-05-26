import sklearn_crfsuite
import joblib
import utils

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

def dialog2labels(dialog):
    return [emo for utt, spk, emo in dialog]

if __name__ == "__main__":
    emo_dict = joblib.load('data/emo_all_iemocap.pkl')
    dialogs = joblib.load('data/dialog_iemocap.pkl')
    dialogs_edit = joblib.load('data/dialog_4emo_iemocap.pkl')
    out_dict = joblib.load('data/outputs.pkl')

    train_dialogs1, train_dialogs2, train_dialogs3, train_dialogs4, train_dialogs5, test_dialogs1, test_dialogs2, test_dialogs3, test_dialogs4, test_dialogs5 = construct_train_test(emo_dict, dialogs_edit)
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
    crf1 = joblib.load('model/Ses01.model')
    crf2 = joblib.load('model/Ses02.model')
    crf3 = joblib.load('model/Ses03.model')
    crf4 = joblib.load('model/Ses04.model')
    crf5 = joblib.load('model/Ses05.model')

    y1_pred = crf1.predict(X1_test)
    for sub_list in y1_pred:
        predict += sub_list
    
    y2_pred = crf2.predict(X2_test)
    for sub_list in y2_pred:
        predict += sub_list

    y3_pred = crf3.predict(X3_test)
    for sub_list in y3_pred:
        predict += sub_list

    y4_pred = crf4.predict(X4_test)
    for sub_list in y4_pred:
        predict += sub_list

    y5_pred = crf5.predict(X5_test)
    for sub_list in y5_pred:
        predict += sub_list

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
    