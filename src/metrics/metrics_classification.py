import sklearn.metrics as sklearnMetrics
from keras import backend as K
import math
import numpy as np
from scipy import stats

#1d array-like
def getPrecision(y_true, y_pred,average='weighted'): #average=None if we want metric per class
    precision = sklearnMetrics.precision_score(y_true, y_pred=y_pred, average=average) #precission per class
    #sklearnMetrics.precision_score(y_true, y_pred=pred, average='weighted') #precission in general
    return precision

def getRecall(y_true, y_pred,average='weighted'):
    recall = sklearnMetrics.recall_score(y_true, y_pred=y_pred, average=average) #precission per class
    #sklearnMetrics.recall_score(y_true, y_pred=pred, average='weighted') #precission in general
    return recall

def getConfussionMatrix(y_true, y_pred, classes):
    cm = sklearnMetrics.confusion_matrix(y_true, y_pred, classes)
    return cm

def getAccucacy(y_true, y_pred):
    accuracy = sklearnMetrics.accuracy_score(y_true, y_pred)
    return accuracy

def getF1(y_true, y_pred, average='weighted'):
    f1 = sklearnMetrics.f1_score(y_true, y_pred, average=average)
    return f1


def get_summary_metrics(train_pred, labels_train, test_pred, labels_test,classes= [1,-1], saveData=False, path2save = ""):
    precission_train = getPrecision(y_pred=train_pred, y_true=labels_train)
    recall_train = getRecall(y_pred=train_pred, y_true=labels_train)
    accuracy_train = getAccucacy(y_pred=train_pred, y_true=labels_train)
    F1_train = getF1(y_pred=train_pred, y_true=labels_train)
    confussion_matrix_train = getConfussionMatrix(y_pred=train_pred, y_true=labels_train, classes=classes)
    precission_test =getPrecision(y_pred=test_pred, y_true=labels_test)
    recall_test = getRecall(y_pred=test_pred, y_true=labels_test)
    accuracy_test = getAccucacy(y_pred=test_pred, y_true=labels_test)
    F1_test = getF1(y_pred=test_pred, y_true=labels_test)
    confussion_matrix_test = getConfussionMatrix(y_pred=test_pred, y_true=labels_test, classes=classes)
    print("....................METRICS.................................")
    print(path2save)
    print("------TRAIN-------")
    print('Precission train: ' + str(precission_train))
    print('Recall train: ' + str(recall_train))
    print("Accuracy train" + str(accuracy_train))
    print("F1 train" + str(F1_train))
    print("Confussion matrix train:" + str(confussion_matrix_train))
    print("-------TEST-------")
    print('Precission test: ' + str(precission_test))
    print('Recall test: ' + str(recall_test))
    print("Accuracy test" + str(accuracy_test))
    print("F1 test" + str(F1_test))
    print("Confussion matrix test:" + str(confussion_matrix_test))
    print("........................................................")
    if(saveData):
        if(path2save==""):
            path2save= "/home/cris/PycharmProjects/InterSpeech19/data/results/baseline/default.txt"
        with open(path2save, "w") as f:
            f.write("-----------TRAIN-------"+"\n")
            f.write("Precission: "+str(precission_train)+"\n")
            f.write("Recall: " + str(recall_train)+"\n")
            f.write("Accuracy:" + str(accuracy_train)+"\n")
            f.write("F1:" + str(F1_train)+"\n")
            f.write("Classes: " + str(classes)+"\n")
            f.write("Confussion matrix train:" + str(confussion_matrix_train)+"\n")
            f.write("-----------TEST-------"+"\n")
            f.write("Precission test: " + str(precission_test)+"\n")
            f.write("Recall test: " + str(recall_test)+"\n")
            f.write("Accuracy test" + str(accuracy_test)+"\n")
            f.write("F1 test" + str(F1_test)+"\n")
            f.write("Classes: " + str(classes) + "\n")
            f.write("Confussion matrix test:" + str(confussion_matrix_test)+"\n")
        with open("/home/cris/PycharmProjects/InterSpeech19/data/results/baseline/common_metrics.txt", "a+") as f:
            f.write(path2save+";"+str(confussion_matrix_train).replace("\n","")+";"+str(confussion_matrix_test).replace("\n","")+"\n")
    return precission_train, recall_train, accuracy_train, F1_train, confussion_matrix_train, precission_test, recall_test, accuracy_test, F1_test, confussion_matrix_test

#Previous implementations of keras functions
def keras_precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def keras_recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def keras_fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.

    Here it is only computed as a batch-wise average, not globally.
    """
    return keras_fbeta_score(y_true, y_pred, beta=1)


def keras_fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = keras_precision(y_true, y_pred)
    r = keras_recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def classification_binary_calculator(predictions, labels):
    tpos = 0
    tneg = 0
    fpos = 0
    fneg = 0
    brierScoreSum = []
    for predict, label in zip(predictions, labels):

        brierScoreSum.append((predict-label)*(predict-label))
        if np.argmax(predict):
            if np.argmax(predict) == np.argmax(label):
                tpos = tpos + 1
            else:
                fpos = fpos + 1
        else:
            if np.argmax(predict) == np.argmax(label):
                tneg = tneg + 1
            else:
                fneg = fneg + 1

    return tpos,tneg,fpos,fneg,brierScoreSum


def classification_binary_metrics(predictions, labels):
    tpos, tneg, fpos, fneg, bs = classification_binary_calculator(predictions, labels)
    metric = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'F1', 'E1', 'Informedness', 'Markedness', 'MCC',
              'DP', 'CP', 'Brier Score', 'b', 'Capacity of IDS']
    value = []
    temp = [tneg, tpos, fpos, fneg]

    print(temp)

    if np.nonzero(temp)[0].shape[0] < 3:
        print('LOG_ERROR: Model working wrong. Review data set structure')
        value = [-1]*len(metric)
        return value, metric

    # Sensitivity
    sens = float(tpos) / (tpos + fneg)
    value.append(sens)
    # Specificity
    spec = float(tneg) / (tneg + fpos)
    value.append(spec)
    # PPV
    ppv = float(tpos) / (tpos+fpos)
    value.append(ppv)
    # NPV
    npv = float(tneg) / (tneg+fneg)
    value.append(npv)
    # Accuracy
    acc = float(tpos + tneg) / (tpos + tneg + fpos + fneg)
    value.append(acc)
    # F1
    if sens==0 or ppv == 0:
        f1 = 0
    else:
        f1 = stats.hmean(np.array([sens, ppv]))
    value.append(f1)
    # E1
    e1 = (2*tneg)/(2*tneg+fpos+fneg)
    value.append(e1)
    # Informedless
    info = sens + spec - 1
    value.append(info)
    # Markedless
    mark = ppv+npv-1
    value.append(mark)
    # MCC
    mcc = np.mean(np.array([mark, info]))
    value.append(mcc)
    # DP
    if npv==0 or ppv==0 or sens==0:
        dp=0
    else:
        dp = stats.hmean(np.array([sens, ppv, npv]))
    value.append(dp)
    # CP
    if f1==0 or e1==0:
        cp=0
    else:
        cp = stats.hmean(np.array([f1,e1]))
    value.append(cp)
    # Brier Score
    bs = np.sum(bs)/len(bs)
    value.append(bs)
    # Capacity of IDS
    alpha = 1 - spec
    beta = 1 - sens
    b = float(tpos + fneg) / (tpos + tneg + fpos + fneg)

    #Completar, log base 2 y metodo para cambiar metric de 0 a 1 si hubiera
    if ppv == 0:
        ppv=1
    if npv==0:
        npv=1
    try:
        if ppv==1:
            cap = 1 - (((b * beta * math.log(1 - npv, 2)) + (
            (1 - b) * (1 - alpha) * math.log(npv, 2)) + ((1 - b) * alpha * math.log(0.000000000000000000001, 2)))
                       / ((b * math.log(b, 2)) + ((1 - b) * math.log(1 - b, 2))))
        elif npv==1:
            cap = 1 - (((b * (1 - beta) * math.log(ppv, 2)) + (b * beta * math.log(0.000000000000000000001, 2)) + ((1 - b) * alpha * math.log(1 - ppv, 2)))
                       / ((b * math.log(b, 2)) + ((1 - b) * math.log(1 - b, 2))))
        else:
            cap = 1 - (((b*(1-beta)*math.log(ppv, 2)) + (b*beta*math.log(1-npv, 2)) + ((1-b)*(1-alpha)*math.log(npv, 2)) + ((1-b)*alpha*math.log(1-ppv, 2)))
                /((b*math.log(b, 2)) + ((1-b)*math.log(1-b, 2))))
        value.append(b)
        value.append(cap)
    except ValueError:
        cap=1
        value.append(b)
        value.append(cap)

    return value, metric, cap