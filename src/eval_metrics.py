import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label==1) & (predicted_label==1)))
    tn = float(np.sum((true_label==0) & (predicted_label==0)))
    p = float(np.sum(true_label==1))
    n = float(np.sum(true_label==0))

    return (tp * (n/p) +tn) / (2*n)


def eval_mosei_senti(results, truths, results_weighted, truths_rounded, exclude_zero=False):
    # test_preds pair with test_truth_rounded
    # test_preds_weighted pair with test_truth
    # Acc a7, a5: test_preds
    # F1, Acc2: test_preds_weighted
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()
    test_preds_weighted = results_weighted.view(-1).cpu().detach().numpy()
    test_truth_rounded = truths_rounded.view(-1).cpu().detach().numpy()
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth_rounded, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth_rounded, a_min=-2., a_max=2.)
    mae = np.mean(np.absolute(test_preds_weighted - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds_weighted, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score((test_preds_weighted[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds_weighted[non_zeros] > 0)
    if results.shape[0] == 4643:
        print("mult_acc_7: ", mult_a7.round(3))
        print("MAE: ", mae.round(3), "Correlation Coefficient: ", corr.round(3), "mult_acc_7: ", mult_a7.round(3), \
          "mult_acc_5: ", mult_a5.round(3), "F1 score: ", f_score.round(3), "Accuracy: ", \
          accuracy_score(binary_truth, binary_preds).round(3))

    return mult_a7, mult_a5


def eval_mosei_emo(results, truths, exclude_zero=False):
    emo = ["emo1, emo2, emo3, emo4, emo5, emo6"]
    f1_total = []
    acc_total = []
    test_preds = results.cpu().detach().numpy()
    test_truth = truths.cpu().detach().numpy()
    for emo_ind in range(6):
        test_preds_i = test_preds[:,emo_ind]
        test_truth_i = test_truth[:,emo_ind]
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        f1_total.append(f1)
        acc_total.append(acc)
    return acc_total, f1_total

def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)


def eval_iemocap(results, truths):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    f1_total = []
    acc_total = []
    test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
    test_truth = truths.view(-1, 4).cpu().detach().numpy()
    for emo_ind in range(4):
        test_preds_i = np.argmax(test_preds[:,emo_ind], axis=1)
        test_truth_i = test_truth[:,emo_ind]
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        f1_total.append(f1)
        acc_total.append(acc)

    return acc_total, f1_total
