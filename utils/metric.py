import numpy as np
import torch

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def calibration(true_labels, pred_labels, confidences, num_bins=15):
    """Collects predictions into bins used to draw a reliability diagram.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins

    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.

    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.

    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=float)
    bin_confidences = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return { "accuracies": bin_accuracies, 
             "confidences": bin_confidences, 
             "counts": bin_counts, 
             "bins": bins,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "expected_calibration_error": ece,
             "max_calibration_error": mce }

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

def f1_score_metric(true_labels, pred_labels, num_classes):
    """Calculates macro-averaged F1 score."""
    return f1_score(true_labels, pred_labels, average='macro')

def g_mean_metric(true_labels, pred_labels, num_classes):
    """Calculates Geometric Mean of sensitivity (recall) per class."""
    cm = confusion_matrix(true_labels, pred_labels, labels=range(num_classes))
    # Sensitivity per class: TP / (TP + FN)
    # cm[i, i] is TP for class i
    # sum(cm[i, :]) is TP + FN for class i (total actual positives for class i)
    
    sensitivities = []
    for i in range(num_classes):
        total_actual = np.sum(cm[i, :])
        if total_actual > 0:
            sensitivity = cm[i, i] / total_actual
            sensitivities.append(sensitivity)
        else:
            # Handle case where a class is not present in true_labels
            # This might happen in small batches or if test set is small
            sensitivities.append(0.0) 
            
    # Geometric mean
    if not sensitivities:
        return 0.0
    
    g_mean = np.exp(np.mean(np.log(np.array(sensitivities) + 1e-10))) # Add epsilon to avoid log(0)
    return g_mean

def auc_metric(true_labels, pred_probs, num_classes):
    """Calculates AUC score."""
    try:
        if num_classes == 2:
            # Binary case: pred_probs should be probability of positive class
            # Assuming class 1 is positive
            return roc_auc_score(true_labels, pred_probs[:, 1])
        else:
            # Multi-class case: One-vs-Rest
            return roc_auc_score(true_labels, pred_probs, multi_class='ovr', average='macro')
    except ValueError:
        # Handle cases where only one class is present in true_labels
        return 0.0