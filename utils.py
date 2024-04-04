import numpy as np
import torch
import torch.nn as nn


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

def top1_accuracy(prediction, labels):
    """
    prediction: Batch_size x  num_class
    labels: Batch_size x num_class 
    """
    with torch.no_grad():
        maxk = 1
        batch_size = labels.size(0)

        _, pred = prediction.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)

        return correct_k.mul_(100.0 / batch_size)
def accuracy(prediction, labels):
    with torch.no_grad():
        prediction_argmax = torch.argmax(prediction, axis=1)
        labels_argmax = torch.argmax(labels, axis=1)
        return torch.sum(prediction_argmax == labels_argmax) / prediction_argmax.shape[0]

def num_correct(prediction, labels):
    with torch.no_grad():
        prediction_argmax = torch.argmax(prediction, axis=1)
        labels_argmax = torch.argmax(labels, axis=1)
        return torch.sum(prediction_argmax == labels_argmax)
    

def precision(prediction, labels):
    """
    prediction: Batch_size x  num_class
    labels: Batch_size x num_class 
    """

    prediction_top_class = np.argmax(prediction, axis=1)
    labels_top_class = np.argmax(labels, axis=1)
    TP = ((prediction_top_class == 1) & (labels_top_class == 1)).sum()
    FP = ((prediction_top_class == 1) & (labels_top_class == 0)).sum()
    precision = TP / (TP + FP)
    return precision
    

def f1_score(prediction, labels):
    """
    prediction: Batch_size x  num_class
    labels: Batch_size x num_class 
    """
    pass