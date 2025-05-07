import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from dataset import mnist
from model import MNISTModel

def test(model: MNISTModel, use_cuda=True) -> float:
    #Evaluate the accuracy of the model on MNIST test data

    model.eval()

    #Check if cuda GPU is available and send model to GPU
    if use_cuda and not next(model.parameters()).is_cuda and torch.cuda.is_available():
        model.cuda()

    actual = []
    pred = []

    data_set = mnist(train=False, batch_size=500)
    with torch.no_grad():
        for batch, labels in data_set:

            #Send batch and labels to GPU
            if use_cuda and torch.cuda.is_available():
                batch = batch.cuda()
                labels = labels.cuda()

            out = model(input=batch)
            _, predicted = torch.max(out.data,1)

            pred.extend((predicted.flatten().tolist()))
            actual.extend(labels.flatten().tolist())

    accuracy = accuracy_score(actual, pred)
    return accuracy

