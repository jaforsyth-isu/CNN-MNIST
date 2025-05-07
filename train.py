import torch
import torch.nn as nn
import matplotlib

from dataset import mnist
from model import MNISTModel
from test import test

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def convert_labels(labels: torch.tensor, batch_size: int) -> torch.tensor:
    #Convert labels to a nx10x1 tensor
    converted_labels = torch.zeros(batch_size, 10)
    for d in range(labels.shape[0]):
        converted_labels[d][labels[d]] = 1
    return converted_labels

def train(model: MNISTModel,
          epochs: int,
          batch_size:int,
          optimizer: torch.optim,
          criterion: nn,
          scheduler: torch.optim.lr_scheduler,
          use_cuda=True,
          show_charts=True,
          chart_interval=100):

    #Check run conditions
    if epochs <= 0:
        print("Error: epochs must be a positive value")
        exit(1)

    if batch_size <= 0:
        print("Error: batch_size must be a positive value")
        exit(1)

    if chart_interval <= 0:
        print("Error: chart_interval must be a positive value")
        exit(1)

    #MNIST training data and labels
    data_set = mnist(train=True, batch_size=batch_size)

    #Send model to GPU
    if use_cuda and torch.cuda.is_available():
        model.cuda()
    else:
        use_cuda = False

    #Stored values for charts
    loss_values,accuracy_values, x1, x2 =[],[],[],[]

    for epoch in range(epochs):
        model.train()

        # Running_loss keeps track of loss after each step
            #and averages loss at the print_rate
        running_loss = 0.0

        #Reshuffles dataset for each epoch
        training_data = mnist(train = True, batch_size=batch_size)

        #Optimization step for loop
        for i, (batch, labels) in enumerate(training_data):
            labels = convert_labels(labels, batch_size)

            if use_cuda:
                #Move batch and labels to GPU
                batch = batch.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(input=batch)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            #Print progress in console
            if i % chart_interval == chart_interval-1:
                loss_avg = running_loss/chart_interval
                loss_values.append(loss_avg)
                x1.append(len(x1))

                print(f"Epoch {epoch+1} | Steps: {i+1:<3} | Loss: {loss_avg:.6f}")

                running_loss = 0
                scheduler.step(loss_avg)

        #Evaluate the models accuracy using test data
        accuracy = test(model=model, use_cuda=use_cuda)
        accuracy_values.append(accuracy)
        x2.append(epoch*20)

        #Print accuracy progress
        print(f"Epoch {epoch+1} | Accuracy: {accuracy:.5f}")

    if show_charts:
        plt.subplot(2, 1, 1)
        plt.plot(x1, loss_values)
        plt.title(f"Average Loss Over {chart_interval} Steps")
        plt.subplot(2, 1, 2)
        plt.plot(x2, accuracy_values)
        plt.title("Accuracy Progress")
        plt.show()
