# mnist_train_eval.py
# Andrew Skow
# Deep Learning For Computer Vision EN.525.733
# March 2, 2025

import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchmetrics.classification import MulticlassConfusionMatrix
from torch.utils.data import DataLoader, random_split
import AlexNet
import AlexNet_No_L5
import AlexNet_No_L4
import AlexNet_No_FC1

def create_alexnet_model(alex_net_version, num_classes):
    '''
    Returns an alexnet model based on the name
    
    Param (alex_next_version): he alexnet variant to evaluate: "AlexNet", 
    "AlexNet_No_L5", "AlexNet_No_L4", "AlexNet_No_FC1" 
    Param (num_classes): Number of classes the model is trained and evaluated 
    on
    Return (model): The appropriate alexnet model
    '''

    # Initialize the model
    if alex_net_version == "AlexNet" :
        model = AlexNet.AlexNet(num_classes)
    elif alex_net_version =="AlexNet_No_L5" :
        model = AlexNet_No_L5.AlexNet(num_classes)
    elif alex_net_version == "AlexNet_No_L4" :
        model = AlexNet_No_L4.AlexNet(num_classes)
    elif alex_net_version == "AlexNet_No_FC1" :
        model = AlexNet_No_FC1.AlexNet(num_classes)
    else :
        print(f"Invalid alex net type {alex_net_version}.  Returning.")
        return
    
    return model

def train_alexnet(alex_net_version, num_classes, train_loader, val_loader, 
                  test_loader , num_epochs=20, learning_rate=1e-3, 
                  device='cpu'):
    '''
    Trains the AlexNet model.
    
    Param (alex_net_version): The alexnet variant to evaluate: "AlexNet", 
    "AlexNet_No_L5", "AlexNet_No_L4", "AlexNet_No_FC1" 
    Param (num_classes): The number of output classes
    Param (train_loader): The training data loader
    Param (val_loader): The validation data loader
    Param (test_loader): The test data loader
    Param (num_epochs): The number of training epochs
    Param (learning_rate): The learning rate for back prop
    Param (device): The device on which to train the model
    
    Return 
        (model): The trained model
        (train_losses): The history of training loss
        (val_losses): The history of validation loss
        (test_losses): The history of test loss
    '''
    # Initialize the model
    model = create_alexnet_model(alex_net_version, num_classes)
    
    # Send the model to the device
    model.to(device)
    # Define the criterion
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    momentum = 0.9
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create loss arrays
    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []

    # Begin training each epoch
    model.train()
    for epoch in range(num_epochs):
        # Set the model to train mode
        model.train()
        # Initializse the training loss
        train_loss = 0
        run_loss = 0
        correct = 0
        # Unpack the images from the.  We do not need labels
        for batch_idx, data in enumerate(train_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            # Run the model
            outputs = model(images)

            # Calculate the loss
            loss = criterion(outputs, labels)
            _, pred = torch.max(outputs.data, 1)

            # pred = output.topk(1)[1].flatten()
            correct += (pred == labels).sum().item()

            # Conduct back prop
            loss.backward()
            optimizer.step()

            # print statistics
            run_loss += loss.item()
            train_loss += loss.item()

            if batch_idx % 200 == 199:  # print every 200 mini-batches
                print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {run_loss / 200:.3f}')
                run_loss = 0.0


        # Save the accuracy
        train_accuracy = correct / len(train_loader.dataset)
        train_accuracies.append(train_accuracy)

        # Normalize the training loss
        train_loss /= len(train_loader)
        # Add results to the training loss
        train_losses.append(train_loss)
    
        # Run validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            # Load the validation images with no labels
            for images, _ in val_loader:
                images = images.to(device)

                # Run the model
                outputs = model(images)

                # Conduct back prop
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Obtain the predicted class
                _, pred = torch.max(outputs.data, 1)

                correct += (pred == labels).sum().item()
        
        # Save the accuracy
        val_accuracy = correct / len(val_loader.dataset)
        val_accuracies.append(val_accuracy)

        # Normalize the loss
        val_loss  /= len(val_loader)
        val_losses.append(val_loss)
    print(f'Val Loss: {val_loss:.4f}')

        # Run test
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            # Load the validation images with no labels
            for images, _ in test_loader:
                images = images.to(device)

                # Run the model
                outputs = model(images)

                # Conduct back prop
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Obtain the predicted class
                _, pred = torch.max(outputs.data, 1)

                correct += (pred == labels).sum().item()
        
        # Save the accuracy
        test_accuracy = correct / len(test_loader.dataset)
        test_accuracies.append(test_accuracy)

        # Normalize the loss
        test_loss  /= len(test_loader)
        test_losses.append(test_loss)
    print(f'Test Loss: {test_loss:.4f}')

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

        print('\nAccuracy: {}/{} ({:.06f}%)\n'.format(
            correct, len(test_loader.dataset),
            test_accuracy * 100))
        
        # Save the model
        torch.save(
            model.state_dict(), 
            f"../results/{alex_net_version}/weights/model_epoch_{epoch}.pt"
        )
    
    return model, train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies

def test(model, classes, test_loader, device='cpu') :
    '''
    This function will run forward propagation and compare to the expected output

    Param (model): the torch model of the nn
    Param (test_loader): The torch test data loader
    Return (return predictions): The lost of model predictions of class 
    Return (truth): The true class of the images
    Return (class_confidence): The class confidences output by the model
    Return (outputs)L: The actua model outputs
    Return (label_history): The string history of the labels
    '''
    # Define the loss function
    loss_func = nn.CrossEntropyLoss()

    # prepare to count predictions for each class
    class_confidence = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # Initialize class predictions 
    for classname in classes:
        class_confidence[classname] = []
    # predictions and expected labels
    predictions = []
    truth =[]

    output_list = []
    labels_list = []

    # switch model to eval model (dropout becomes pass through)
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)

            # Store the outputs and the labels
            output_list.append(output.cpu().numpy())
            labels_list.append(labels.cpu().numpy())


            test_loss += loss_func(output, labels).item() * inputs.shape[0]
            _, pred = torch.max(output.data, 1)

            # Add the confidence values to the list
            for confidence in output.data:
                for i, conf in enumerate(confidence):
                    class_confidence[classes[i]].append(float(conf.cpu()))

            # collect the correct predictions for each class
            for label, prediction in zip(labels, pred):
                # Store the prediction and ground truth
                predictions.append(prediction.cpu())
                truth.append(label.cpu())
                # Count the prediction results
                total_pred[classes[label]] += 1

            # pred = output.topk(1)[1].flatten()
            correct += (pred == labels).sum().item()

    # Format the output and labels lists
    outputs = np.vstack(output_list)
    label_history = np.concatenate(labels_list)

    test_loss /= len(test_loader.dataset)
    print('Accuracy: {}/{} ({:.06f}%)'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # print accuracy for each class
    return predictions, truth, class_confidence, outputs, label_history

def confusion_matrix(predictions, truth, name):
    '''
    confusion_matrix
    Produces the confusion matrix from the inference results of the model
    
    Param (predictions): The inference output
    Param (truth): The expected inference output
    Param (name): The name of the model
    '''
    # Produce the confusion matrix
    metric = MulticlassConfusionMatrix(num_classes=10)
    metric.update(torch.tensor(predictions), torch.tensor(truth))
    fig_, ax_ = metric.plot()
    fig_.savefig(f'../results/{name}/conf_mat.png',dpi=300)
    print("confusion matrix saved")

def prec_recall(confidence_dict, labels, classes, name):
    '''
    prec_recall
    Produces two dictionaries for precision and recall from the complete
    predictions dictionary from inference.
    
    Param (confidence_dict): Confidence values for each class for each image of inference
    Param (lables): For each image in the test set
    Param (classes): List of the image classes
    Param (name): Label for the file name
    Return (precision) A dictionary of precision values for each class over a range of thresholds
    Return (recall) A dictionary of recall values for each class over a range of thresholds
    '''
    # Defince precision and recall dictionaries 
    precisions = {classname: 0 for classname in classes}
    recalls = {classname: 0 for classname in classes}
    # Initialize the dictionaries with lists
    for classname in classes:
        precisions[classname] = []
        recalls[classname] = []

    # Calculate the threshold range 
    min = np.min(confidence_dict['plane'])
    max = np.max(confidence_dict['plane'])
    step = (max - min)/100

    # Loop through each threshold value
    thresh = min
    while thresh <= max:
        # Loop through each class
        for classname in classes:
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            true_negatives = 0
            # Obtain the confidence values for that class
            confidences = confidence_dict[classname]
            # Loop through each confidence
            for i, conf in enumerate(confidences):
                # Check thresh
                if conf > thresh:
                    is_positive = True
                else:
                    is_positive = False
                
                # Check if it is a false positive 
                if classes[labels[i]] == classname:
                    if is_positive:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if is_positive:
                        false_positives += 1
                    else:
                        true_negatives += 1
            # calculate precision and recall for the class
            # make sure we do not divide by zero
            if (true_positives + false_positives) == 0:
                # print("In the zero check for precision")
                precisions[classname].append(1.0)
            else:
                precisions[classname].append(float(true_positives)/(true_positives + false_positives))
            if (true_positives + false_negatives) == 0:
                # print("In the zero check for recall")
                recalls[classname].append(1.0)
            else:
                recalls[classname].append(float(true_positives)/(true_positives + false_negatives))
        
        thresh += step
    
    plt.figure()
    for classname in classes:
        plt.plot(recalls[classname], precisions[classname], lw=2, label=classname)
    
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title(f"{name} precision vs. recall curve")
    plt.grid(True)
    plt.savefig(f"../results/{name}/precision_recall.png")
    plt.cla()

    return precisions, recalls


    
def plot_learning_curve(train_losses, val_losses, test_losses, alexnet_version): 
    '''
    Plots the results of the model training.
    
    Param (train_losses): The array of training loss history
    Param (val_losses): The aray of validation loss history
    Param (alexnet_version): The type of alex net being evaluated
    '''
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.title(f"Learning curve (hidden size: {alexnet_version})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../results/{alexnet_version}/learning_curve.png")
    plt.cla

def plot_accuracy(train_accuracy, val_accuracy, test_accuracy, alexnet_version): 
    '''
    Plots the results of the model training.
    
    Param (train_losses): The array of training accuracy history
    Param (val_losses): The aray of validation accuracy history
    Param (alexnet_version): The type of alex net being evaluated
    '''
    plt.figure()
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.plot(test_accuracy, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Acuracy')
    plt.title(f"Classification Accuracy {alexnet_version})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../results/{alexnet_version}/accuracy.png")
    plt.cla

def main():
    # Load the data
    image_dim = 227
    transform = transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(image_dim),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Download and load MNIST dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        transform=transform, 
        download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        transform=transform, 
        download=True
    )

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Define hyperparameters
    batch_size = 16
    epochs = 100
    learning_rate = 1e-4
    

    # # Split training dataset into training (80%) and validation (20%)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Engage hardware acceleration
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            mps_device = torch.device("mps")
            x = torch.ones(1, device=mps_device)
            print(x)
    else:
        print("MPS device not found.")
        mps_device = torch.device("cpu")

    # Define the different hidden sizes that we want to evaluate
    alexnet_versions = ["AlexNet", "AlexNet_No_L5", "AlexNet_No_L4", "AlexNet_No_FC1"]
    trained_models = {}

    # Train the different AlexNets
    for alexnet_version in alexnet_versions:
        model_filename = f"../results/{alexnet_version}/weights/model_final.pt"

        # Check if model file exists
        if os.path.exists(model_filename):
            print(f"\nLoading pre-trained model with Hidden Size: {alexnet_version}")
            model = model = create_alexnet_model(alexnet_version, len(classes))
            model.to(mps_device)
            model.load_state_dict(torch.load(model_filename))
        else:
            print(f"\nTraining AlexNet version: {alexnet_version}")
            # train the model
            training_output = train_alexnet(alexnet_version, 
                                            len(classes),
                                            train_loader, 
                                            val_loader,
                                            test_loader,
                                            num_epochs=epochs,
                                            learning_rate=learning_rate,
                                            device=mps_device)
            model, tr_loss, v_loss, t_loss, tr_ac, v_ac, t_ac = training_output
            # Plot the learning curve
            plot_learning_curve(tr_loss, v_loss, t_loss, alexnet_version)
            plot_accuracy(tr_ac, v_ac, t_ac, alexnet_version)
            # Save the model
            torch.save(model.state_dict(), f"../results/{alexnet_version}/weights/model_final.pt")
            print(f"Model saved as {model_filename}")

        trained_models[alexnet_version] = model.eval()

    print("Evauating the models")
    for alexnet_version in alexnet_versions:

        model = trained_models[alexnet_version]
        # Test the models 
        print(f"Running inference on {alexnet_version}")
        an_predictions, an_truth, an_confidence, an_outputs, an_labels= test(model, classes, test_loader, mps_device)

        print(f"Making the confusion matrix {alexnet_version}")
        confusion_matrix(an_predictions, an_truth, f"{alexnet_version}")

        # Calculate precision and recall
        print(f"Producing PR for {alexnet_version}")
        an_precision, an_recall = prec_recall(an_confidence, an_truth, classes, alexnet_version)

        print(f"Saving PR files {alexnet_version}")
        with open(f"../results/{alexnet_version}/precision.json", "w") as outfile: 
            json.dump(an_precision, outfile)
        with open(f"../results/{alexnet_version}/recall.json", "w") as outfile: 
            json.dump(an_recall, outfile)

if __name__ == "__main__":
    main()