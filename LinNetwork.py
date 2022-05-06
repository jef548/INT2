from operator import mod
from turtle import color
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from torchvision import utils
import random
import time 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    #transform = transforms.Compose(
        #[transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform = transforms.ToTensor()

    batch_size = 4


    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    labels_map = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=2)

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(32*32*3, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits


    model = NeuralNetwork() #generates original nn
    model = torch.load("model3.pth") #loads one of the saved models
    model.to(device=device)

    learning_rate = 1e-3 #how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training
    batch_size = 64 #the number of data samples propagated through the network before the parameters are updated
    epochs = 5 #the number times to iterate over the dataset

    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test_loop(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        counter = 0
        rand = 0
        with torch.no_grad():
            for X, y in dataloader:
                counter += 1
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                #print(pred.argmax(1).type(torch.float).sum().item())
                #print(y[0])
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                """this if statement will open a window, choose a random piece of 
                data from the test set, show it with the correct label in red and
                then the prediction in black."""
                """if counter == 32:
                    rand = random.randint(0,63) # random 0 to 63 as there are 64 images in each batch
                    plt.imshow(utils.make_grid(X[rand]).permute(1, 2, 0))
                    plt.title(labels_map[pred.argmax(1)[rand].type(torch.float).sum().item()])
                    plt.suptitle(labels_map[y[rand].item()], color="red")
                    plt.axis("off")
                    plt.show()"""



        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        
        
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)

    epochs = 10

    #set up timing mechanism
    start_time = time.time()
    seconds = 2
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time > seconds: 
            break

        #for i in range(epochs):
            #train_loop(trainloader, model, loss_fn, optimizer)
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda() #moves inputs and labels onto the GPU
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    #torch.save(model, "model13.pth")
