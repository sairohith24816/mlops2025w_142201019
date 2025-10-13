import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
import random
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_data(dataset_name, subset_ratio=1.0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset_name == "cifar10":
        trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10('./data', train=False, transform=transform)
    else:  # cifar100
        trainset = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100('./data', train=False, transform=transform)
    
    if subset_ratio < 1.0:
        subset_size = int(len(trainset) * subset_ratio)
        subset_indices = torch.randperm(len(trainset))[:subset_size]
        trainset = torch.utils.data.Subset(trainset, subset_indices)
        print(f"Using {subset_size}/{len(trainset.dataset)} samples ({subset_ratio*100:.1f}%) for training.")
    else:
        print(f"Using full training dataset: {len(trainset)} samples.")
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    
    return trainloader, testloader


def train(model, trainloader, testloader, epochs, dataset_name, subset_ratio=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total
        
        wandb.log({
            f"epoch": epoch + 1,
            f"train_acc": train_acc,
            f"test_acc": test_acc,
            f"train_loss": train_loss / len(trainloader),
            f"test_loss": test_loss / len(testloader),
        })
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return model


# Scenario A: CIFAR-100 → CIFAR-10
wandb.init(project="Q4-sequential-cifar-new", name="Scenario_A_CIFAR100_to_CIFAR10")

print("Scenario A: CIFAR-100, CIFAR-10")
cifar100_train, cifar100_test = get_data("cifar100", subset_ratio=0.01)
model = SimpleCNN(100)
model = train(model, cifar100_train, cifar100_test, epochs=100, dataset_name="CIFAR100", subset_ratio=0.01)

cifar10_train, cifar10_test = get_data("cifar10", subset_ratio=0.01)
model.classifier[-1] = nn.Linear(256, 10)
model = train(model, cifar10_train, cifar10_test, epochs=100, dataset_name="CIFAR10", subset_ratio=0.01)

wandb.finish()

# Scenario B: CIFAR-10 → CIFAR-100
wandb.init(project="Q4-sequential-cifar-new", name="Scenario_B_CIFAR10_to_CIFAR100")

print("\nScenario B: CIFAR-10, CIFAR-100")
cifar10_train, cifar10_test = get_data("cifar10", subset_ratio=0.01)
model = SimpleCNN(10)
model = train(model, cifar10_train, cifar10_test, epochs=100, dataset_name="CIFAR10", subset_ratio=0.01)

cifar100_train, cifar100_test = get_data("cifar100", subset_ratio=0.01)
model.classifier[-1] = nn.Linear(256, 100)
model = train(model, cifar100_train, cifar100_test, epochs=100, dataset_name="CIFAR100", subset_ratio=0.1)

wandb.finish()