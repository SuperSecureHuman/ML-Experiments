# Imports
import torch
from tqdm import tqdm
from torchvision import datasets, transforms


def resnetModel(output_size, Pretrained=True, Device="cuda"):
    model = torch.hub.load('pytorch/vision:v0.6.0',
                           'resnet101', pretrained=Pretrained)
    model.fc = torch.nn.Linear(2048, output_size)
    model.to(Device)
    return model

def cifar10Dataloader(num_workers, batch_size, train, shuffle=True, data_dir="./data", pin_memory=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    return dataloader




def accuracy_check(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(testloader, unit="batch") as t2epoch:
            for data, target in t2epoch:
                t2epoch.set_description("Test")
                data, target = data.cuda(), target.cuda()
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                t2epoch.set_postfix(Accuracy=(100 * correct / total))


def train(model, epochs, TrainLoader, TestLoader, optimizer, criterion, device, start_epoch=0):
    model.train()
    for epoch in range(start_epoch+1, epochs+start_epoch+1):
        with tqdm(TrainLoader, unit="Batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
    accuracy_check(model, TestLoader)
    return model
