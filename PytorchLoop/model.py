import torch
from torchvision import datasets, transforms

import callbacks
from tqdm import tqdm

# Download resnet 100 and put in gpu
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
# Replace last layer with 10 nodes
model.fc = torch.nn.Linear(512, 10)

# Put model in gpu
model = model.cuda()

# Download cifar 10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(
    root='./Half-Precision-Training/data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./Half-Precision-Training/data', train=False,
                           download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(  # type: ignore
    trainset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)

testloader = torch.utils.data.DataLoader(  #type: ignore
    testset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

callback = callbacks.Callback()
call = callbacks.TensorboardHandler(log_dir='./logs')

call.on_train_begin(trainloader, model)
for epoch in range(1,3):
    call.on_epoch_begin()
    model.train()
    with tqdm(trainloader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            data, target = data.cuda(), target.cuda()
            call.on_batch_begin()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            call.on_batch_begin()
            tepoch.set_postfix(loss=loss.item())
    call.on_epoch_end(epoch, {'loss': loss.item()})
call.on_train_end()