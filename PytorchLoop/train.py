import torch

def train(train_dl, model, epochs, opt, loss_fun):
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            # Forward pass
            opt.zero_grad()
            out = model(x)
            loss = loss_fun(out, y)
            loss.backward()
            opt.step()
