# Neural Network from Scratch

## Overview

This project is a self-coded neural network implementation designed for regression and classification tasks, built from scratch. It includes various loss functions and activation functions.

## Features

- Loss Functions: `MSE, CCE, BCE`
= Activations: ```relu, tanh, sigmoid, softmax, linear, leaky_relu, silu```  

1. **Imports**
```
from nn.net import Model
from nn.losses import CrossEntropyLoss
from nn.layer import Layer
from nn.dataloader import DataLoader  
 ``` 
2. **Initialize model**  
 ```
model = Model()
model.add(Layer(dim_in=10, dim_out=16, activation='sigmoid'))
model.add(Layer(16, 32, 'sigmoid'))
model.add(Layer(32, 16, 'sigmoid'))
model.add(Layer(16, 3, 'softmax'))

model.loss_fxn = CrossEntropyLoss()
model.lr = 1e-3  
```  
3. **Dataloader**
```
train_loader = DataLoader(X_train, y_train, batch_size=64, drop_last=False)
val_loader = DataLoader(X_val, y_val, batch_size=64, drop_last=False)  
```
4. **Train-loop** 
```
for epoch in range(epochs):
    loss = 0
    for x, y in train_loader:
        y_pred = model(x)           # forward pass
        loss += model.loss_fxn(y_pred, y)
        model.backward()            # calculate gradients
        model.update_gradients()    # update weights

    loss = loss / len(train_loader)  # take the average loss  
```
5. **Validation Loop**  
 ```
loss = 0
for x, y in val_loader:
    y_pred = model(x)
    loss += model.loss_fxn(y_pred, y)
    # only forward pass and no calculating/updating gradients

loss = loss / len(val_loader)  
```
## About the Project
This project was developed as part of the 6th semester, B.Tech CSE program at Jorhat Engineering College and completed in [May, 2024].

## License
This project is licensed under the [Apache License Version 2.0](./LICENSE)
