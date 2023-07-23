import pandas as pd
import torch
import torchvision
from torchvision import datasets, transforms

# Load the data
data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

# Split the data into train and test sets
train_set, test_set = torch.utils.data.random_split(data, [50000, 10000])

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

# Define the model
class Logistic_Regression(torch.nn.Module):
    def __init__(self):
        super(Logistic_Regression, self).__init__()
        self.linear = torch.nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        y_pred = self.linear(x)
        return y_pred
    
# Instantiate the model
model = Logistic_Regression()

# Define the loss function and optimizer
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        y_pred = model(images)
        # Compute the loss
        cost = loss(y_pred, labels)
        # Backward pass
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        # Print the loss
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, 10, i+1, len(train_loader), cost.item()))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        # Forward pass
        y_pred = model(images)
        # Get the prediction
        _, predicted = torch.max(y_pred.data, 1)
        # Total number of labels
        total += labels.size(0)
        # Total correct predictions
        correct += (predicted == labels).sum().item()
    # Print accuracy
    print('Accuracy: {} %'.format(100 * correct / total))






