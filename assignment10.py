data_path1= r"C:\Users\kutra\Downloads\CNN_digitdata1.csv"
labels_path1 = r"C:\Users\kutra\Downloads\CNN_digitdata1_labels.csv"
data_path2 = r"C:\Users\kutra\Downloads\CNN_digitdata2.csv"
labels_path2 = r"C:\Users\kutra\Downloads\CNN_digitdata2_labels.csv"

import pandas as pd

def get_labels(label_path):
    labels = pd.read_csv(label_path, header=None)  

    data_list = labels.values.flatten().tolist()
    l=[]
    for i in data_list:
        a=i.split("\t")
        l.append(a)

    labell=l[0]
    labell.pop()
    labels_array = labell
    for i in labels_array:
        i = float(i)
    labels_array
    ll1=[]
    for i in labels_array:
        a = float(i)
        a = int(a)
        ll1.append(a)
    return ll1
labels_of_1 = get_labels(labels_path1)
labels_of_2 = get_labels(labels_path2)

import numpy as np
def get_dataset_array(data_path):
    data = pd.read_csv(data_path, header=None)  
    
    dataset = []

    i = 0
    while i < len(data):
        if 'Image' in str(data.iloc[i][0]):
            
            image_chunk = []
            
            
            for j in range(1, 29):  
                if i + j < len(data): 
                    numbers = data.iloc[i + j][0].strip().split('\t')
                    image_chunk.append([float(num) for num in numbers])
            
           
            if len(image_chunk) == 28:
                dataset.append(image_chunk)
        
        
        i += 29 

    
    dataset_array = np.array(dataset)
    return dataset_array

data_array1 = get_dataset_array(data_path1)
data_array2 = get_dataset_array(data_path2)

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

dataset1 = CustomImageDataset(data_array1, labels_of_1)

# Step 3: Split into train and test sets (80:20)
train_size = int(0.8 * len(dataset1))
test_size = len(dataset1) - train_size
train_dataset, test_dataset = random_split(dataset1, [train_size, test_size])

# Step 4: Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  
        self.fc2 = nn.Linear(128, 2)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleCNNModel(nn.Module):
    def __init__(self):
        super(SimpleCNNModel, self).__init__()
        # A single convolutional layer with fewer output channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # A smaller fully connected layer
        self.fc1 = nn.Linear(16 * 14 * 14, 32)  
        self.fc2 = nn.Linear(32, 2)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Kernel5CNNModel(nn.Module):
    def __init__(self):
        super(Kernel5CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # Adding padding=2 to keep output size consistent
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2) # Adding padding=2 to keep output size consistent
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After two pooling layers on a 28x28 input, the size will be reduced to 7x7 with this configuration
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  
        self.fc2 = nn.Linear(128, 2)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First conv + pool
        x = self.pool(F.relu(self.conv2(x)))  # Second conv + pool
        x = x.view(-1, 64 * 7 * 7)           # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))               # First fully connected layer
        x = self.fc2(x)                       # Output layer
        return x



# Initialize model, loss function, and optimizer
model1= SimpleCNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001)

model2 =CNNModel()
criterion1 = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model2.parameters(), lr =0.001)

model3 = Kernel5CNNModel()
criterion2 = nn.CrossEntropyLoss()
optimizer2 = optim.Adam(model3.parameters(),lr = 0.001)

# Step 5: Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    train_losses = []  # List to store loss values for each epoch

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)  # Append loss of this epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Plotting training loss vs. epoch
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs. Epoch")
    plt.legend()
    plt.show()

# Step 6: Testing loop
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Train and test the model
print("simpleCNN")
train_model(model1, train_loader, criterion, optimizer, num_epochs=25)
test_model(model1, test_loader)
print("CNN")
# Train and test the model
train_model(model2, train_loader, criterion1, optimizer1, num_epochs=25)
test_model(model2, test_loader)
print("KERNEL5")
train_model(model3, train_loader, criterion2, optimizer2, num_epochs=25)
test_model(model3, test_loader)


dataset1 = CustomImageDataset(data_array2, labels_of_2)

# Step 3: Split into train and test sets (80:20)
train_size = int(0.8 * len(dataset1))
test_size = len(dataset1) - train_size
train_dataset, test_dataset = random_split(dataset1, [train_size, test_size])

# Step 4: Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  
        self.fc2 = nn.Linear(128, 2)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleCNNModel(nn.Module):
    def __init__(self):
        super(SimpleCNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(16 * 14 * 14, 32)  
        self.fc2 = nn.Linear(32, 2)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Kernel5CNNModel(nn.Module):
    def __init__(self):
        super(Kernel5CNNModel, self).__init__()
        
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2) 
        
    
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  
        self.fc2 = nn.Linear(128, 2)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, 64 * 7 * 7)           
        x = F.relu(self.fc1(x))               
        x = self.fc2(x)                       
        return x



# Initialize model, loss function, and optimizer
model1= SimpleCNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001)

model2 =CNNModel()
criterion1 = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model2.parameters(), lr =0.001)

model3 = Kernel5CNNModel()
criterion2 = nn.CrossEntropyLoss()
optimizer2 = optim.Adam(model3.parameters(),lr = 0.001)

# Step 5: Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    train_losses = []  # List to store loss values for each epoch

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)  # Append loss of this epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Plotting training loss vs. epoch
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs. Epoch")
    plt.legend()
    plt.show()

# Step 6: Testing loop
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Train and test the model
print("simpleCNN")
train_model(model1, train_loader, criterion, optimizer, num_epochs=25)
test_model(model1, test_loader)
print("CNN")
# Train and test the model
train_model(model2, train_loader, criterion1, optimizer1, num_epochs=25)
test_model(model2, test_loader)
print("KERNEL5")
train_model(model3, train_loader, criterion2, optimizer2, num_epochs=25)
test_model(model3, test_loader)