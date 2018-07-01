import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#=========================================>
#   < Preprocessing the Images >
#=========================================>

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#=========================================>
#   < Building the Neural Network >
#=========================================>

class ResNet(nn.Module):
    
    def __init__(self): 
        super(ResNet, self).__init__()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.norm1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.norm2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.norm3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.norm4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.norm5 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)
        self.norm6 = nn.BatchNorm2d(256)
        
        self.conv7 = nn.Conv2d(256, 512, 3, padding = 1)
        self.norm7 = nn.BatchNorm2d(512)
        
        self.conv8 = nn.Conv2d(512, 512, 3, padding = 1)
        self.norm8 = nn.BatchNorm2d(512)
        
        
        self.fc1 = nn.Linear(512 * 4 * 4, 100)
        self.norm9 = nn.BatchNorm1d(100)
        
        self.fc2 = nn.Linear(100, 100)
        self.norm10 = nn.BatchNorm1d(100)
       
        self.fc3 = nn.Linear(100, 50)
        self.norm11 = nn.BatchNorm1d(50)
        
        self.fc4 = nn.Linear(50, 10)
        

    def forward(self, x):       
        
        out = F.relu(self.norm1(self.conv1(x)))
        out = F.relu(self.norm2(self.conv2(out)))
        out = self.pool(out)
        
        out = F.relu(self.norm3(self.conv3(out)))
        out = F.relu(self.norm4(self.conv4(out)))
        out = self.pool(out)
        
        out = F.relu(self.norm5(self.conv5(out)))
        out = F.relu(self.norm6(self.conv6(out)))
        out = self.pool(out)
        
        out = F.relu(self.norm7(self.conv7(out)))
        out = F.relu(self.norm8(self.conv8(out)))
        
        out = out.view(128, 512 * 4 * 4)
        
        out = F.relu(self.norm9(self.fc1(out)))
        out = F.relu(self.norm10(self.fc2(out)))
        out = F.relu(self.norm11(self.fc3(out)))
        out = self.fc4(out)

        return out
    

#=========================================>
#   < Training the Network >
#=========================================>

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
net = ResNet()
net = net.to(device)

print(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

for epoch in range(10):  

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
       
        inputs, labels = data  
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        outputs = net(inputs)
        outputs = outputs.to(device)
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()
        
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('\n < Finished Training > \n')


#=========================================>
#   < Testing the Netwok >
#=========================================>

dataiter = iter(testloader)
images, labels = dataiter.next()
images, labels = images.to(device), labels.to(device)

outputs = net(images)

_, predicted = torch.max(outputs, 1)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy : %d %%' % (
    100 * correct / total))


