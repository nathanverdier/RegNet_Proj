import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import argparse
import os

# -------------------------------
# Definition of a ConvLSTM cell
# -------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = nn.Conv2d(in_channels=input_channels + hidden_channels,
                              out_channels=4 * hidden_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=True)

    def forward(self, x, h_prev, c_prev):
        # Ensure that the dimensions of x and h_prev match
        if x.size() != h_prev.size():
            raise ValueError(f"The dimensions of x {x.size()} and h_prev {h_prev.size()} do not match.")
        
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

# -------------------------------
# Definition of a RegNet block (Regulated block)
# -------------------------------
class RegNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, stride=1, use_conv_lstm=True):
        super(RegNetBlock, self).__init__()
        self.use_conv_lstm = use_conv_lstm
        self.relu = nn.ReLU(inplace=True)
        
        # First convolution with stride handling
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        
        if self.use_conv_lstm:
            self.convlstm = ConvLSTMCell(input_channels=out_channels,
                                         hidden_channels=hidden_channels,
                                         kernel_size=3, padding=1)
            self.conv_fuse = nn.Conv2d(out_channels + hidden_channels, out_channels, kernel_size=1, bias=False)
            self.bn_fuse   = nn.BatchNorm2d(out_channels)
            
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
        # If the number of channels changes or stride != 1, adjust the residual connection
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x, h_state=None, c_state=None):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.use_conv_lstm:
            # Initialize the states if necessary
            if h_state is None or c_state is None:
                h_state = torch.zeros(out.size(0), self.convlstm.hidden_channels, out.size(2), out.size(3), device=out.device)
                c_state = torch.zeros(out.size(0), self.convlstm.hidden_channels, out.size(2), out.size(3), device=out.device)
            h_state, c_state = self.convlstm(out, h_state, c_state)
            out = torch.cat([out, h_state], dim=1)
            out = self.conv_fuse(out)
            out = self.bn_fuse(out)
            out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        out = self.relu(out)
        
        if self.use_conv_lstm:
            return out, h_state, c_state
        else:
            return out

# -------------------------------
# RegNet architecture for CIFAR-10
# -------------------------------
class RegNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(RegNetCIFAR, self).__init__()
        self.in_channels = 16
        self.relu = nn.ReLU(inplace=True)
        
        # Input layer for 32x32 images
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Three groups of blocks
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        # The first block in each layer uses the provided stride
        layers.append(block(self.in_channels, out_channels, hidden_channels=out_channels, stride=stride, use_conv_lstm=True))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, hidden_channels=out_channels, stride=1, use_conv_lstm=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        # For each layer, reset h_state and c_state to avoid dimension conflicts
        for layer in [self.layer1, self.layer2, self.layer3]:
            h_state, c_state = None, None  # Reset for each layer
            for block in layer:
                out, h_state, c_state = block(out, h_state, c_state)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# -------------------------------
# RegNet architecture for ImageNet
# -------------------------------
class RegNetImageNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(RegNetImageNet, self).__init__()
        self.in_channels = 64
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, hidden_channels=out_channels, stride=stride, use_conv_lstm=True))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, hidden_channels=out_channels, stride=1, use_conv_lstm=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        h_state, c_state = None, None
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                out, h_state, c_state = block(out, h_state, c_state)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# -------------------------------
# Training and testing functions
# -------------------------------
def train_model(model, criterion, optimizer, trainloader, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(trainloader)} | Loss: {running_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.2f}%")

def test_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    print("Test Accuracy: {:.2f}%".format(acc))
    return acc

# -------------------------------
# Main program with dataset selection
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='Test RegNet on CIFAR-10 or ImageNet')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'ImageNet'],
                        help='Choose the dataset: CIFAR10 or ImageNet')
    parser.add_argument('--data', type=str, default='./data', help='Path to the data (for ImageNet, the root folder path)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    if args.dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        
        model = RegNetCIFAR(RegNetBlock, [2, 2, 2], num_classes=10)
        
    elif args.dataset == 'ImageNet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        trainset = datasets.ImageFolder(os.path.join(args.data, 'train'), transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        testset = datasets.ImageFolder(os.path.join(args.data, 'val'), transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
        
        model = RegNetImageNet(RegNetBlock, [3, 4, 6, 3], num_classes=1000)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nStarting epoch {epoch}")
        train_model(model, criterion, optimizer, trainloader, device, epoch)
        test_model(model, testloader, device)
    
if __name__ == '__main__':
    main()
