import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# -------------------------------
# Définition d'une cellule ConvLSTM
# -------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        # On concatène l'entrée et l'état caché, puis on calcule 4 sorties pour les portes
        self.conv = nn.Conv2d(in_channels=input_channels + hidden_channels,
                              out_channels=4 * hidden_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=True)

    def forward(self, x, h_prev, c_prev):
        # x: (batch, input_channels, H, W)
        # h_prev: (batch, hidden_channels, H, W)
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)
        (cc_i, cc_f, cc_o, cc_g) = torch.split(conv_out, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

# -------------------------------
# Définition d'un bloc RegNet
# -------------------------------
class RegNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, use_conv_lstm=True):
        super(RegNetBlock, self).__init__()
        self.use_conv_lstm = use_conv_lstm
        self.relu = nn.ReLU(inplace=True)
        
        # Première convolution 3x3
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        
        if self.use_conv_lstm:
            # Cellule ConvLSTM pour réguler l'information
            self.convlstm = ConvLSTMCell(input_channels=out_channels,
                                         hidden_channels=hidden_channels,
                                         kernel_size=3, padding=1)
            # Fusionner la sortie de la convolution et l'état caché via une convolution 1x1
            self.conv_fuse = nn.Conv2d(out_channels + hidden_channels, out_channels, kernel_size=1, bias=False)
            self.bn_fuse   = nn.BatchNorm2d(out_channels)
            
        # Deuxième convolution 3x3
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
        # Si le nombre de canaux change, on ajuste la connexion résiduelle
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
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
            # Initialisation des états si nécessaire
            if h_state is None or c_state is None:
                h_state = torch.zeros_like(out)
                c_state = torch.zeros_like(out)
            # Passage dans la cellule ConvLSTM
            h_state, c_state = self.convlstm(out, h_state, c_state)
            # Concaténation de la sortie de conv1 et de l'état caché
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
# Définition du modèle RegNet pour CIFAR-10
# -------------------------------
class RegNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(RegNet, self).__init__()
        self.in_channels = 16
        self.relu = nn.ReLU(inplace=True)
        
        # Couche d'entrée
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        
        # Trois groupes de blocs, avec redimensionnement des caractéristiques
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        # Pooling global et couche de classification
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        # Si stride > 1, on ajuste la taille des caractéristiques par une convolution de stride 2
        if stride != 1:
            layers.append(nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            self.in_channels = out_channels
        # Premier bloc de la couche
        layers.append(block(self.in_channels, out_channels, hidden_channels=out_channels, use_conv_lstm=True))
        self.in_channels = out_channels
        # Blocs suivants
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, hidden_channels=out_channels, use_conv_lstm=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Passage par les différentes couches de blocs
        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                if isinstance(block, RegNetBlock) and block.use_conv_lstm:
                    out, _, _ = block(out)
                else:
                    out = block(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# -------------------------------
# Fonctions d'entraînement et de test
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
    print("Test Accuracy: {:.2f}%".format(100.*correct/total))

# -------------------------------
# Programme principal
# -------------------------------
def main():
    # Configuration du dispositif (GPU si disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Utilisation de :", device)
    
    # Transformations et chargement du dataset CIFAR-10
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
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    # Création du modèle RegNet
    # Ici, nous utilisons une architecture avec 3 groupes de blocs [2, 2, 2]
    model = RegNet(RegNetBlock, [2, 2, 2], num_classes=10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train_model(model, criterion, optimizer, trainloader, device, epoch)
        test_model(model, testloader, device)
    
if __name__ == '__main__':
    main()
