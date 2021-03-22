import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms


batch_size = 64
image_size = 28
val_path = "SVHN/testSet"

class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(RNN, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.classifier(out[:, -1, :])
        return out

model_path = "RNN_NMNIST_MB_SVHN.pth"
Net = torch.load(model_path)

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            # transforms.RandomAffine(15),
            # transforms.ColorJitter(
            #     brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            # ),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    ),
}

test_dataset = datasets.ImageFolder(
    val_path, transform=data_transforms["val"])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

import numpy as np
Net = torch.load(model_path)
test_acc = []
for step, (img, label) in enumerate(test_loader):
    b, c, h, w = img.size()
    img = img.squeeze(1)
    out = Net(img)
    _, pred = torch.max(out, 1)
    acc = (pred == label).sum().data.item() / batch_size
    test_acc.append(acc)
print("Test accuracy:", np.sum(test_acc) / len(test_acc))


## Read img file
image_size = 28
transformer = transforms.Compose([lambda x: Image.open(x).convert('L').resize((28, 28), Image.ANTIALIAS),
                                  transforms.ToTensor(),
                                  ])
img_path = "G-Capcha Img/Test_2/0.png"
img = 1 - transformer(img_path)
plt.imshow(Image.open(img_path).convert('L'))
img = img[0].unsqueeze(0)
out = Net(img)
_, pred = torch.max(out, 1)
print("Predict Class:", pred[0].numpy())