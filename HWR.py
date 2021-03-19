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
learning_rate = 1e-3  # 0.001
num_epoches = 3
image_size = 28
train_path  = "NMNIST_Combine/trainSet"
val_path = "SVHN/testSet"
model_path = "RNN_NMNIST_Combine_SVHN.pth"



# Pytorch Transforms
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            # transforms.RandomAffine(15),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.RandomRotation(15),
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

train_dataset = datasets.ImageFolder(
   train_path, transform=data_transforms["train"]
)
test_dataset = datasets.ImageFolder(
    val_path, transform=data_transforms["val"]
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# RNN Module
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


model = RNN(28, 128, 3, 10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

accuracy, losses = [], []
for epoch in range(num_epoches):
    print("epoch {}".format(epoch + 1))
    print("**************************************")
    for step, (img, label) in enumerate(train_loader):
        b, c, h, w = img.size()
        img = img.squeeze(1)
        out = model(img)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = loss.data.item() * label.size(0) / batch_size
        _, pred = torch.max(out, 1)
        train_acc = (pred == label).sum().data.item() / batch_size
        accuracy.append(train_acc)
        losses.append(train_loss)
        if step % 50 == 0:
            print(
                "[{}/{}] step: {} Loss: {:.6f}, Acc: {:.6f}".format(
                    epoch + 1, num_epoches, step, train_loss, train_acc
                )
            )
        # if step % 100 == 0:
        #     for p in optimizer.param_groups:
        #         p['lr'] *= 0.5

torch.save(model, model_path)



## Output plot chart
# fig = plt.figure(figsize=(16, 6))
# fig.add_subplot(121)
# plt.plot(losses, 'r')
# plt.xlabel("Iteration")
# plt.title("Loss")
# fig.add_subplot(122)
# plt.plot(accuracy)
# plt.xlabel("Iteration")
# plt.title("Accuracy")
# plt.show()

## Print Average acuracy
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

train_acc = []
for step, (img, label) in enumerate(train_loader):
    b, c, h, w = img.size()
    img = img.squeeze(1)
    out = Net(img)
    _, pred = torch.max(out, 1)
    acc = (pred == label).sum().data.item() / batch_size
    train_acc.append(acc)
print("Train accuracy:", np.sum(train_acc) / len(train_acc))

## Read img file
image_size = 28
transformer = transforms.Compose([lambda x: Image.open(x).convert('L').resize((28, 28), Image.ANTIALIAS),
                                  transforms.ToTensor(),
                                  ])

img_path = "G-Capcha Img/Test_1/5.png"
img = 1 - transformer(img_path)
plt.imshow(Image.open(img_path).convert('L'))
img = img[0].unsqueeze(0)
out = Net(img)
_, pred = torch.max(out, 1)
print("Predict Class:", pred[0].numpy())