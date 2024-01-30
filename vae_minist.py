import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image  # Save a given Tensor into an image file.
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np


#构建VAE模型，主要由Encoder和Decoder组成
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.rand_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var



def loss_function(x_reconst, x, mu, log_var): #损失函数，抄的网上的
    BCE_loss = nn.BCELoss(reduction='sum')
    reconstruction_loss = BCE_loss(x_reconst, x)
    KL_divergence = -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mu ** 2)
    return reconstruction_loss + KL_divergence




if __name__ == '__main__':
    image_size = 784
    h_dim = 400
    z_dim = 20
    num_epochs = 30
    batch_size = 128
    learning_rate = 0.001

    dataset = torchvision.datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    examples = enumerate(data_loader)  # 组合成一个索引序列
    batch_idx, (example_data, example_targets) = next(examples)

    fig = plt.figure() # 显示几张data_loader里的图片
    for i in range(6) :
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title('Groud Truth: {}'.format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(30) : # 开始训练
        train_loss = 0
        train_acc = 0
        model.train()
        for imgs, labels in data_loader :
            imgs = imgs.to(device)
            labels = labels.to(device)
            real_imgs = torch.flatten(imgs, start_dim=1)
            # 前向传播
            gen_imgs, mu, log_var = model(real_imgs)
            loss = loss_function(gen_imgs, real_imgs, mu, log_var)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录误差
            train_loss += loss.item()
        print('epoch: {}, loss: {}'.format(epoch, train_loss / len(data_loader)))
        losses.append(train_loss / len(data_loader))

        fake_images = gen_imgs.view(-1, 1, 28, 28)
        x_concat = torch.cat([imgs.view(-1, 1, 28, 28), fake_images], dim=3)
        save_image(x_concat, 'MNIST_fake_pics/fake_images-{}.png'.format(epoch + 1)) #将原图和生成的图片放一起对比
    torch.save(model.state_dict(), './vae.pth')
    plt.title('trainloss')
    plt.plot(np.arange(len(losses)), losses, linewidth=1.5, linestyle='dashed', label='train_losses')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()