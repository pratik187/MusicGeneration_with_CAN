import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import numpy as np

def train_DCGAN(G, D, optim_G, optim_D, loss_f1, loss_f2, train_loader, num_epochs, label2onehot, 
            CrossEntropy_uniform, n_class, device):

    for epoch in range(num_epochs):

        for i, (img,labels) in enumerate(train_loader):
            batch_size = img.shape[0]
            labels = label2onehot(labels)
            #labels = labels.long()
            
            real_label = torch.ones(batch_size, device=device)
            fake_label = torch.zeros(batch_size, device=device)

            # ========================
            #   Train Discriminator
            # ========================
            # train with real data
            img = img.to(device)
            real_r_score, real_c_score = D(img)
            

            # train with fake data
            noise = torch.randn(batch_size, 100, device=device)
            img_fake = G(noise)
            print('fake image shape=')
            print(img_fake.shape)
            fake_r_score, fake_c_score = D(img_fake)
            

            # update D      
            print(np.resize(labels,(labels.length())).shape)
            d_loss = loss_f1(real_r_score.flatten(), real_label) + loss_f2(real_c_score.flatten(), labels)
            d_loss = d_loss + loss_f1(fake_r_score, fake_label)
            D.zero_grad()
            d_loss.backward()
            optim_D.step()

            # ========================
            #   Train Generator
            # ========================
            
            g_loss = loss_f(fake_r_score, real_label)
            g_loss = g_loss + CrossEntropy_uniform(fake_c_score)

            # update G
            G.zero_grad()
            g_loss.backward()
            optim_G.step()

            if i % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(train_loader), d_loss.item(), g_loss.item(),
                         real_score.mean().item(), fake_score.mean().item(), g_score.mean().item()))

        noise = torch.randn(24, 100, device=device)
        img_fake = G(noise)
        grid = make_grid(img_fake)
        plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
        plt.show()
        
def CrossEntropy_uniform(self, pred):
    logsoftmax = nn.LogSoftmax(dim=1)
    unif = torch.full((self.batch_size, self.n_class), 1/self.n_class)
    return torch.mean(-torch.sum(unif * logsoftmax(pred), 1))
    
    