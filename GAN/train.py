import model as nets

from torch import nn as nn
from torch import optim
import torch
from torchvision import datasets as tvsets
from torchvision import transforms as T
import torch.utils.data as D


transform = T.Compose([
                       T.Resize(32),
                       T.ToTensor(),
])
trainset = tvsets.MNIST(root='data', train=True, transform=transform, download=True)
trainloader = D.DataLoader(trainset, batch_size=1,
                           shuffle=True, num_workers=1)


size = 32*32
#filename =
print_freq = 100
#save_freq =

generator = nets.Generator(size)
discriminator = nets.Discriminator(size)

def noise():
    return torch.randn(1, size)

def train():

    g_optimizer = optim.Adam(generator.parameters(), lr = 0.01)
    d_optimizer = optim.Adam(discriminator.parameters(), lr = 0.01)
    loss_func = nn.BCELoss()

    sum_g_loss = 0
    sum_d_loss = 0
    print("Beginning training...")
    for epoch in range(1):
        for num, real in enumerate(trainloader):

            #train generator
            g_optimizer.zero_grad()

            generated = generator(noise())
            prob_real = discriminator(generated)

            g_loss = loss_func(prob_real, torch.tensor([[1.0]]))

            sum_g_loss+=g_loss.item()

            g_loss.backward()
            g_optimizer.step()

            #train discriminator
            d_optimizer.zero_grad()

            real_prob = discriminator(real[0].view(1, -1))
            fake_prob = discriminator(generated.detach())

            real_loss = loss_func(real_prob, torch.tensor([[1.0]]))
            fake_loss = loss_func(fake_prob, torch.tensor([[0.0]]))
            d_loss = (real_loss + fake_loss) / 2

            sum_d_loss+=d_loss.item()

            d_loss.backward()
            d_optimizer.step()

            #print loss every x iterations
            if (num % print_freq == print_freq-1):
                print("generator loss: " + str(sum_g_loss / print_freq))
                print("discriminator loss: " + str(sum_d_loss / print_freq))
                print("\n")
                sum_g_loss = 0
                sum_d_loss = 0

            #save model params every x iterations
            #if (num % save_freq == save_freq-1):
            #    torch.save(autoenc.state_dict(), "saved_models/" + filename)
            #    print("saved model")

        print("Epoc " + str(epoch) + " complete")
    print("Training complete.")

if __name__ == '__main__':
    train()

#def test(model, testloader):
#    num_correct = 0
#    num_total = 0
#    for input, target in testloader:
#        predictions = model(input)
#        prediction = torch.argmax(predictions, dim = 1)
#        #print(prediction)
#        #print(target)
#        if (prediction == target):
#            num_correct+=1
#        num_total+=1
#    print("Percent correct: " + str(num_correct/num_total*100) + "%")