import torch

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def calculate_channel_means(dataset, label):
    #calculates and returns per channel mean of dataset containing image data
    mean = 0
    n_images = dataset.__len__()
    for i in range(n_images):
        mean += torch.mean(dataset.__getitem__(i)[label].type(torch.float), dim= [1,2])
    return mean/n_images

def calculate_channel_sds(dataset, label):
    #calculates and returns per channel standard deviation of dataset containing image data
    var = 0
    n_images = dataset.__len__()
    for i in range(n_images):
        var += torch.var(input = dataset.__getitem__(i)[label].float(), dim = [1,2])
    return torch.sqrt(var/n_images)

def train_model(model, dataloader,epochs, optimizer, criterion, device):
    """
    Trains model and returns dict containing trained model and the loss/acc scores per epoch
    """
    #create lists to return acc and loss scores per epoch to user
    acc_per_epoch = [0] * epochs
    loss_per_epoch = [0] * epochs

    for e in range(epochs):
        run_loss = 0
        epoch_loss = 0

        correct = 0
        acc =0
        for i, batch in enumerate(dataloader):

            imgs, labels = batch['image'].to(device), batch['idx_label'].to(device)

            #zero grad
            optimizer.zero_grad()

            #make predictions and calculate loss
            preds = model(imgs)
            loss = criterion(preds, labels)

            #calculate gradient and step
            loss.backward()
            optimizer.step()

            #calculate metrics for mini-batch
            run_loss += loss.item()
            correct += (torch.argmax(preds, dim = 1) == labels).float().sum()


        #calculate epoch stats
        print("Epoch {} Statistics:".format(e + 1))
        print("=" * 20)

        epoch_loss = run_loss / len(dataloader)
        loss_per_epoch[e] = epoch_loss
        print("epoch loss: ", str(round(epoch_loss, 3)))

        acc = correct.double() / len(dataloader.dataset)
        acc_per_epoch[e] = acc

        print("epoch acc: {:.2f}%\n".format(acc * 100))

    return {"trained model": model,
            "acc per epoch": acc_per_epoch,
            "loss per epoch": loss_per_epoch}

def model_check_cuda(model):
    """
    Raise error if model not on cuda
    """
    assert next(model.parameters()).is_cuda, "Model not on cuda device"

def plot_loss(loss_scores:list, 
              epochs:int, 
              figsize:tuple = (5,5), 
              title:str = None, 
              font_dict:dict=None,
              save_fig:bool= False,
              path:str = None):

    plt.plot(epochs, loss_scores)
    plt.suptitle(title, fontdict=font_dict)
    plt.xlabel('Epochs', fontdict=font_dict)
    plt.ylabel('Loss', fontdict=font_dict)

    if save_fig:
        plt.savefig(path, bbox_inches='tight')

def plot_acc(acc_scores:list, 
            epochs:int, 
            figsize:tuple = (5,5), 
            title:str = None, 
            font_dict:dict=None,
            save_fig:bool= False,
            path:str = None):

    plt.plot(epochs, acc_scores)
    plt.suptitle(title, fontdict=font_dict)
    plt.xlabel('Epochs', fontdict=font_dict)
    plt.ylabel('Accuracy %', fontdict=font_dict)

    if save_fig:
        plt.savefig(path, bbox_inches='tight')

