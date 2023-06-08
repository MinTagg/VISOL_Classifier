import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import MODEL
from dataloader import MyDataset
import time
import tqdm

# hyperparameter
batch_size = 128
learning_rate = 0.001
epoch = 60
model_name = 'resnet50'
freeze = True

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# data path
train_data_path = '../VISOL/train/'

# make dataset
dataset = MyDataset()
# seperate dataset into train and validation using sklearn
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
# make dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# make model
model = MODEL(freeze = freeze, network = model_name).to(device)

# make optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# make loss function. it is classification task, and label is one hot encoded label
criterion = nn.BCELoss()

# make directory in RESULT folder that named now time
now = time.localtime()
now_time = "%04d-%02d-%02d-%02d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
os.mkdir('RESULT/{}'.format(now_time))

print(now_time)

save_path= ('RESULT/{}'.format(now_time))
# make log txt in save_path
f = open(save_path + '/log.txt', 'w')
# save hyperparameter in log.txt
f.write(f"batch_size: {freeze}\n")
f.write(f"batch_size: {model_name}\n")

# train
for i in range(epoch):
    # average loss
    avg_loss = 0
    model.train()

    loader = tqdm.tqdm(train_dataloader, total=len(train_dataloader))
    for img, label in loader:
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

    
    print("epoch: {}, loss: {}".format(i, avg_loss / len(train_dataloader)))
    f.write("epoch: {}, loss: {}\n".format(i, avg_loss / len(train_dataloader)))
    # validation start
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in val_dataloader:
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            for j in range(len(output)):
                if torch.argmax(output[j]) == torch.argmax(label[j]):
                    correct += 1
                total += 1
    print("validation accuracy: {}".format(correct / total))
    f.write("validation accuracy: {}\n".format(correct / total))
    # save model
    torch.save(model.state_dict(), save_path + '/epoch_{}.pt'.format(i))