from utils import read_temps, device, RNN, Dataset_temp
import torch
from torch.utils.data import Dataset, DataLoader

temp_train = read_temps("data/tempAMAL_train.csv").unsqueeze(2)
temp_test = read_temps("data/tempAMAL_test.csv").unsqueeze(2)

nbClasse = 5
longueurData = 30
BATCH_SIZE = 6
longueurSeq = 4

temp_train = temp_train[:longueurData, :nbClasse]
temp_test = temp_test[:longueurData, :nbClasse]

train_loader = DataLoader(Dataset_temp(temp_train, None, longueurSeq), shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(Dataset_temp(temp_test, None, longueurSeq), shuffle=True, batch_size=BATCH_SIZE)

num_epochs = 10
latent_size = 10
input_dim = 1
output_dim = nbClasse
lr=1e-3

model = RNN(latent_size, input_dim, output_dim)

optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
optimizer.zero_grad()

criterion = torch.nn.CrossEntropyLoss()

# Training loop
print("Training ...")

for epoch in range(num_epochs):
    model.train()
    for i, (sequences, labels) in enumerate(train_loader):

        optimizer.zero_grad()

        hidden_states = model(sequences.permute(1,0,2))
        outputs = model.decode(hidden_states[-1])

        train_loss = criterion(outputs, labels)
        train_loss.backward()
        optimizer.step()
        #writer.add_scalar('Loss/train', train_loss, epoch)
    

    model.eval()
    for i, (sequences, labels) in enumerate(test_loader):
        with torch.no_grad():

            hidden_states = model(sequences.permute(1,0,2))
            outputs = model.decode(hidden_states[-1])
            test_loss = criterion(outputs, labels)

        #writer.add_scalar('Loss/test', test_loss, epoch)
  #if(epoch%10==0):
    print(f"Itérations {epoch}: train loss {train_loss}, test loss {test_loss}")

"""from utils import read_temps, device, RNN, Dataset_temp
import torch
from torch.utils.data import Dataset, DataLoader

#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

temp_test, temp_test_labels = read_temps("data/tempAMAL_test.csv").unsqueeze(1), torch.arange(30)
temp_train, temp_train_labels = read_temps("data/tempAMAL_train.csv").unsqueeze(1), torch.arange(30)
print(f"train shape {temp_train.shape}")
print(f"test shape {temp_test.shape}")

import ipdb; ipdb.set_trace()

BATCH_SIZE = 30

train_loader = DataLoader(Dataset_temp(temp_train, temp_train_labels), shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(Dataset_temp(temp_test, temp_test_labels), shuffle=True, batch_size=BATCH_SIZE)



num_epochs = 50
latent_size = 20
input_dim = 1
output_dim = temp_train.shape[1]

model = RNN(latent_size, input_dim, output_dim)

optimizer = torch.optim.Adam(params=[model.Wx,model.Wh,model.Wd,model.bh,model.bd],lr=1e-3)
optimizer.zero_grad()

error = torch.nn.CrossEntropyLoss()

# Training loop
print("Training ...")

train_loss_list = []
test_loss_list = []

for epoch in range(num_epochs):
    model.train()
    for i, (sequences, labels) in enumerate(train_loader):
        
        optimizer.zero_grad()
        hidden_states = model(sequences)
        outputs = model.decode(hidden_states[-1])
        train_loss = error(outputs, sequences)
        train_loss.backward()
        optimizer.step()
        
        #writer.add_scalar('Loss/train', train_loss, epoch)

    model.eval()
    for i, (sequences, labels) in enumerate(test_loader):
        with torch.no_grad():
            hidden_states = model(sequences)
            outputs = model.decode(hidden_states[-1])
        test_loss = error(outputs, sequences)
        
        #writer.add_scalar('Loss/test', test_loss, epoch)
  #if(epoch%10==0):
    print(f"Itérations {epoch}: train loss {train_loss}, test loss {test_loss}")

"""
