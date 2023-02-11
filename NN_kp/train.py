from dataset import Im2ControlsDataset
from datetime import datetime
from model import Model1
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
import copy
from matplotlib import pyplot as plt
torch.manual_seed(42)
def main():
    dataset = Im2ControlsDataset("/scratch/kaustab.pal/dataset/occ_map/","/scratch/kaustab.pal/dataset/mean_controls/")
    total_size = len(dataset)
    print(total_size)
    test_size = int(total_size * 0.0)
    val_size = int(total_size * 0.3)
    train_size = total_size - test_size - val_size
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], torch.Generator().manual_seed(42)
        )
    
    train_dataloader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_ds, batch_size=16, shuffle=True, num_workers=0)
    # test_dataloader = DataLoader(test_ds, batch_size=4, shuffle=True, num_workers=0)

    learning_rate = 0.0001
    num_epochs = 100
    save_step = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model1(feature_size=256)
    model.to(device)
    
    params = list(model.parameters())
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max = 50, # Maximum number of iterations.
                                eta_min = 1e-5, verbose= True)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_vloss = 999999999
    iter = 0
    train_loss = []
    val_loss = []

    for i in range(num_epochs):
        running_loss = 0
        data_iter = 0
        for i_tbatch, sample in enumerate(train_dataloader):
            data_iter += 1
            model.train()
            occ_map = sample['occ_map'].to(device)
            controls = sample["controls"].to(device)
            
            pred_controls = model(occ_map)
            optimizer.zero_grad()
            loss = criterion(pred_controls, controls)
            running_loss += loss.item()
            
            # if(i_tbatch%100 == 99):
            #     last_loss = running_loss/100
            #     running_loss = 0
            loss.backward()
            optimizer.step()
        average_tloss = running_loss/(data_iter)
        train_loss.append(average_tloss)
        running_vloss = 0
        v_iter = 0
        model.eval()
        with torch.no_grad():
            for i_vbatch, vsample in enumerate(val_dataloader):
                v_iter += 1
                occ_map = sample['occ_map'].to(device)
                controls = sample["controls"].to(device)
                
                pred_controls = model(occ_map)
                vloss = criterion(pred_controls, controls)
                running_vloss += vloss.item()
        average_vloss = running_vloss/(v_iter)
        print("Epoch: {}. Loss train: {}. Loss Validation: {}".format(i, average_tloss, average_vloss))
        val_loss.append(average_vloss)
        scheduler.step()
        #if(average_vloss < best_vloss):
        #    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        #    model_path = 'weights/model_{}_{}'.format(timestamp,i)
        #    torch.save(model.state_dict(), model_path)
        #    best_vloss = average_vloss
    plt.plot(train_loss[1:], label="train loss")
    plt.plot(val_loss[1:], label="val loss")
    plt.legend(loc="upper right")
    plt.savefig('losscurve.png')

if __name__ == "__main__":
    main()
            
        
