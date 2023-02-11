from nn.dataset import Im2ControlsDataset
from datetime import datetime
from nn.model import Model1
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt

from dataclasses import dataclass 
import tyro

torch.manual_seed(42)

@dataclass
class Args:
    data_dir: str = 'data/dataset_beta/'
    val_split: float = 0.3
    seed: int = 42
    exp_id: str = 'exp1'
args = tyro.cli(Args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def main():
    ### Datset Setup ####
    dataset = Im2ControlsDataset(data_dir=args.data_dir)
    total_size = len(dataset)
    test_size = int(total_size * 0.0)
    val_size = int(total_size * args.val_split)
    train_size = total_size - test_size - val_size
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], torch.Generator().manual_seed(args.seed)
        )
    
    train_dataloader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_ds, batch_size=16, shuffle=True, num_workers=0)
    # test_dataloader = DataLoader(test_ds, batch_size=4, shuffle=True, num_workers=0)

    ### Hyper Params ###
    learning_rate = 0.0001
    num_epochs = 100
    save_step = 100

    model = Model1().to(device)
    params = list(model.parameters())
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max = 50, # Maximum number of iterations.
                                eta_min = 1e-5, verbose= True)


    ########### Beginning Epochs ##############
    best_vloss = 999999999
    train_loss = []
    val_loss = []
    for i in range(num_epochs):

        ###### Training ########
        running_loss = 0
        data_iter = 0
        model.train()

        for i_tbatch, sample in enumerate(tqdm(train_dataloader)):
            data_iter += 1
            occ_map = sample['occ_map'].to(device)
            controls = sample["controls"].to(device)
            pred_controls = model(occ_map)

            optimizer.zero_grad()
            loss = criterion(pred_controls, controls)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        average_tloss = running_loss/(data_iter)
        train_loss.append(average_tloss)
        

        ###### Validation ########
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
        val_loss.append(average_vloss)

        print("Epoch: {}. Loss train: {}. Loss Validation: {}".format(i, average_tloss, average_vloss))
        if(average_vloss < best_vloss): # Saving best model
        #    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
           model_path = 'data/model_{}.pt'.format(args.exp_id)
           torch.save(model.state_dict(), model_path)
           best_vloss = average_vloss

    ########### Finishing Epochs ##############
    
    plt.plot(train_loss[1:], label="train loss")
    plt.plot(val_loss[1:], label="val loss")
    plt.legend(loc="upper right")
    plt.savefig('losscurve.png')

if __name__ == "__main__":
    main()
            
        
