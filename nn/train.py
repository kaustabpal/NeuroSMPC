from nn.dataset import Im2ControlsDataset
from datetime import datetime
from nn.model import Model1
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lion_pytorch import Lion
import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from dataclasses import dataclass 
import tyro
import os
import time
# torch.backends.cudnn.deterministic=True

@dataclass
class Args:
    dataset_dir: str = '/home2/kaustab.pal/iros_23/dataset/train/' #'../iros_23/dataset/' #'/scratch/kaustab.pal/iros_23/dataset/' # 'data/dataset_beta/'
    weights_dir: str = '/home2/kaustab.pal/iros_23/weights/' #'../iros_23/weights/' #'/scratch/kaustab.pal/iros_23/weights/' 
    loss_dir: str = '/home2/kaustab.pal/iros_23/loss/'  #'../iros_23/loss/' #'/scratch/kaustab.pal/iros_23/loss/' 
    val_split: float = 0.2
    num_epochs: int = 500
    seed: int = 12321
    exp_id: str = 'exp19'
args = tyro.cli(Args)


def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.loss_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ",device)
    ### Datset Setup ####
    dataset = Im2ControlsDataset(dataset_dir=args.dataset_dir)
    total_size = len(dataset)
    test_size = int(total_size * 0.0)
    val_size = int(total_size * args.val_split)
    train_size = total_size - test_size - val_size
    print("Total dataset size: ", total_size)
    print("Training dataset size: ", train_size)
    print("Validation dataset size: ", val_size)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], torch.Generator().manual_seed(args.seed)
        )
    
    train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=0)
    # test_dataloader = DataLoader(test_ds, batch_size=4, shuffle=True, num_workers=0)

    ### Hyper Params ###
    learning_rate = 0.0001
    num_epochs = args.num_epochs #10 #500
    save_step = 1000

    model = Model1().to(device)
    params = list(model.parameters())
    criterion = nn.MSELoss()
    #optimizer = torch.optim.Adam(params, lr=learning_rate)
    optimizer = Lion(model.parameters(), lr = learning_rate)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.33, patience=20, threshold=0.01, verbose=True)
    scheduler = CosineAnnealingLR(optimizer,
                                T_max = 400, # Maximum number of iterations.
                                eta_min = 1e-5, verbose= True)


    ########### Beginning Epochs ##############
    best_vloss = 999999999
    train_v_loss = []
    train_w_loss = []
    val_v_loss = []
    val_w_loss = []
    dt=0.1
    diag_dt = 0.1*torch.ones(30,30, device=device)
    diag_dt = torch.tril(diag_dt)
    theta_0 = torch.deg2rad(torch.tensor([90], \
                dtype=torch.float32, device=device))*torch.ones(1,1,30,device=device)
    x_0 = torch.tensor([0], dtype=torch.float32, device=device)*torch.ones(30,1, device=device)
    y_0 = torch.tensor([0], dtype=torch.float32, device=device)*torch.ones(30,1, device=device)
    for i in range(num_epochs):

        ###### Training ########
        running_v_loss = 0
        running_w_loss = 0
        data_iter = 0
        model.train()
        for i_tbatch, sample in enumerate(tqdm(train_dataloader)):
            data_iter += 1
            occ_map = sample['occ_map'].to(device)
            controls = sample["controls"].to(device)
            # print(controls.shape)
            v_idx = [range(0,controls.shape[1],2)]
            w_idx = [range(1,controls.shape[1],2)]
            # print(controls)
            v_gt = controls[:,v_idx].clone()
            w_gt = controls[:,w_idx].clone()

            w_dt_gt = w_gt*torch.tensor([0.1], device=device)
            w_sum_gt = torch.cumsum(w_dt_gt,dim=2)
            theta_gt = theta_0 + w_sum_gt
            c_theta_gt = torch.cos(theta_gt)
            s_theta_gt = torch.sin(theta_gt)
            vc_theta_gt = v_gt*torch.cos(theta_gt)
            vs_theta_gt = v_gt*torch.sin(theta_gt)

            pred_controls = model(occ_map)
            v_pred = pred_controls[:,v_idx].clone()
            w_pred = pred_controls[:,w_idx].clone()
            
            w_dt_pred = w_pred*torch.tensor([0.1], device=device)
            w_sum_pred = torch.cumsum(w_dt_pred,dim=2)
            theta_pred = theta_0 + w_sum_pred
            c_theta_pred = torch.cos(theta_pred)
            s_theta_pred = torch.sin(theta_pred)
            vc_theta_pred = v_pred*torch.tensor([4.13], device=device, requires_grad=True)*torch.cos(theta_pred)
            vs_theta_pred = v_pred*torch.tensor([4.13], device=device, requires_grad=True)*torch.sin(theta_pred)

            optimizer.zero_grad()
            v_loss = criterion(v_pred*torch.tensor([4.13], device=device, requires_grad=True), v_gt)
            c_theta_loss = criterion(c_theta_gt, c_theta_pred)
            s_theta_loss = criterion(s_theta_gt, s_theta_pred)
            w_loss = criterion(w_pred*torch.tensor([57.2958], device=device, requires_grad=True), \
                    w_gt*torch.tensor([57.2958], device=device, requires_grad=True))
            vc_loss = criterion(vc_theta_pred, vc_theta_gt)
            vs_loss = criterion(vs_theta_pred, vs_theta_gt)
            #loss = vc_loss + vs_loss #+ w_loss
            loss = v_loss + w_loss 
            loss.backward()
            optimizer.step()
            running_v_loss += v_loss.item()
            running_w_loss += w_loss.item()        

        average_v_loss = running_v_loss/(data_iter)
        average_w_loss = running_w_loss/(data_iter)
        train_v_loss.append(average_v_loss)
        train_w_loss.append(average_w_loss)
        print("Epoch: {}. Training: V Loss: {}. W Loss: {}".format(i, average_v_loss, average_w_loss))
        ###### Validation ########
        running_vloss = 0
        running_loss = 0
        v_iter = 0
        model.eval()
        with torch.no_grad():
            for i_vbatch, vsample in enumerate(val_dataloader):
                v_iter += 1
                occ_map = sample['occ_map'].to(device)
                controls = sample["controls"].to(device)
                v_idx = [range(0,controls.shape[1],2)]
                w_idx = [range(1,controls.shape[1],2)]
                v_gt = controls[:,v_idx]
                w_gt = controls[:,w_idx]
                w_dt_gt = w_gt*torch.tensor([0.1], device=device)
                w_sum_gt = torch.cumsum(w_dt_gt,dim=2)
                theta_gt = theta_0 + w_sum_gt
                c_theta_gt = torch.cos(theta_gt)
                s_theta_gt = torch.sin(theta_gt)
                vc_theta_gt = v_gt*torch.cos(theta_gt)
                vs_theta_gt = v_gt*torch.sin(theta_gt)
                pred_controls = model(occ_map)
                v_pred = pred_controls[:,v_idx]
                w_pred = pred_controls[:,w_idx]
                w_dt_pred = w_pred*torch.tensor([0.1], device=device)
                w_sum_pred = torch.cumsum(w_dt_pred,dim=2)
                theta_pred = theta_0 + w_sum_pred
                c_theta_pred = torch.cos(theta_pred)
                s_theta_pred = torch.sin(theta_pred)
                vc_theta_pred = v_pred*torch.tensor([4.13], device=device, requires_grad=True)*torch.cos(theta_pred)
                vs_theta_pred = v_pred*torch.tensor([4.13], device=device, requires_grad=True)*torch.sin(theta_pred)
                v_loss = criterion(v_pred*torch.tensor([4.13], device=device, requires_grad=True), v_gt)
                w_loss = criterion(w_pred*torch.tensor([57.2958], device=device, requires_grad=True), \
                        w_gt*torch.tensor([57.2958], device=device, requires_grad=True))
                vc_loss = criterion(vc_theta_pred, vc_theta_gt)
                vs_loss = criterion(vs_theta_pred, vs_theta_gt)
                loss = vc_loss + vs_loss #+ w_loss
                running_loss += loss.item()
                #loss = v_loss + w_loss #criterion(pred_controls, controls)
                running_v_loss += v_loss.item()
                running_w_loss += w_loss.item()
                # vloss = criterion(pred_controls, controls)
                # running_vloss += vloss.item()
        average_v_loss = running_v_loss/(v_iter)
        average_w_loss = running_w_loss/(v_iter)
        val_v_loss.append(average_v_loss)
        val_w_loss.append(average_w_loss)
        print("Epoch: {}. Validation: V Loss: {}. W Loss: {}".format(i, average_v_loss, average_w_loss))
        #scheduler.step(average_v_loss + average_w_loss) 
        scheduler.step() 
        # average_vloss = running_vloss/(v_iter)
        # val_loss.append(average_vloss)

        # print("Epoch: {}. Loss train: {}. Loss Validation: {}".format(i, average_tloss, average_vloss))
        if(average_v_loss+average_w_loss < best_vloss): # Saving best model
        #    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = args.weights_dir + 'model_{}.pt'.format(args.exp_id)
            torch.save(model.state_dict(), model_path)
            best_vloss = average_v_loss+average_w_loss
            print("########## MODEL-SAVED #########")

    ########### Finishing Epochs ##############
    plt_path = args.loss_dir + 'model_{}.png'.format(args.exp_id)
    np.save(args.loss_dir + 'model_{}_v_train_loss'.format(args.exp_id),train_v_loss)
    np.save(args.loss_dir + 'model_{}_w_train_loss'.format(args.exp_id),train_w_loss)
    np.save(args.loss_dir + 'model_{}_v_val_loss'.format(args.exp_id),val_v_loss)
    np.save(args.loss_dir + 'model_{}_w_val_loss'.format(args.exp_id),val_w_loss)
    plt.plot(train_v_loss, label="train v loss")
    plt.plot(train_w_loss, label="train w loss")
    plt.plot(val_v_loss, label="val v loss")
    plt.plot(val_w_loss, label="val w loss")
    plt.legend(loc="upper right")
    plt.savefig(plt_path)

if __name__ == "__main__":
    main()
    
