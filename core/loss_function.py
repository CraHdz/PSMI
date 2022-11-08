import torch
from torch import nn



def loss_fuc(x_ori_edit, x_atta_edit, x_ori_swap, x_atta_swap, img_ori, img_pert):
    mse_func = nn.MSELoss()
    edit_loss = 0
    x_ori_edit = x_ori_edit.clone().detach_()
    x_atta_edit = x_atta_edit.clone().detach_()
    img_ori = img_ori.clone().detach_()
    for i in range(len(x_ori_edit)):
        img_change_ori_edit = img_ori - x_ori_edit[i]
        img_change_pert_edit  = img_pert - x_atta_edit[i]
        edit_loss = mse_func(torch.abs(img_change_ori_edit.sign()) * img_change_pert_edit, 
                        img_change_ori_edit) + edit_loss

    # img_change_ori_swap = img_ori - x_ori_swap
    # img_change_pert_swap  = img_pert - x_atta_swap
    # swap_loss = mse_func(torch.abs(img_change_ori_swap.sign()) * img_change_pert_swap ,  img_change_ori_swap)
    # swap_loss = mse_func(x_atta_swap, x_ori_swap)
    
    # loss = -torch.log(edit_loss / 32.0) - torch.log(swap_loss)
    # log = "edit_loss:" + str(edit_loss.float()) + ", swap_loss:" + str(swap_loss.float())
    # loss = -torch.log(edit_loss)
    loss = torch.exp(- edit_loss)
    log = "edit_loss:" + str(edit_loss.float())
    print(log) 
    return loss