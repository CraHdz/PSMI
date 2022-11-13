import torch
from torch import nn
import torch.nn.functional as F


def loss_fuc(x_ori_edit, x_atta_edit, x_ori_id, x_atta_id, x_ori_swap, x_atta_swap, img_ori=None, img_pert=None):
    edit_loss = 0
    x_ori_edit = x_ori_edit.clone().detach_()
    # x_atta_edit = x_atta_edit.clone().detach_()
    # img_ori = img_ori.clone().detach_()
    edit_loss = F.mse_loss(x_ori_edit, x_atta_edit)
    # for i in range(len(x_ori_edit)):
    #     img_change_ori_edit = img_ori - x_ori_edit[i]
    #     img_change_pert_edit  = img_pert - x_atta_edit[i]
    #     # edit_loss = mse_func(torch.abs(img_change_ori_edit.sign()) * img_change_pert_edit, 
    #     #                 img_change_ori_edit) + edit_loss
    #     edit_loss = mse_func(img_change_pert_edit, 
    #                     img_change_ori_edit) + edit_loss

    # img_change_ori_swap = img_ori - x_ori_swap
    # img_change_pert_swap  = img_pert - x_atta_swap
    # swap_loss = mse_func(torch.abs(img_change_ori_swap.sign()) * img_change_pert_swap ,  img_change_ori_swap)
    id_loss = torch.mean(F.cosine_similarity(x_atta_id, x_ori_id, dim=1))

    swap_loss = F.mse_loss(x_atta_swap, x_ori_swap)
    
    loss = -torch.log(edit_loss) + id_loss - torch.log(swap_loss)
    log = "edit_loss:" + str(edit_loss.float()) + ", id_loss:" + str(id_loss.float()) + ", swap_loss:" + str(swap_loss)
    # loss = -torch.log(edit_loss)
    # loss = torch.exp(1 / edit_loss)
    # log = "edit_loss:" + str(edit_loss.float())
    print(log) 
    return loss