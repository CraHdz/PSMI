import torch
from torch import nn



def loss_fuc(x_ori_edit, x_atta_edit, x_ori_swap, x_atta_swap):
    mse_func = nn.MSELoss()

    edit_loss = mse_func(x_ori_edit, x_atta_edit)
    swap_loss = mse_func(x_ori_swap, x_atta_swap)

    # loss = torch.exp(1 / edit_loss) + torch.exp(1 / swap_loss)
    loss = -(edit_loss + swap_loss)
    return loss