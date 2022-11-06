import torch
from torch import nn
from torch.nn import functional as F

def img_per_weight(img, kernel_size, device=None):
    if device is not None:
        img = img.clone().detach_().to(device)
    else:
        img = img.clone().detach_()
    with torch.no_grad():
        img_b, img_c, img_h, img_w = img.shape

        mat_mes_loss = img.repeat(1, 1, kernel_size, kernel_size).view(
            img_b, img_c, kernel_size, img_h, kernel_size, img_w).permute(
            0, 1, 3, 5, 2, 4).contiguous().view(
            img_b, img_c, img_h, img_w, kernel_size*kernel_size)

        padding_size = [kernel_size // 2 for i in range(4)]
        ReplicationPad2d = nn.ReplicationPad2d(padding=padding_size)
        img = ReplicationPad2d(img)

        img_unfs = F.unfold(img, 
            kernel_size=kernel_size, dilation=1, stride=1)
        
        B, C_kh_kw, H_W = img_unfs.size()
        img_unfs = img_unfs.permute(0, 2, 1)
        img_unfs = img_unfs.view(B, img_h, img_w, img_c, 
            kernel_size, kernel_size).permute(0, 3, 1, 2, 4, 5).contiguous().view(
            img_b, img_c, img_h, img_w, kernel_size*kernel_size)
        

        img_mse = torch.mean((img_unfs - mat_mes_loss) ** 2, dim=4) ** 0.5

        return img_mse


if __name__ == "__main__":
    x = torch.arange(0, 2*3*4*4).float()
    x = x.view(2,3,4,4)
    result = img_per_weight(x, 3)
    print(result)
