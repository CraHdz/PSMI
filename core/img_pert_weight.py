import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import cv2

def img_per_weight(img, kernel_size, device=None, interval_min=0.75, interval_max=1.5):
    if device is not None:
        img = img.clone().detach_().to(device)
    else:
        img = img.clone().detach_()
    img.requires_grad = False
    with torch.no_grad():
        img_b, img_c, img_h, img_w = img.shape

        mat_mes_loss = img.repeat(1, 1, kernel_size, kernel_size).view(
            img_b, img_c, kernel_size, img_h, kernel_size, img_w).permute(
            0, 1, 3, 5, 2, 4).contiguous().view(
            img_b, img_c, img_h, img_w, kernel_size*kernel_size)


        padding_size = [kernel_size // 2 for i in range(4)]
        ReplicationPad2d = nn.ReplicationPad2d(padding=padding_size)
        img = ReplicationPad2d(img)

        #make slide windows opt in image
        img_unfs = F.unfold(img, 
            kernel_size=kernel_size, dilation=1, stride=1)
        
        B, C_kh_kw, H_W = img_unfs.size()
        img_unfs = img_unfs.permute(0, 2, 1)
        img_unfs = img_unfs.view(B, img_h, img_w, img_c, 
            kernel_size, kernel_size).permute(0, 3, 1, 2, 4, 5).contiguous().view(
            img_b, img_c, img_h, img_w, kernel_size*kernel_size)
        
        #the img_mse shape is (b, c, h, w)
        img_mse = torch.mean((img_unfs - mat_mes_loss) ** 2, dim=4) ** 0.5
    
        img_mse_max_w, _ = torch.max(img_mse, dim=3)
        img_mse_max_h_w, _ = torch.max(img_mse_max_w, keepdim=True, dim=2)
        img_mse_max = img_mse_max_h_w.repeat(1, 1, img_h*img_w).view(img_b, img_c, img_h, img_w)
       
        
        img_mse_min_w, _ = torch.min(img_mse, dim=3)
        img_mse_min_h_w, _ = torch.min(img_mse_min_w, dim=2, keepdim=True)
        img_mse_min = img_mse_min_h_w.repeat(1, 1, img_h*img_w).view(img_b, img_c, img_h, img_w)
        

        k = (interval_max - interval_min) / (img_mse_max - img_mse_min).clamp_(1e-8)
        img_pert_wight_nat = interval_min + k * (img_mse - img_mse_min)

        return img_pert_wight_nat


if __name__ == "__main__":
    x = torch.randn(2, 3, 4, 4).float()
    # x = x.view(2,3,4,4)
    result = img_per_weight(x, 3)
    # img = cv2.cvtColor(cv2.imread("./../demo/image_att.jpg"),
    #     cv2.COLOR_BGR2RGB
    # )

    # tf = transforms.Compose(
    #             [
    #                 transforms.ToTensor(),
    #                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #             ]
    #         )

    # img = tf(img).unsqueeze(0)
    # result = img_per_weight(img, 3)

