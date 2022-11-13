import torch
import numpy as np
from torchvision.utils import save_image
import cv2
import torchvision.utils as vutils 
from torchvision import transforms

def tensor2img_denorm(tensor):
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    tensor = std * tensor.detach().cpu() + mean
    img = tensor.numpy()
    img = img.transpose(0, 2, 3, 1)[0]
    img = np.clip(img * 255, 0.0, 255.0).astype(np.uint8)
    return img


def tensor2img(tensor):
    tensor = tensor.detach().cpu().numpy()
    img = tensor.transpose(0, 2, 3, 1)[0]
    img = np.clip(img * 255, 0.0, 255.0).astype(np.uint8)
    return img

to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
to_tensor =  transforms.Compose([
        transforms.ToTensor(),
    ])

def img_read(imag_Path):
    return cv2.cvtColor(
        cv2.imread(imag_Path),
        cv2.COLOR_BGR2RGB
    )

def tensor2img_save( image_tensor, out_file, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
    if image_numpy.shape[0] == 1:  # grayscale to RGB
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    for i in range(len(mean)): #反标准化
        image_numpy[i] = image_numpy[i] * std[i] + mean[i]
    image_numpy = image_numpy * 255
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
    img_save_np(image_numpy, out_file)



def img_save(img, out_file):
    vutils.save_image(img.cpu(), out_file, nrow=1, normalize=True, range=(-1., 1.))

def img_save_np(img, out_file):
    cv2.imwrite(out_file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def checkpoint_save(model, optimizer, epoch, save_path):
    checkpoint = {"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch}
    torch.save(checkpoint, save_path)

def checkpoint_load(model, optimizer, load_path):
    model_CKPT = torch.load(load_path)
    model.load_state_dict(model_CKPT['model_state_dict'])
    # optimizer.load_state_dict(model_CKPT['optimizer_state_dict'])
    # epoch = model_CKPT['epoch']
    # return model, optimizer, epoch
