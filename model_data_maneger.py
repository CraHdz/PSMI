from os.path import join
from core.pert_generator import PertGenerator

from data import CelebA
import simswap
from stargan.solver import Solver
import torch.utils.data as data
import argparse
import yaml
import json
from simswap.src.simswap import SimSwap 
from util import img_read, to_tensor
import cv2
from log.log import logger
# config = None
#获取配置文件


# class ObjDict(dict):
#     """
#     Makes a  dictionary behave like an object,with attribute-style access.
#     """
#     def __getattr__(self,name):
#         try:
#             return self[name]
#         except:
#             raise AttributeError(name)
#     def __setattr__(self,name,value):
#         self[name]=value

def init_simSwap(simswap_config):
    simswap = SimSwap(config=simswap_config.pipeline)
    return simswap


# def getconfig(config):
    # global config
    # if config is not None:
    #     return config
    # # with open("config.yaml", "r") as f:
    # #     config = yaml.safe_load(f)
    # #     config = ObjDict(config)
    # with open("config.json", "r") as f:
    #     config = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    # return config

#init stargan model
def init_stargan(stargan_config, celeba_data_loader):
    return Solver(celeba_loader=celeba_data_loader, rafd_loader=None, config=stargan_config)

def get_dataloader(data_path, attr_path, img_size, mode, attrs, selected_attrs, batch_size):
    data_set = CelebA(data_path, attr_path, img_size, mode, attrs, selected_attrs)
    data_loader = data.DataLoader(
        data_set, batch_size=batch_size, num_workers=0,
        shuffle=False, drop_last=False
    )
    print("The dataload length is " + str(len(data_loader)))
    # if args_attack. global_settings.num_test is None:
    #     print('Testing images:', len(test_dataset))
    # else:
    #     print('Testing images:', min(len(test_dataset), args_attack. global_settings.num_test))
    return data_loader

def get_att_image(img_path, crop_size):
    att_image  = img_read(img_path)
    cv2.resize(att_image, (crop_size, crop_size))
    att_image = to_tensor()(att_image).unsqueeze(0)
    return att_image 


def prepare(config):
    # prepare deepfake models
    # config = getconfig()
    
    logger(config.global_settings.log_dir)

    global_settings = config.global_settings
    
    print ("current mode is:" + global_settings.mode)

    pert_gen_net = PertGenerator(config.pertgenerator)

    # attgan, attgan_args = init_attGAN(args_attack)
    # attack_dataloader = init_attack_data(args_attack, attgan_args)

    # attentiongan_solver = init_attentiongan(args_attack, test_dataloader)
    # attentiongan_solver.restore_model(attentiongan_solver.test_iters)
    batch_size = 1

    if ( global_settings.mode == "train" or global_settings.mode == "test"):
        batch_size =  global_settings.batch_size

    data_loader = get_dataloader(global_settings.data_path, global_settings.attr_path, 
        global_settings.image_size, global_settings.mode, config.attgan.selected_attrs, 
        config.stargan.selected_attrs, batch_size)
    stargan = init_stargan(config.stargan, data_loader)
    stargan.restore_model(stargan.test_iters)

    simswap = init_simSwap(config.simswap)
    att_image = get_att_image(global_settings.att_image_path, global_settings.image_size)
    #inin perturbation generation network
    # transform, F, T, G, E, reference, gen_models = prepare_HiSD()
    print("Finished deepfake models initialization!")
    return data_loader, stargan, pert_gen_net, simswap, att_image
