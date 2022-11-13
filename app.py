from email.mime import application, image
from re import U
import model_data_init
import os

import torch
import torchvision.utils as vutils
import torch.nn.functional as F

from torchvision import transforms
import hydra
# from omegaconf import DictConfig
from simswap.src.simswap import SimSwap
import cv2
import numpy as np
import attacks
import kornia
from simswap.src.FaceAlign.face_align import align_face, inverse_transform_batch
from util import tensor2img
from core.pert_generator import PertGenerator
from util import *
from log.log import logger
from core.loss_function import loss_fuc
import argparse
import yaml
import json
from torch.cuda.amp import  autocast,GradScaler
from core.PSMI_model import PSMI
from model_data_init import *


class PerbApplication:
    def __init__(self, config) -> None:
        self.config = config

        data_loader, att_image, id_image = prepare(self.config)
       
        self.data_loader = data_loader

        #att_image is the base image for swap face, the pixel range is (0,1)
        self.att_image = att_image

        self.device =  torch.device(config.global_settings.device)
        self.mode = config.global_settings.mode

        #init net
        self.stargan = init_stargan(config.stargan, data_loader)
        self.simswap_net = init_simSwap(config.simswap)
        self.face_id_net = init_face_id_net(config.face_id_net)
        self.pert_gen_net = init_per_gen_net(config.pertgenerator)
        
        # if self.mode == "valid":
        #     self.face_detector_net = init_face_detection_net(config.face_detector_net)
        #     self.bise_net = init_bias_net(config.bise_net)

        #     self.PSMI = PSMI(att_image, config.PSMI, self.pert_gen_net, self.stargan.G, self.simswap_net,
        #         self.face_id_net, self.face_detector_net, self.bise_net).to(self.device)
        # else: 
        #     self.PSMI = PSMI(att_image, config.PSMI, self.pert_gen_net, self.stargan.G, self.simswap_net,
        #         self.face_id_net).to(self.device)
        self.face_detector_net = init_face_detection_net(config.face_detector_net)
        # self.bise_net = init_bias_net(config.bise_net)
        self.bise_net = None

        self.PSMI = PSMI(id_image, att_image, config.PSMI, self.pert_gen_net, self.stargan.G, self.simswap_net,
                self.face_id_net, self.face_detector_net, self.bise_net).to(self.device)
        
        #train config
        self.epochs = config.global_settings.epochs
        self.checkpoint_file = config.global_settings.checkpoint_file
        self.device_ids = config.global_settings.device_ids
        self.model_path = config.global_settings.model_path
        
        #checkpoint config
        self.checkpoint_path = config.global_settings.checkpoint_path
        self.is_load_checkpoint = config.global_settings.is_load_checkpoint
        self.checkpoint_file = config.global_settings.checkpoint_file
        
        #DataParallel
        if self.config.global_settings.is_dataParallel:
            self.PSMI = torch.nn.DataParallel(self.PSMI, device_ids=self.device_ids).cuda()
            self.pert_gen_net = self.PSMI.module.pert_gen_net
        else:
            self.pert_gen_net = self.PSMI.pert_gen_net
        

    def get_CUMA_perturbation(self):
        return torch.load(self.config.global_settings.universal_perturbation_path)


    def train(self):
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0

        optimizer = torch.optim.Adam(self.pert_gen_net.parameters(), lr=0.01, weight_decay=0.005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

        if self.is_load_checkpoint:
            checkpoint_load(self.pert_gen_net, optimizer, self.checkpoint_file)
        scaler = GradScaler()


        for epoch in range(self.epochs):
            loss_epoch = 0
            for idx, (imgs, c_org) in enumerate(self.data_loader):
                #the image.shape is b, c, h, w, and the number is limit in [-1, 1] 
                
                imgs = imgs.to(self.device)
    
                c_org = c_org.to(self.device)
                c_org = c_org.type(torch.float)
                
                optimizer.zero_grad()

                c_trg_list = self.stargan.create_labels(c_org, self.stargan.c_dim, self.stargan.dataset, self.stargan.selected_attrs)
                with autocast():
                    x_noattack_list_edit, x_attack_list_edit, x_noatta_id_lats, x_attack_id_lats, x_noatta_swap, x_attaswap = self.PSMI(imgs, c_trg_list)
                    loss = loss_fuc(x_noattack_list_edit, x_attack_list_edit, x_noatta_id_lats, x_attack_id_lats, x_noatta_swap, x_attaswap)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_epoch = loss_epoch + loss
                logger.info("epoch:" + str(epoch) + ",   loss:" + str(loss))
                # if idx % 10 == 0:
                #     self.PSMI.print_pert_gen_net_train_info()
            scheduler.step()
            if epoch % 10 == 0:
                checkpoint_path = self.checkpoint_path + "/point_epoch_{}.cp".format(epoch)
                checkpoint_save(self.pert_gen_net, optimizer, epoch, checkpoint_path)
            logger.info("this epoch loss mean:" + str(loss_epoch / idx))
        torch.save(self.pert_gen_net, self.model_path)

        # print('stargan {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))



    def test(self):
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0
        
        model_CKPT = torch.load(self.model_path)
        self.PSMI.pert_gen_net.load_state_dict(model_CKPT['model_state_dict'])
        # self.PSMI.print_pert_gen_net_train_info()
        with torch.no_grad():
            for idx, (imgs, c_org) in enumerate(self.data_loader):
                #the image.shape is b, c, h, w, and the number is limit in [-1, 1] 
                imgs = imgs.to(self.device)
                
                c_org = c_org.to(self.device)
                c_org = c_org.type(torch.float)
                c_trg_list = self.stargan.create_labels(c_org, self.stargan.c_dim, self.stargan.dataset, self.stargan.selected_attrs)

                self.PSMI.pert_gen_net = self.PSMI.pert_gen_net.to(self.device)
                perturbs = self.PSMI.pert_gen_net(imgs)
                # perturbs = self.get_CUMA_perturbation()

                x_noattack_list_stargan, x_attack_list_stargan = self.PSMI.face_edit_attck(imgs, perturbs, c_trg_list)
                
                

                x_noattack_list_simswap, x_attack_list_simswap = self.PSMI.face_swap_attack(imgs, perturbs)
                x_total_list_stargan = [imgs, imgs + perturbs]
                for j in range(len(x_noattack_list_stargan)):
                    # gen_noattack = x_noattack_list_stargan[j]
                    # gen = x_attack_list_stargan[j]

                    x_total_list_stargan.append(x_noattack_list_stargan[j])
                    x_total_list_stargan.append(x_attack_list_stargan[j])

                    # l1_error += F.l1_loss(gen, gen_noattack)
                    # l2_error += F.mse_loss(gen, gen_noattack)
                    # l0_error += (gen - gen_noattack).norm(0)
                    # min_dist += (gen - gen_noattack).norm(float('-inf'))
                    # if F.mse_loss(gen, gen_noattack) > 0.05:
                    #     n_dist += 1
                    # n_samples += 1
                
                # x_total_list_simswap = [imgs, imgs + perturbs]
                x_total_list_simswap = [imgs*0.5 + 0.5, (imgs + perturbs) * 0.5 + 0.5,
                    x_noattack_list_simswap, x_attack_list_simswap]
                
                # save origin image
                x_concat = torch.cat(x_total_list_stargan, dim=3)
                out_file = os.path.join(self.config.global_settings.demo_result + '/stargan.jpg')
                vutils.save_image(x_concat, out_file, nrow=1, normalize=True, range=(-1., 1.))

                x_concat = torch.cat(x_total_list_simswap, dim=3)
                out_file = os.path.join(self.config.global_settings.demo_result + '/simswap.jpg')
                vutils.save_image(x_concat, out_file, nrow=1, normalize=False, range=(0., 1.))
                # loss_fuc(x_noattack_list_stargan,x_attack_list_stargan,
                #     x_noattack_list_simswap, x_attack_list_simswap, )
                break

    def vaild(self):

        # perturbs = self.get_CUMA_perturbation().to(self.device)
        start_idx = 182636
        with torch.no_grad():
            for idx, (imgs, c_org) in enumerate(self.data_loader):
                # imgs = [transforms.ToPILImage()(img) for img in imgs]
                #get perturbation and gen image
                # print(type(id_images_align))
                # print(len(id_images_align))
                for img in imgs:
                    id_images_align, id_transforms, _ = self.PSMI.run_detect_align(
                        image=img, for_id=False, crop_size=256)
                    if id_images_align is not None:
                        img_save(id_images_align[0], "./images/{}.jpg".format(start_idx))
                    start_idx = start_idx + 1
                break


    def attack__demo(self):
        pgd_attack = attacks.LinfPGDAttack(model=self.PSMI.face_id_net, device="cuda" , feat=None)
        
        id_image_path = self.config.global_settings.demo_image_id
        att_image_path = self.config.global_settings.demo_image_att

        id_image = img_read(str(id_image_path))
        att_image = img_read(str(att_image_path))

        tf = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

        id_image = tf(id_image).unsqueeze(0)
        #in simwap model, att_image don't need normalize
        att_image = tf(att_image).unsqueeze(0)
        #get face from id_iamge and att_image
        id_image_align, id_transforms, _ = self.PSMI.run_detect_align(
            image=id_image, for_id=False, crop_size=256)
        att_image_align, att_transforms, _ = self.PSMI.run_detect_align(
            image=att_image, for_id=False, crop_size=256)
        
        if id_image_align is None or att_image is None:
            print("Don't detect face")
            return
        

        # id_image_with_pert, pert = pgd_attack.perturb_simswap(
        #     id_image_align, att_image_align, swap_image_with_no_perb, 
        #     self.PSMI.face_id_net, self.PSMI.face_swap_net
        # )
        id_image_align = id_image_align.to(self.device)

        model_CKPT = torch.load(self.model_path)
        self.PSMI.pert_gen_net.load_state_dict(model_CKPT['model_state_dict'])
        pert = self.PSMI.pert_gen_net(id_image_align) * 3
        # pert = self.get_CUMA_perturbation()
        
        id_image_pert = id_image_align + pert
        id_image_pert = id_image_pert.to(self.device)

        ori_id_image_pert = self.PSMI.restore_id_image(id_image, id_image_align, id_transforms, id_image_pert)
        ori_id_image_pert = transforms.Resize(256)(ori_id_image_pert).to(self.device)
        # ori_id_image_pert =  transforms.Resize(256)(id_image.to(self.device) +  transforms.Resize((393, 400))(pert)).to(self.device)
        id_image = transforms.Resize(256)(id_image).to(self.device)
        # image_with_perb, perb = pgd_attack.perturb_simswap_att(
        #     id_images, att_image_align, swap_image_with_no_perb, 
        #     self.PSMI.face_id_net, self.PSMI.face_swap_net
        # )
        
        # c_org = torch.zeros(id_images.shape[0], 5)
        # c_trg_list = self.stargan.create_labels(c_org, self.stargan.c_dim, self.stargan.dataset, self.stargan.selected_attrs)
        # image_with_perb, perb = pgd_attack.perturb_stargan((att_image_align - 0.5) * 2, c_trg_list, self.PSMI.face_edit_net)
       
        # image_with_perb = self.PSMI.restore_id_image(id_image, id_images_align, id_transforms, image_with_perb)
        # tf_r = transforms.Compose(
        #         [
        #             transforms.ToTensor(),
        #             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        #             transforms.Resize(256)
        #         ]
        #     )
        # image_with_perb = tf_r(image_with_perb)
        # opt = None
        # checkpoint_load(self.PSMI.pert_gen_net, opt, self.checkpoint_file)
        # pert = self.PSMI.pert_gen_net(id_images.to(self.device))
        # image_with_perb = id_images + pert.cpu()

        ######################################################################
        #face_edit_attack and save
        ######################################################################

        c_org = torch.zeros(id_image_align.shape[0], 5).cuda()
        c_trg_list = self.stargan.create_labels(c_org, self.stargan.c_dim, self.stargan.dataset, self.stargan.selected_attrs)
        x_noattack_edit, x_attack_edit = self.PSMI.face_edit_attck_pert(id_image, ori_id_image_pert, c_trg_list)
        

        x_total_list_stargan = [id_image, ori_id_image_pert]
        for j in range(len(x_noattack_edit)):
            x_total_list_stargan.append(x_noattack_edit[j])
            x_total_list_stargan.append(x_attack_edit[j])
        x_concat = torch.cat(x_total_list_stargan, dim=3)
        out_file = os.path.join(self.config.global_settings.demo_result + '/stargan_demo.jpg')
        vutils.save_image(x_concat, out_file, nrow=1, normalize=True, range=(-1., 1.))
        #######################################################################
        #end
        ########################################################################
        # att_image = att_image_align.to(self.device) + self.get_CUMA_perturbation().to(self.device) * 0.5
        # origin_perb_path = self.config.global_settings.demo_result + "/origin_perb_att.jpg"
        # image_att = transforms.ToPILImage()(att_image_align[0].cpu().detach())
        # image_att.save(origin_perb_path)
        # origin_perb_path = self.config.global_settings.demo_result + "/origin_perb.jpg"
        # img_save(image_with_perb, origin_perb_path)

        ###################################################################################
        #face swap attack and save
        ###################################################################################
        # #the id image input is tensor image range(-1,1), output is tensor 512-d face_id
        # id_latent_with_no_perb: torch.Tensor = self.PSMI.face_id_net(id_image_align, need_trans=False)

        # #the output of swap_net is tensor image ,pixle range is (0, 1)
        # swap_image_with_no_perb: torch.Tensor = self.PSMI.face_swap_net(
        #     att_image_align, id_latent_with_no_perb, need_trans=False)
        # print(type(swap_image_with_no_perb))
        # img_no_perb_path = self.config.global_settings.demo_result + "/swapped_img_no_perb.jpg"   
        # image_no_perb = transforms.ToPILImage()(swap_image_with_no_perb[0].cpu().detach())
        # image_no_perb.save(img_no_perb_path)

        # id_latent_with_perb = self.PSMI.face_id_net(id_image_pert, need_trans=False)
        # swap_image_with_perb = self.PSMI.face_swap_net(att_image_align, id_latent_with_perb, need_trans=False)

        # img_per_path = self.config.global_settings.demo_result + "/swapped_img_perb.jpg"
        # image_perb = transforms.ToPILImage()(swap_image_with_perb[0].cpu().detach())
        # image_perb.save(img_per_path)

        ###################################################################################
        #end
        ###################################################################################
        # loss_fuc(x_noattack_list_stargan, x_attack_list_stargan, swap_image_with_no_perb, swap_image_with_perb)
        

    


def main():
    #vaild(config)
    config_Path = './configs/config.yaml'
    config = model_data_init.get_config(config_Path)
    app  = PerbApplication(config)
    # app.attack__demo()
    app.train()
    # app.test()
    # app.vaild()

import torch.distributed as dist
import torch.utils.data.distributed
from torch.multiprocessing import Process
if __name__ == "__main__" :
    main()
    # size = 3
    # processes = []
    # for rank in range(size):
    #     p = Process(target=main, args=(rank,))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()
