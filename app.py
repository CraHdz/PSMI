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

        data_loader, att_image = prepare(self.config)
       
        self.data_loader = data_loader

        #att_image is the base image for swap face 
        self.att_image = att_image

        self.device =  torch.device(config.global_settings.device)

        #init net
        self.stargan_net = init_stargan(config.stargan, data_loader)
        self.simswap = init_simSwap(config.simswap)
        self.face_detector_net = init_face_detection_net(config.face_detector_net)
        self.face_id_net = init_face_id_net(config.face_id_net)
        self.pert_gen_net = init_per_gen_net(config.pertgenerator)
        
        self.PSMI = PSMI(att_image, self.pert_gen_net, self.stargan_net.G, self.simswap.simswap_net
            , self.face_id_net, self.face_detector_net).to(self.device)
        

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


    def attack_face_swap_demo(self):
        pgd_attack = attacks.LinfPGDAttack(model=self.PSMI.face_id_net, device="cuda" , feat=None)
        

        
        id_image_path = self.config.global_settings.demo_image_id
        att_image_path = self.config.global_settings.demo_image_att

        id_image = img_read(str(id_image_path))
        att_image = img_read(str(att_image_path))

        #get face from id_iamge and att_image
        #the return images type is list [face ndarry, ....] 
        id_images_align, id_transforms, _ = self.PSMI.run_detect_align(
            image=id_image, for_id=False, crop_size=256)
        att_image, att_transforms, _ = self.PSMI.run_detect_align(
            image=att_image, for_id=False, crop_size=256)
        
        if id_image is None or att_image is None:
            print("Don't detect face")
            return
        # for img in id_images_align:
        #     img_save_np(img, self.config.global_settings.demo_result + "/origin_id_{}.jpg".format(id_images_align.index(img)))
        # for img in att_image:
        #     img_save_np(img, self.config.global_settings.demo_result + "/origin_att_{}.jpg".format(att_image.index(img)))
        tf = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        # #id_images shape [b,c,h,w]
        id_images = torch.stack(
            [tf(id_image) for id_image in id_images_align], dim=0
        )
        #in simwap model, att_image don't need normalize
        att_image = torch.stack(
            [transforms.ToTensor()(img) for img in att_image], dim=0
        )
        
        
        id_latent_with_no_perb: torch.Tensor = self.PSMI.face_id_net(id_images, need_trans=False)
        swap_image_with_no_perb: torch.Tensor = self.PSMI.face_swap_net(
            att_image, id_latent_with_no_perb, need_trans=False)

        img_no_perb_path = self.config.global_settings.demo_result + "/swapped_img_no_perb.jpg"   
        # image_no_perb = transforms.ToPILImage()(swap_image_with_no_perb[0].cpu().detach())
        # image_no_perb.save(img_no_perb_path)
        img_save(swap_image_with_no_perb, img_no_perb_path)

        image_with_perb, perb = pgd_attack.perturb_simswap(
            id_images, att_image, swap_image_with_no_perb, 
            self.PSMI.face_id_net, self.PSMI.face_swap_net
        )
        # opt = None
        # checkpoint_load(self.PSMI.pert_gen_net, opt, self.checkpoint_file)
        # pert = self.PSMI.pert_gen_net(id_images.to(self.device))
        # image_with_perb = id_images + pert.cpu()


        # image_with_perb = id_images.to(self.device) + get_CUMA_perturbation(self.config).to(self.device)
        origin_perb_path = self.config.global_settings.demo_result + "/origin_perb.jpg"
        img_save(image_with_perb, origin_perb_path)


        id_latent_with_perb = self.PSMI.face_id_net(image_with_perb, need_trans=False)
        print(torch.nn.MSELoss()(id_latent_with_perb, id_latent_with_no_perb))
        swap_image_with_perb = self.PSMI.face_swap_net(att_image, id_latent_with_perb, need_trans=False)

        img_per_path = self.config.global_settings.demo_result + "/swapped_img_perb.jpg"
        image_perb = transforms.ToPILImage()(swap_image_with_perb[0].cpu().detach())
        image_perb.save(img_per_path)

        # self.restore_id_image(id_image, id_images_align, id_transforms, image_with_perb)

    
    def face_swap_attack(self, id_images, pertub):
        if id_images is None or self.att_image is None:
            print("Don't detect face")
            return
        
        with torch.no_grad():
            id_latent_no_pert: torch.Tensor = self.face_id_net(id_images, need_trans=False)
            
            swap_images_no_pert: torch.Tensor = self.simswap_net(
                self.att_image, id_latent_no_pert, need_trans=False)
        
        id_images_pert = id_images + pertub

        id_latent_with_pert: torch.Tensor = self.face_id_net(id_images_pert, need_trans = False)
        swap_images_with_pert: torch.Tensor = self.simswap_net(
            self.att_image, id_latent_with_pert, need_trans=False)

        return swap_images_no_pert, swap_images_with_pert
    

    def face_edit_attck(self, x_real, c_org, perturb):

        # Load the trained generator.
        # self.restore_model(self.test_iters)

        # Prepare input images and target domain labels.
        c_trg_list = self.stargan.create_labels(c_org, self.stargan.c_dim, self.stargan.dataset, self.stargan.selected_attrs)
        # Translated images.
        x_adv = x_real + perturb
        x_attack_list = []
        x_noattack_list = []

        for idx, c_trg in enumerate(c_trg_list):
            # with torch.no_grad():
            #     gen_noattack, gen_noattack_feats = self.stargan_net(x_real, c_trg)
            #     gen, gen_feats = self.stargan_net(x_adv, c_trg)
            #     x_attack_list.append(gen)
            #     x_noattack_list.append(gen_noattack)
            with torch.no_grad():
                gen_noattack, gen_noattack_feats = self.stargan_net(x_real, c_trg)
            gen, gen_feats = self.stargan_net(x_adv, c_trg)
            x_attack_list.append(gen)
            x_noattack_list.append(gen_noattack)

        return x_noattack_list, x_attack_list

    def train(self):
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0

       
        # perturbs = self.get_CUMA_perturbation(self.config)
        optimizer = torch.optim.Adam(self.pert_gen_net.parameters(), lr=0.1, weight_decay=0.005)
        if self.is_load_checkpoint:
            checkpoint_load(self.pert_gen_net, optimizer, self.checkpoint_file)
        scaler = GradScaler()
        for epoch in range(self.epochs):
            loss_epoch = 0
            for idx, (imgs, att_a, c_org) in enumerate(self.data_loader):
                #the image.shape is b, c, h, w, and the number is limit in [-1, 1] 
                
                imgs = imgs.to(self.device)
    
                c_org = c_org.to(self.device)
                c_org = c_org.type(torch.float)
                
                optimizer.zero_grad()
                # self.pert_gen_net = self.pert_gen_net.to(self.device)
                # perturbs = self.pert_gen_net(imgs)
                
                # x_noattack_list_stargan, x_attack_list_stargan = self.face_edit_attck(imgs, c_org, perturbs)
                # x_noattack_list_stargan = torch.stack(
                #     x_noattack_list_stargan, dim = 0
                # )
                # x_attack_list_stargan = torch.stack(
                #     x_attack_list_stargan, dim=0
                # )

                # x_noattack_list_simswap, x_attack_list_simswap = self.face_swap_attack(imgs, perturbs)
                c_trg_list = self.stargan.create_labels(c_org, self.stargan.c_dim, self.stargan.dataset, self.stargan.selected_attrs)
                with autocast():
                    x_noattack_list_edit, x_attack_list_edit,x_noatta_id_lats, x_attack_id_lats, img_ori, img_pert = self.PSMI(imgs, c_trg_list)
                    loss = loss_fuc(x_noattack_list_edit, x_attack_list_edit, x_noatta_id_lats, x_attack_id_lats, img_ori, img_pert)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_epoch = loss_epoch + loss
                logger.info("epoch:" + str(epoch) + ",   loss:" + str(loss))
                # if idx % 10 == 0:
                #     self.PSMI.print_pert_gen_net_train_info()
            if epoch % 7 == 0:
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
        self.PSMI.print_pert_gen_net_train_info()
        with torch.no_grad():
            for idx, (imgs, att_a, c_org) in enumerate(self.data_loader):
                #the image.shape is b, c, h, w, and the number is limit in [-1, 1] 
                imgs = imgs.to(self.device)
                # imgs = images.to(self.device)
                #save dateset image
                # for index in range(imgs.shape[0]):
                #     out_file = os.path.join(self.config.global_settings.demo_result + '/celebA/celebA_{}.jpg'.format(index))
                #     img_save(imgs[index], out_file)
                
                c_org = c_org.to(self.device)
                c_org = c_org.type(torch.float)
                c_trg_list = self.stargan.create_labels(c_org, self.stargan.c_dim, self.stargan.dataset, self.stargan.selected_attrs)

                self.PSMI.pert_gen_net = self.PSMI.pert_gen_net.to(self.device)
                perturbs = self.PSMI.pert_gen_net(imgs)

                x_noattack_list_stargan, x_attack_list_stargan = self.PSMI.face_edit_attck(imgs, perturbs, c_trg_list)
                x_noattack_list_stargan = torch.stack(
                    x_noattack_list_stargan, dim = 0
                )
                x_attack_list_stargan = torch.stack(
                    x_attack_list_stargan, dim=0
                )

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
                x_total_list_simswap = [x_noattack_list_simswap, x_attack_list_simswap]
                # for j in range(len(x_noattack_list_simswap)):
                    # gen_noattack = x_noattack_list_simswap[j]
                    # gen = x_noattack_list_stargan[j]

                    # x_total_list_simswap.append(x_noattack_list_simswap[j])
                    # x_total_list_simswap.append(x_attack_list_simswap[j])


                    # l1_error += F.l1_loss(gen, gen_noattack)
                    # l2_error += F.mse_loss(gen, gen_noattack)
                    # l0_error += (gen - gen_noattack).norm(0)
                    # min_dist += (gen - gen_noattack).norm(float('-inf'))
                    # if F.mse_loss(gen, gen_noattack) > 0.05:
                    #     n_dist += 1
                    # n_samples += 1

                # save origin image
                x_concat = torch.cat(x_total_list_stargan, dim=3)
                out_file = os.path.join(self.config.global_settings.demo_result + '/stargan.jpg')
                vutils.save_image(x_concat, out_file, nrow=1, normalize=True, range=(-1., 1.))

                x_concat = torch.cat(x_total_list_simswap, dim=3)
                out_file = os.path.join(self.config.global_settings.demo_result + '/simswap.jpg')
                vutils.save_image(x_concat, out_file, nrow=1, normalize=False, range=(0., 1.))
                break

    def vaild(self):

        perturbs = self.get_CUMA_perturbation().to(self.device)

        with torch.no_grad():
            for idx, (imgs, att_a, c_org) in enumerate(self.data_loader):
                imgs = imgs.to(self.device)
                
                c_org = c_org.to(self.device)
                c_org = c_org.type(torch.float)

                #get perturbation and gen image
                x_noattack_list, x_attack_list = self.face_edit_attck(imgs, c_org, perturbs)
                x_total_list = [imgs, imgs + perturbs]
                for j in range(len(x_attack_list)):
                    gen_noattack = x_noattack_list[j]
                    gen = x_attack_list[j]

                    x_total_list.append(x_noattack_list[j])
                    x_total_list.append(x_attack_list[j])

                
                # # save origin image
                # out_file = config.global_settings.result_path + '/stargan_original.jpg'
                # vutils.save_image(imgs.cpu(), out_file, nrow=1, normalize=True, range=(-1., 1.)) 
                x_concat = torch.cat(x_total_list, dim=3)
                # out_file = os.path.join(config.global_settings.result_path + '/stargan_gen_{}.jpg'.format(idx))
                # out_file = os.path.join(config.global_settings.result_path + '_per/stargan_per_{}.jpg'.format(idx))
                out_file = os.path.join(self.config.global_settings.demo_result + '/stargan_vailed_{}.jpg'.format(idx))
                print(out_file)
                vutils.save_image(x_concat, out_file, nrow=1, normalize=True, range=(0, 1.))
                
                #gen deepfake image and adversarial example
                # for j in range(len(x_attack_list)):
                #     #save deepfake image
                #     gen_noattack = x_noattack_list[j]
                #     out_file = config.global_settings.result_path + '/stargan_gen_{}.jpg'.format(j)
                #     vutils.save_image(gen_noattack, out_file, nrow=1, normalize=True, range=(-1., 1.))

                #     # save deepfake adversarial example
                #     gen = x_fake_list[j]
                #     out_file = config.global_settings.result_path + '/stargan_advgen_{}.jpg'.format(j)
                #     vutils.save_image(gen, out_file, nrow=1, normalize=True, range=(-1., 1.))
                break


    


def main():
    #vaild(config)
    config_Path = './configs/config.yaml'
    config = model_data_init.get_config(config_Path)
    app  = PerbApplication(config)
    app.attack_face_swap_demo()
    # app.train()
    # app.test()
    # app.vaild()

if __name__ == "__main__" :
    main()