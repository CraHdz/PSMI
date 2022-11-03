from email.mime import application, image
from re import U
import model_data_maneger
import os

import torch
import torchvision.utils as vutils
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
import hydra
# from omegaconf import DictConfig
from simswap.src.simswap import SimSwap
import cv2
import numpy as np
import attacks
import kornia
from simswap.src.FaceAlign.face_align import align_face, inverse_transform_batch
from util import tensor2img, tensor2img_denorm
class PerbApplication:
    def __init__(self, config) -> None:
        self.config = config

        data_loader, stargan, pert_gen_net, simswap = model_data_maneger.prepare(self.config)
        self.data_loader = data_loader
        self.stargan = stargan
        self.per_gen_net = pert_gen_net
        self.simswap = simswap
        self.device = "cuda:0"
        

    def get_CUMA_perturbation(self):
        return torch.load(self.config.global_settings.universal_perturbation_path)

    def to_tensor_norm(self):
        return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        
    def to_tensor(self):
        return transforms.Compose([
            transforms.ToTensor(),
        ])

    def img_read(self, imag_Path):
        return cv2.cvtColor(
            cv2.imread(imag_Path),
            cv2.COLOR_BGR2RGB
        )

    def tensor2img_save(self, image_tensor, out_file, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)): #反标准化
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
        self.img_save_np(image_numpy, out_file)



    def img_save(self, img, out_file):
        vutils.save_image(img.cpu(), out_file, nrow=1, normalize=True, range=(-1., 1.))

    def img_save_np(self, img, out_file):
        cv2.imwrite(out_file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def vaild(self):
        
        # model_data_maneger.getconfig("config.yaml")

        #att_gan
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0

        perturbs = self.get_CUMA_perturbation(self.config)

        tf = self.to_tensor_norm()
        
        image = self.img_read(self.config.global_settings.demo_images)

        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        # images = simswap.run_detect_align(image=image, for_id=False)

        # H, W = image.shape[0], image.shape[1]
        # image = tf(image).unsqueeze(0)
        # image = transforms.Resize([H, W])(image) 
        
        # image = tensor2img(image.squeeze(0))

        images = None
        # images, att_transforms, _ = simswap.run_detect_align(image=image, for_id=False, crop_size=256)
        
        
        if images is None:
            print("Don't detect face")
            images = (cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC), )

        images = torch.stack(
                [tf(x) for x in images], dim=0
            )
        for idx, (imgs, att_a, c_org) in enumerate(self.data_loader):
            imgs = imgs.cuda() if self.config.global_settings.gpu else imgs
            # imgs = images.cuda() if config.global_settings.gpu else images
            for index in range(imgs.shape[0]):
                out_file = os.path.join(self.config.global_settings.demo_result + '/celebA/celebA_{}.jpg'.format(index))
                self.img_save(imgs[index], out_file)
            c_org = c_org.cuda() if self.config.global_settings.gpu else c_org
            c_org = c_org.type(torch.float)

            #get perturbation and gen image
        
            x_noattack_list, x_attack_list = self.stargan.test_universal_model_level(imgs, c_org, perturbs, self.config.stargan)

            x_total_list = [imgs, imgs + perturbs]
            for j in range(len(x_attack_list)):
                gen_noattack = x_noattack_list[j]
                gen = x_attack_list[j]

                x_total_list.append(x_noattack_list[j])
                x_total_list.append(x_attack_list[j])

                l1_error += F.l1_loss(gen, gen_noattack)
                l2_error += F.mse_loss(gen, gen_noattack)
                l0_error += (gen - gen_noattack).norm(0)
                min_dist += (gen - gen_noattack).norm(float('-inf'))
                if F.mse_loss(gen, gen_noattack) > 0.05:
                    n_dist += 1
                n_samples += 1
            
            # # save origin image
            # out_file = config.global_settings.result_path + '/stargan_original.jpg'
            # vutils.save_image(imgs.cpu(), out_file, nrow=1, normalize=True, range=(-1., 1.)) 
            x_concat = torch.cat(x_total_list, dim=3)
            # out_file = os.path.join(config.global_settings.result_path + '/stargan_gen_{}.jpg'.format(idx))
            # out_file = os.path.join(config.global_settings.result_path + '_per/stargan_per_{}.jpg'.format(idx))
            out_file = os.path.join(self.config.global_settings.demo_result + '/stargan_per_{}.jpg'.format(idx))
            vutils.save_image(x_concat, out_file, nrow=1, normalize=True, range=(-1., 1.))
            
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
        print('stargan {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))

    def detect_image(self, img, ):
        pass

    def attack_face_swap(self):
        pgd_attack = attacks.LinfPGDAttack(model=self.simswap.face_id_net, device="cuda" , feat=None)
        

        
        id_image_path = self.config.global_settings.demo_image_id
        att_image_path = self.config.global_settings.demo_image_att

        id_image = self.img_read(str(id_image_path))
        att_image = self.img_read(str(att_image_path))

        #get face from id_iamge and att_image
        #the return images type is list [face ndarry, ....] 
        id_images_align, id_transforms, _ = self.simswap.run_detect_align(
            image=id_image, for_id=False, crop_size=256)
        att_image, att_transforms, _ = self.simswap.run_detect_align(
            image=att_image, for_id=False, crop_size=256)
        
        if id_image is None or att_image is None:
            print("Don't detect face")
            return
        for img in id_images_align:
            self.img_save_np(img, self.config.global_settings.demo_result + "/origin_id_{}.jpg".format(id_images_align.index(img)))
        for img in att_image:
            self.img_save_np(img, self.config.global_settings.demo_result + "/origin_att_{}.jpg".format(att_image.index(img)))
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
        
        
        id_latent_with_no_perb: torch.Tensor = self.simswap.face_id_net(id_images, need_trans = False)
        swap_image_with_no_perb: torch.Tensor = self.simswap.simswap_net(
            att_image, id_latent_with_no_perb, need_trans = False)

        img_no_perb_path = self.config.global_settings.demo_result + "/swapped_img_no_perb.jpg"   
        # image_no_perb = transforms.ToPILImage()(swap_image_with_no_perb[0].cpu().detach())
        # image_no_perb.save(img_no_perb_path)
        self.img_save(swap_image_with_no_perb, img_no_perb_path)

        image_with_perb, perb = pgd_attack.perturb_simswap(
            id_images, att_image, swap_image_with_no_perb, 
            self.simswap.face_id_net, self.simswap.simswap_net
        )
        # image_with_perb = id_images.cuda() + get_CUMA_perturbation(self.config).cuda()
        origin_perb_path = self.config.global_settings.demo_result + "/origin_perb.jpg"
        self.img_save(image_with_perb, origin_perb_path)


        id_latent_with_perb = self.simswap.face_id_net(image_with_perb, need_trans = False)
        swap_image_with_perb = self.simswap.simswap_net(att_image, id_latent_with_perb, need_trans = False)

        img_per_path = self.config.global_settings.demo_result + "/swapped_img_perb.jpg"
        # img_save(swap_image_with_no_perb, img_no_perb_path)
        self.img_save(swap_image_with_perb, img_per_path)
        # image_perb = transforms.ToPILImage()(swap_image_with_perb[0].cpu().detach())
        # image_perb.save(img_per_path)

        self.restore_id_image(id_image, id_images_align, id_transforms, image_with_perb)

    def restore_id_image(self, id_image, id_images_align, id_transforms, image_with_perb):
        align_id_img_batch_for_parsing_model: torch.Tensor = torch.stack(
            [self.to_tensor_norm()(x) for x in id_images_align], dim=0
        )
        align_id_img_batch_for_parsing_model = (
            align_id_img_batch_for_parsing_model.to(self.device)
        )

        id_transforms: torch.Tensor = torch.stack(
            [torch.tensor(x) for x in id_transforms], dim=0
        )
        id_transforms = id_transforms.to(self.device)

        align_id_img_batch: torch.Tensor = torch.stack(
            [self.to_tensor()(x) for x in id_images_align], dim=0
        )
        align_id_img_batch = align_id_img_batch.to(self.device)

        img_white = torch.zeros_like(align_id_img_batch) + 255

        inv_id_transforms: torch.Tensor = inverse_transform_batch(id_transforms)

        # Get face masks for the id image
        face_mask, ignore_mask_ids = self.simswap.bise_net.get_mask(
            align_id_img_batch_for_parsing_model, self.simswap.crop_size
        )

        soft_face_mask, _ = self.simswap.smooth_mask(face_mask)

        # Only take face area from the swapped image
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        image_with_perb = std * image_with_perb.detach().cpu() + mean

        image_with_perb = image_with_perb.cuda() * soft_face_mask + align_id_img_batch * (
            1 - soft_face_mask
        )
        image_with_perb[ignore_mask_ids, ...] = align_id_img_batch[ignore_mask_ids, ...]

        frame_size = (id_image.shape[0], id_image.shape[1])

        # Place swapped faces and masks where they should be in the original frame
        target_image = kornia.geometry.transform.warp_affine(
            image_with_perb.double(),
            inv_id_transforms,
            frame_size,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
            fill_value=torch.zeros(3),
        )
        self.img_save(target_image, self.config.global_settings.demo_result + "targe_image.jpg")
        # if torch.sum(ignore_mask_ids.int()) > 0:
        #     img_white = img_white.double()[ignore_mask_ids, ...]
        #     inv_id_transforms = inv_id_transforms[ignore_mask_ids, ...]

        img_mask = kornia.geometry.transform.warp_affine(
            img_white.double(),
            inv_id_transforms,
            frame_size,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
            fill_value=torch.zeros(3),
        )

        img_mask[img_mask > 20] = 255

        # numpy postprocessing
        # Collect masks for all crops
        img_mask = torch.sum(img_mask, dim=0, keepdim=True)

        # Get np.ndarray with range [0...255]
        img_mask = tensor2img(img_mask / 255.0)

        if self.simswap.use_erosion:
            kernel = np.ones(
                (self.simswap.erode_mask_value, self.simswap.erode_mask_value), dtype=np.uint8
            )
            img_mask = cv2.erode(img_mask, kernel, iterations=1)

        if self.simswap.use_blur:
            img_mask = cv2.GaussianBlur(
                img_mask, (self.simswap.smooth_mask_value, self.simswap.smooth_mask_value), 0
            )

        # Collect all swapped crops
        target_image = torch.sum(target_image, dim=0, keepdim=True)
        target_image = tensor2img(target_image)

        img_mask = np.clip(img_mask / 255, 0.0, 1.0)

        result = (img_mask * target_image + (1 - img_mask) * id_image).astype(np.uint8)
        img_path = self.config.global_settings.demo_result + "/restore_perb_id_image.jpg"
        cv2.imwrite(str(img_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        return result
        
    def trian(self):
        data_loader, stargan, pert_gen_net, simswap= model_data_maneger.prepare(self.config)
        # model_data_maneger.getconfig("config.yaml")

        #att_gan
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0
        for idx, (imgs, att_a, c_org) in enumerate(data_loader):
            imgs = imgs.cuda() if self.config.global_settings.gpu else imgs
            c_org = c_org.cuda() if self.config.global_settings.gpu else att_a
            c_org = c_org.type(torch.float)

            #get perturbation and gen image
            perturbs = self.get_CUMA_perturbation()
            x_noattack_list, x_attack_list = stargan.test_universal_model_level(imgs, c_org, perturbs, self.config.stargan)

            
            for j in range(len(x_attack_list)):
                gen_noattack = x_noattack_list[j]
                gen = x_attack_list[j]

                l1_error += F.l1_loss(gen, gen_noattack)
                l2_error += F.mse_loss(gen, gen_noattack)
                l0_error += (gen - gen_noattack).norm(0)
                min_dist += (gen - gen_noattack).norm(float('-inf'))
                if F.mse_loss(gen, gen_noattack) > 0.05:
                    n_dist += 1
                n_samples += 1
            
            # # save origin image
            # out_file = config.global_settings.result_path + '/stargan_original.jpg'
            # vutils.save_image(imgs.cpu(), out_file, nrow=1, normalize=True, range=(-1., 1.)) 

            
            #gen deepfake image and adversarial example
            # for j in range(len(x_attack_list)):
            #     #save deepfake image
            #     gen_noattack = x_noattack_list[j]
            #     out_file = config.global_settings.result_path + '/stargan_gen_{}.jpg'.format(5* idx + j)
            #     vutils.save_image(gen_noattack, out_file, nrow=1, normalize=True, range=(-1., 1.))

            #     # save deepfake adversarial example
            #     gen = x_fake_list[j]
            #     out_file = config.global_settings.result_path + '/stargan_advgen_{}.jpg'.format(5 * idx + j)
            #     vutils.save_image(gen, out_file, nrow=1, normalize=True, range=(-1., 1.))

            # if idx == 50:
            #     break
        print('stargan {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))



def test(self):
    pass

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config):
    #vaild(config)
    app  = PerbApplication(config)
    app.attack_face_swap()

if __name__ == "__main__" :
    main()