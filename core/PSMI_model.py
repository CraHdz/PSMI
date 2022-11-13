from turtle import forward
import torch
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Tuple, Union
from FaceDetector.face_detector import Detection, FaceDetector
from FaceAlign.face_align import align_face, FaceAlignmentType, inverse_transform_batch
from until import *
import kornia
from PostProcess.utils import SoftErosion

class PSMI(nn.Module):
    def __init__(self, id_image, att_image, config,
        pert_gen_net, face_edit_net, face_swap_net, face_id_net, 
        face_detector=None, bise_net=None):
        super(PSMI, self).__init__()

        self.tf = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

        self.device = torch.device(config.device)

        #init net
        self.pert_gen_net = pert_gen_net
        self.face_edit_net = face_edit_net
        self.face_id_net = face_id_net
        self.face_swap_net = face_swap_net
        self.face_detector = face_detector
        self.bise_net = bise_net
        
        self.att_image = att_image
        self.id_image_face_id = self.get_face_id(id_image).to(self.device)

        self.bise_net = bise_net

        #init restore image para
        self.smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).to(
            self.device
        )
        self.erode_mask_value = config.erode_mask_value
        self.smooth_mask_value = config.smooth_mask_value
        self.crop_size = config.crop_size

        self.use_erosion = True
        if self.erode_mask_value == 0:
            self.use_erosion = False

        self.use_blur = True
        if self.smooth_mask_value == 0:
            self.use_erosion = False


        # self.set_model_grad()

    def get_face_id(self, img):
        imgs_align, img_transforms, _= self.run_detect_align(image=img, for_id=False, crop_size=256)

        #id_latent dim is (n * 512)
        id_latent= self.face_id_net(imgs_align, need_trans=False)
        return id_latent.clone().detach_()

    def set_model_grad(self):
        for param in self.face_edit_net.parameters():
            param.requires_grad = False
        for param in self.face_swap_net.parameters():
            param.requires_grad = False
        for param in self.face_id_net.parameters():
            param.requires_grad = False
        # if self.face_detector is not None:
        #     for param in self.face_detector.handler.parameters():
        #         param.requires_grad = False
        # if self.bise_net is not None:
        #     for param in self.bise_net.parameters():
        #         param.requires_grad = False

    def print_pert_gen_net_train_info(self):
            for name, parms in self.pert_gen_net.named_parameters():	
                print('-->name:', name)
                print('-->para:', parms)
                print('-->grad_requirs:',parms.requires_grad)
                print('-->grad_value:',parms.grad)

    def forward(self, inputs, c_trg_list):
        self.pert_gen_net = self.pert_gen_net.to(self.device)
        perturbs = self.pert_gen_net(inputs)
       
        id_latent_no_pert, id_latent_with_pert = self.face_id_attack(inputs, perturbs)
        
        x_face_edit_without_pert, x_face_edit_with_pert = self.face_edit_attck(
            inputs, perturbs, c_trg_list)

        x_noattack_list_simswap, x_attack_list_simswap = self.face_swap_attack(inputs, perturbs)
       
        return x_face_edit_without_pert, x_face_edit_with_pert, id_latent_no_pert, id_latent_with_pert, x_noattack_list_simswap, x_attack_list_simswap

    def face_id_attack(self, id_images, pertub):
        if id_images is None or self.att_image is None:
            print("Don't detect face")
            return
        
        with torch.no_grad():
            id_latent_no_pert= self.face_id_net(id_images, need_trans=False)
        
        id_images_pert = id_images + pertub
        id_latent_with_pert = self.face_id_net(id_images_pert, need_trans=False)
        return id_latent_no_pert, id_latent_with_pert

    def face_swap_attack(self, attimages, pertub):
        if attimages is None or self.id_image_face_id is None:
            print("Don't detect face")
            return
        
        #the input image range is (-1, 1)
        with torch.no_grad():
            swap_images_no_pert: torch.Tensor = self.face_swap_net(
                attimages, self.id_image_face_id, need_trans=False)
        
        attimg_with_pert = attimages + pertub

        swap_images_with_pert: torch.Tensor = self.face_swap_net(
            attimg_with_pert, self.id_image_face_id, need_trans=False)

        return swap_images_no_pert, swap_images_with_pert
    

    def face_edit_attck(self, x_real, perturb, c_trg_list):
        x_adv = x_real + perturb
        x_attack_list = []
        x_noattack_list = []

        for idx, c_trg in enumerate(c_trg_list):
            with torch.no_grad():
                gen_noattack, _ = self.face_edit_net(x_real, c_trg)
            gen, _ = self.face_edit_net(x_adv, c_trg)
            x_attack_list.append(gen)
            x_noattack_list.append(gen_noattack)

        x_noattack_edit = torch.stack(
            x_noattack_list, dim = 0
        )
        x_attack_edit = torch.stack(
            x_attack_list, dim=0
        )
        
        #the output shape is (select_att_num, batch_size, C, H, W)
        return x_noattack_edit, x_attack_edit

    def face_edit_attck_pert(self, x_real, x_adv, c_trg_list):
        x_attack_list = []
        x_noattack_list = []

        for idx, c_trg in enumerate(c_trg_list):
            with torch.no_grad():
                gen_noattack, _ = self.face_edit_net(x_real, c_trg)
            gen, _ = self.face_edit_net(x_adv, c_trg)
            x_attack_list.append(gen)
            x_noattack_list.append(gen_noattack)

        x_noattack_edit = torch.stack(
            x_noattack_list, dim = 0
        )
        x_attack_edit = torch.stack(
            x_attack_list, dim=0
        )
        
        #the output shape is (select_att_num, batch_size, C, H, W)
        return x_noattack_edit, x_attack_edit

    
    def get_pert(self, inputs):
        return self.pert_gen_net(inputs)


    def run_detect_align(
        self, image: np.ndarray, for_id: bool = False, crop_size = None, face_alignment_type : str = "none"
    ) -> Tuple[
        Union[Iterable[np.ndarray], None], Union[Iterable[np.ndarray], None], np.ndarray
    ]:
        image = tensor2img_denorm(image)

        detection: Detection = self.face_detector(image)

        if detection.bbox is None:
            if for_id:
                raise "Can't detect a face! Please change the ID image!"
            return None, None, detection.score

        kps = detection.key_points

        if for_id:
            max_score_ind = np.argmax(detection.score, axis=0)
            kps = detection.key_points[max_score_ind]
            kps = kps[None, ...]
        if crop_size is None:
            align_imgs, transforms = align_face(
                image,
                kps,
                crop_size=self.crop_size,
                mode="ffhq"
                if face_alignment_type == FaceAlignmentType.FFHQ
                else "none",
            )
        else:
            align_imgs, transforms = align_face(
                image,
                kps,
                crop_size=crop_size,
                mode="ffhq"
                if face_alignment_type == FaceAlignmentType.FFHQ
                else "none",
            )

        align_imgs = torch.stack(
            [self.tf(img) for img in align_imgs], dim=0
        )

        return align_imgs, transforms, detection.score
    
    def restore_id_image(self, id_image, id_images_align, id_transforms, image_with_perb):
        id_image = id_image.squeeze(0).permute(1,2,0)
        id_image = ((id_image * 0.5 + 0.5) * 255).numpy() 

        # align_id_img_batch_for_parsing_model: torch.Tensor = torch.stack(
        #     [to_tensor_norm()(x) for x in id_images_align], dim=0
        # )
        align_id_img_batch_for_parsing_model = id_images_align
        align_id_img_batch_for_parsing_model = (
            align_id_img_batch_for_parsing_model.to(self.device)
        )

        id_transforms: torch.Tensor = torch.stack(
            [torch.tensor(x) for x in id_transforms], dim=0
        )
        id_transforms = id_transforms.to(self.device)

        # align_id_img_batch: torch.Tensor = torch.stack(
        #     [to_tensor()(x) for x in id_images_align], dim=0
        # )
        align_id_img_batch = id_images_align * 0.5 + 0.5
        align_id_img_batch = align_id_img_batch.to(self.device)

        img_white = torch.zeros_like(align_id_img_batch) + 255

        inv_id_transforms: torch.Tensor = inverse_transform_batch(id_transforms)

        # Get face masks for the id image
        face_mask, ignore_mask_ids = self.bise_net.get_mask(
            align_id_img_batch_for_parsing_model, self.crop_size
        )

        soft_face_mask, _ = self.smooth_mask(face_mask)

        # Only take face area from the swapped image
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        image_with_perb = std * image_with_perb.detach().cpu() + mean

        image_with_perb = image_with_perb.to(self.device) * soft_face_mask + align_id_img_batch * (
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

        if self.use_erosion:
            kernel = np.ones(
                (self.erode_mask_value, self.erode_mask_value), dtype=np.uint8
            )
            img_mask = cv2.erode(img_mask, kernel, iterations=1)

        if self.use_blur:
            img_mask = cv2.GaussianBlur(
                img_mask, (self.smooth_mask_value, self.smooth_mask_value), 0
            )

        # Collect all swapped crops
        target_image = torch.sum(target_image, dim=0, keepdim=True)
        target_image = tensor2img(target_image)

        img_mask = np.clip(img_mask / 255, 0.0, 1.0)

        result = (img_mask * target_image + (1 - img_mask) * id_image).astype(np.uint8)
        img_path = "./demo_result/restore_perb_id_image.jpg"
        cv2.imwrite(str(img_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        result = self.tf(result).unsqueeze(0)
        return result

        

