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

class PSMI(nn.Module):
    def __init__(self, att_image, 
        pert_gen_net, face_edit_net, face_swap_net, face_id_net, face_detector):
        super(PSMI, self).__init__()

        self.pert_gen_net = pert_gen_net
        self.face_edit_net = face_edit_net
        self.face_id_net = face_id_net
        self.face_swap_net = face_swap_net
        self.face_detector = face_detector

        
        self.att_image = att_image
    
        self.device = torch.device("cuda")

        # self.face_detection_net = 

    def print_pert_gen_net_train_info(self):
            for name, parms in self.pert_gen_net.named_parameters():	
                print('-->name:', name)
                print('-->para:', parms)
                print('-->grad_requirs:',parms.requires_grad)
                print('-->grad_value:',parms.grad)

    def run_detect_align(
        self, image: np.ndarray, for_id: bool = False, crop_size = None, face_alignment_type : str = "none"
    ) -> Tuple[
        Union[Iterable[np.ndarray], None], Union[Iterable[np.ndarray], None], np.ndarray
    ]:
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

        return align_imgs, transforms, detection.score

    def forward(self, inputs, c_trg_list):
        self.pert_gen_net = self.pert_gen_net.to(self.device)
        perturbs = self.pert_gen_net(inputs)
       
        id_latent_no_pert: torch.Tensor = self.face_id_net(inputs, need_trans=False)
        id_latent_with_pert: torch.Tensor = self.face_id_net(inputs + perturbs, need_trans=False)
        
        with torch.no_grad():
            x_noattack_list_stargan, x_attack_list_stargan = self.face_edit_attck(
                inputs, perturbs, c_trg_list)
            x_noattack_list_stargan = torch.stack(
                x_noattack_list_stargan, dim = 0
            )
            x_attack_list_stargan = torch.stack(
                x_attack_list_stargan, dim=0
            )

            # x_noattack_list_simswap, x_attack_list_simswap = self.face_swap_attack(inputs, perturbs)
       
        return x_noattack_list_stargan, x_attack_list_stargan, id_latent_no_pert, id_latent_with_pert, inputs, inputs + perturbs

    def face_swap_attack(self, id_images, pertub):
        if id_images is None or self.att_image is None:
            print("Don't detect face")
            return
        
        # with torch.no_grad():
        id_latent_no_pert: torch.Tensor = self.face_id_net(id_images, need_trans=False)
        
        swap_images_no_pert: torch.Tensor = self.face_swap_net(
            self.att_image, id_latent_no_pert, need_trans=False)
        
        id_images_pert = id_images + pertub

        id_latent_with_pert: torch.Tensor = self.face_id_net(id_images_pert, need_trans=False)
        swap_images_with_pert: torch.Tensor = self.face_swap_net(
            self.att_image, id_latent_with_pert, need_trans=False)

        return swap_images_no_pert, swap_images_with_pert
    

    def face_edit_attck(self, x_real, perturb, c_trg_list):
        x_adv = x_real + perturb
        x_attack_list = []
        x_noattack_list = []

        for idx, c_trg in enumerate(c_trg_list):
            gen_noattack, _ = self.face_edit_net(x_real, c_trg)
            gen, _ = self.face_edit_net(x_adv, c_trg)
            x_attack_list.append(gen)
            x_noattack_list.append(gen_noattack)

        return x_noattack_list, x_attack_list


    def get_pert(self, inputs):
        return self.pert_gen_net(inputs)


    def run_detect_align(
        self, image: np.ndarray, for_id: bool = False, crop_size = None, face_alignment_type : str = "none"
    ) -> Tuple[
        Union[Iterable[np.ndarray], None], Union[Iterable[np.ndarray], None], np.ndarray
    ]:
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

        return align_imgs, transforms, detection.score
    
    def restore_id_image(self, id_image, id_images_align, id_transforms, image_with_perb):
        align_id_img_batch_for_parsing_model: torch.Tensor = torch.stack(
            [to_tensor_norm()(x) for x in id_images_align], dim=0
        )
        align_id_img_batch_for_parsing_model = (
            align_id_img_batch_for_parsing_model.to(self.device)
        )

        id_transforms: torch.Tensor = torch.stack(
            [torch.tensor(x) for x in id_transforms], dim=0
        )
        id_transforms = id_transforms.to(self.device)

        align_id_img_batch: torch.Tensor = torch.stack(
            [to_tensor()(x) for x in id_images_align], dim=0
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
        # img_save(target_image, self.config.global_settings.demo_result + "targe_image.jpg")
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

        

