import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn


try:
    import defenses.smoothing as smoothing
except:
    import stargan.defenses.smoothing as smoothing

class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.06, k=10, a=0.01, feat = None):
        """
        FGSM, I-FGSM and PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

        # Feature-level attack? Which layer?
        self.feat = feat

        # PGD or I-FGSM?
        self.rand = True

        # Universal perturbation
        self.up = None
        
    # def norm2norm(ori_mean = [], ori_std = [], res_mean = [], res)
    def perturb_simswap(self, id_image, att_image, swap_image_with_no_perb, arcface_net, simswap_net):
        #attaclk arcface
        #id_image and att_image is align
        if self.rand:
            X = id_image.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, id_image[0].shape).astype('float32'))
        else:
            X = id_image.clone().detach_()

        for i in range(self.k):
            X.requires_grad = True
            output = arcface_net(X, need_trans = False)
            output = simswap_net(att_image, output, need_trans = False)

            arcface_net.zero_grad()
            simswap_net.zero_grad()

            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(output, swap_image_with_no_perb)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - id_image, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(id_image + eta, min=-1, max=1).detach_()
            swap_image_with_no_perb = swap_image_with_no_perb.detach_()

        self.model.zero_grad()

        return X, X - id_image


def clip_tensor(X, Y, Z):
    # Clip X with Y min and Z max
    X_np = X.data.cpu().numpy()
    Y_np = Y.data.cpu().numpy()
    Z_np = Z.data.cpu().numpy()
    X_clipped = np.clip(X_np, Y_np, Z_np)
    X_res = torch.FloatTensor(X_clipped)
    return X_res

def perturb_batch(X, y, c_trg, model, adversary):
    # Perturb batch function for adversarial training
    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()
    
    adversary.model = model_cp

    X_adv, _ = adversary.perturb(X, y, c_trg)

    return X_adv