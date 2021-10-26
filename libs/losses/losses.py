import torch
import torch.nn.functional as F
from torch import nn

from typing import Tuple


class BaselineLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.kernel_5 = nn.parameter.Parameter(torch.ones(2, 1, 5, 5), requires_grad=False)
        self.kernel_5 = nn.parameter.Parameter(torch.ones(2, 1, 5, 5), requires_grad=False)
        self.kernel_1_5 = nn.parameter.Parameter(torch.ones(1, 1, 5, 5), requires_grad=False)
        self.kernel = nn.parameter.Parameter(torch.ones(2, 1, 3, 3), requires_grad=False)
        self.kernel_1_3 = nn.parameter.Parameter(torch.ones(1, 1, 3, 3), requires_grad=False)
        self.l1_weight = opt.get("l1_weight", 0.1)
        self.local_weight = opt.get("local_weight", 0.1)
        self.cs_weight = opt.get("loss_cs", 0.5)
        self.regress_weight = opt.get("loss_regress", 0.1)
        self.cls_weight = opt.get("loss_cls", 1)

    def calculate_loss(self, input_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], output_batch: Tuple[torch.Tensor, torch.Tensor]):
        _, labels_classify, labels_flow = input_batch
        output_flow, outputs_classify = output_batch
        outputs_classify = outputs_classify.squeeze(1)
        input = output_flow.clone()

        n, c, h, w = output_flow.size()

        n_fore = torch.sum(labels_classify)

        labels_classify_ = labels_classify.unsqueeze(1)
        l_c = labels_classify_.expand(n, c, h, w).float()
        input = input * l_c

        i_t = labels_flow - input
        loss_l1 = torch.sum(torch.abs(i_t)) / (n_fore * 2)

        l_c_outputs_classify = torch.nn.functional.sigmoid(outputs_classify).data.round().unsqueeze(1).expand(n, c, h,
                                                                                                              w)
        input = output_flow * l_c_outputs_classify
        loss_CS = 1 + torch.sum(-((torch.sum(labels_flow * input, dim=1)) / (
                    torch.norm(labels_flow, dim=1) * torch.norm(input, dim=1) + 1e-10))) / n_fore

        mask = F.conv2d(labels_classify_, self.kernel_1_3, padding=1)
        mask[mask < 9] = 0
        mask_9 = mask / 9
        loss_local = torch.sum(torch.abs(F.conv2d(i_t, self.kernel, padding=1, groups=2) * mask_9 - i_t * mask)) / (
                    torch.sum(mask_9) * 2)

        loss_cls = F.binary_cross_entropy_with_logits(input, labels_flow, size_average=True)

        loss = self.l1_weight * loss_l1 + self.local_weight * loss_local + self.cs_weight * loss_CS + self.cls_weight * loss_cls
        loss_dict = {"total_loss": loss.item(), "L1": self.l1_weight * loss_l1.item(), "local": self.local_weight * loss_local.item(), "cs": self.cs_weight * loss_CS.item(), self.cls_weight * "cls": loss_cls.item()}
        loss_dict["regression"] = self.regress_weight * (loss_dict["L1"] + loss_dict["local"] + loss_dict["cs"])
        return loss, loss_dict


def get_loss(opt):
    name = opt["name"]
    if name == "baseline":
        return BaselineLoss(opt)
    raise NotImplementedError


