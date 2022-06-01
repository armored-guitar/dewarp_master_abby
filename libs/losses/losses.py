import torch
import torch.nn.functional as F
from torch import nn

from typing import Tuple


class BaselineLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.register_buffer("kernel_5", torch.ones(2, 1, 5, 5))
        self.register_buffer("kernel_1_5", torch.ones(1, 1, 5, 5))
        self.register_buffer("kernel", torch.ones(2, 1, 3, 3))
        self.register_buffer("kernel_1_3", torch.ones(1, 1, 3, 3))
        self.l1_weight = opt.get("l1_weight", 0.1)
        self.local_weight = opt.get("local_weight", 0.1)
        self.cs_weight = opt.get("loss_cs", 0.5)
        self.regress_weight = opt.get("loss_regress", 0.1)
        self.cls_weight = opt.get("loss_cls", 1)

    def _cls_loss(self, mask_pred, mask):
        return F.binary_cross_entropy_with_logits(mask_pred, mask)

    def _flow_loss(self, input_, target, outputs_classify, labels_classify):
        eps = 1e-10
        input_, target, outputs_classify, labels_classify = input_.float(), target.float(), outputs_classify.float(), labels_classify.float()
        input = input_.clone()
        n, c, h, w = input_.size()

        n_fore = torch.sum(labels_classify)
        labels_classify_ = labels_classify.unsqueeze(1)
        l_c = labels_classify_.expand(n, c, h, w).to(input_.dtype)
        input = input * l_c
        i_t = target - input
        loss_l1 = torch.sum(torch.abs(i_t)) / (n_fore * 2)

        l_c_outputs_classify = torch.nn.functional.sigmoid(outputs_classify).data.round().unsqueeze(1).expand(n, c, h,
                                                                                                              w)
        input = input_ * l_c_outputs_classify
        loss_CS = 1 + torch.sum(-((torch.sum(target * input, dim=1)) / (
                    torch.norm(target, dim=1) * torch.norm(input, dim=1) + eps))) / n_fore


        mask = F.conv2d(labels_classify_, self.kernel_1_3, padding=1)
        mask[mask < 9] = 0
        mask_9 = mask / 9
        loss_local = torch.sum(torch.abs(F.conv2d(i_t, self.kernel, padding=1, groups=2) * mask_9 - i_t * mask)) / (
                    torch.sum(mask_9) * 2)

        return loss_l1, loss_local, loss_CS, input, l_c_outputs_classify

    def calculate_loss(self, input_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], output_batch: Tuple[torch.Tensor, torch.Tensor]):
        _, labels_classify, labels, *_ = input_batch
        outputs, outputs_classify = output_batch #bs, 2, h, w
        outputs_classify = outputs_classify.squeeze(1)

        # l_c_outputs_classify = torch.nn.functional.sigmoid(outputs_classify).data
        loss_l1, loss_local, loss_CS, input, l_c_outputs_classify = self._flow_loss(outputs, labels, outputs_classify, labels_classify)
        loss_cls = self._cls_loss(outputs_classify, labels_classify)

        loss_regress = self.l1_weight * loss_l1 + self.local_weight * loss_local + self.cs_weight * loss_CS

        loss = self.cls_weight * loss_cls + self.regress_weight * loss_regress
        loss_dict = {"total_loss": loss.item(),
                    "L1": self.l1_weight * loss_l1.item(),
                    "local": self.local_weight * loss_local.item(), "cs": self.cs_weight * loss_CS.item(),
                    "cls": self.cls_weight * loss_cls.item(), "regression": self.regress_weight * loss_regress.item()}
                    # "debug": torch.mean(torch.sum(torch.abs(labels * input), dim=1)).item(),
                    # "debug_input": torch.norm(input, dim=1).mean().item(),
                    # "debug_flow": torch.norm(labels, dim=1).mean().item(),
                    # "debug_pred_mask": l_c_outputs_classify.mean().item(),
                    # "debug_mask": labels_classify.mean().item()}
        return loss, loss_dict


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def calculate_loss(self, input_batch, output_batch):
        labels = input_batch[1].long()
        pred = output_batch
        loss = self.loss(pred, labels)
        accuracy = (pred.argmax(dim=1) == labels).float().mean()
        return loss, {"ce": loss.item(), "accuracy": accuracy.item()}


def get_loss(opt):
    name = opt["name"]
    if name == "baseline":
        return BaselineLoss(opt)
    elif name == "ce":
        return CELoss()
    raise NotImplementedError


