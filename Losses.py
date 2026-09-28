import torch
import torch.nn as nn
import random
#=================================================================================================#
#  Magnitude Loss
class AMPLoss(nn.Module):
    def __init__(self):
        super(AMPLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='backward')
        x_mag =  torch.abs(x)
        y = torch.fft.rfft2(y, norm='backward')
        y_mag = torch.abs(y)

        return self.criterion(x_mag,y_mag)

# phase Loss
class PhaLoss(nn.Module):
    def __init__(self):
        super(PhaLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='backward')
        x_mag = torch.angle(x)  # 提取相位角
        y = torch.fft.rfft2(y, norm='backward')
        y_mag = torch.angle(y)

        return self.criterion(x_mag, y_mag)

#  total fourier loss
class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)

#==============================================================================================#
def sample_with_j(k, n, j):
    if n >= k:
        raise ValueError("n must be less than k.")
    if j < 0 or j > k:
        raise ValueError("j must be in the range 0 to k.")

    # 创建包含0到k的数字的列表
    numbers = list(range(k))

    # 确保j在数字列表中
    if j not in numbers:
        raise ValueError("j must be in the range 0 to k.")

    # 从数字列表中选择j
    sample = [j]

    # 从剩余的数字中选择n-1个
    remaining = [num for num in numbers if num != j]
    sample.extend(random.sample(remaining, n - 1))

    return sample
#==============================================================================================#
# contrastive loss function for fourier domain using total fourier information respectively.
class FCR(nn.Module):
    def __init__(self, ablation=False):
        super(FCR, self).__init__()
        self.l1 = nn.L1Loss()
        self.multi_n_num = 2

    def forward(self, a, p, n):
        a_fft = torch.fft.fft2(a)
        p_fft = torch.fft.fft2(p)
        n_fft = torch.fft.fft2(n)

        contrastive_all = 0
        for i in range(a_fft.shape[0]):
            # 计算anchor和positive整体的傅里叶域差值
            d_ap = self.l1(a_fft[i], p_fft[i])

            for j in sample_with_j(a_fft.shape[0], self.multi_n_num, i):
                # 计算anchor 和 negative整体的傅里叶域差值
                d_an = self.l1(a_fft[i], n_fft[j])
                contrastive_all += (d_ap / (d_an + 1e-7))

        contrastive_all = contrastive_all / (self.multi_n_num * a_fft.shape[0])

        return contrastive_all

class MFCR(nn.Module):
    def __init__(self, ablation=False):
        super(MFCR, self).__init__()
        self.l1 = nn.L1Loss()
        self.multi_n_num = 2

    def forward(self, a, p, n):
        a_fft = torch.fft.fft2(a)
        p_fft = torch.fft.fft2(p)
        n_fft = torch.fft.fft2(n)

        # 分别提取幅度和相位
        a_fft_mag = torch.abs(a_fft)
        p_fft_mag = torch.abs(p_fft)
        n_fft_mag = torch.abs(n_fft)

        contrastive_mag = 0
        for i in range(a_fft.shape[0]):
            # 计算anchor和positive整体的傅里叶域差值
            d_ap = self.l1(a_fft[i], p_fft[i])

            # 分别计算anchor和positive 幅度的差值
            d_ap_mag = self.l1(a_fft_mag[i], p_fft_mag[i])

            for j in sample_with_j(a_fft.shape[0], self.multi_n_num, i):

                # 分别计算anchor 和 negative幅度傅里叶域差值
                d_an_mag = self.l1(a_fft_mag[i], n_fft_mag[j])
                contrastive_mag += (d_ap_mag / (d_an_mag + 1e-7))

        contrastive_mag = contrastive_mag / (self.multi_n_num * a_fft.shape[0])

        return  contrastive_mag
class PFCR(nn.Module):
    def __init__(self, ablation=False):
        super(PFCR, self).__init__()
        self.l1 = nn.L1Loss()
        self.multi_n_num = 2

    def forward(self, a, p, n):
        a_fft = torch.fft.fft2(a)
        p_fft = torch.fft.fft2(p)
        n_fft = torch.fft.fft2(n)

        # 分别提取幅度和相位
        a_fft_pha = torch.angle(a_fft)
        p_fft_pha = torch.angle(p_fft)
        n_fft_pha = torch.angle(n_fft)

        contrastive_pha = 0
        for i in range(a_fft.shape[0]):

            # 分别计算anchor和positive 相位的差值
            d_ap_pha = self.l1(a_fft_pha[i], p_fft_pha[i])

            for j in sample_with_j(a_fft.shape[0], self.multi_n_num, i):

                # 分别计算anchor 和 negative幅度和相位的傅里叶域差值
                d_an_pha = self.l1(a_fft_pha[i], n_fft_pha[j])
                contrastive_pha += (d_ap_pha / (d_an_pha + 1e-7))

        contrastive_pha = contrastive_pha / (self.multi_n_num * a_fft.shape[0])

        return  contrastive_pha
#=======================================================================================================================#
# contrastive loss function for space domain
class SCR(nn.Module):
    def __init__(self, ablation=False):
        super(SCR, self).__init__()
        self.l1 = nn.L1Loss()
        self.multi_n_num = 2

    def forward(self, a, p, n):
        a_spa = a
        p_spa = p
        n_spa = n

        contrastive_spa = 0
        for i in range(a_spa.shape[0]):
            # 计算anchor和positive 空域差值
            d_ap = self.l1(a_spa[i], p_spa[i])

            for j in sample_with_j(a_spa.shape[0], self.multi_n_num, i):
                # 计算anchor 和 negative整体的傅里叶域差值
                d_an = self.l1(a_spa[i], n_spa[j])
                contrastive_spa += (d_ap / (d_an + 1e-7))

        contrastive_spa = contrastive_spa / (self.multi_n_num * a_spa.shape[0])
        return contrastive_spa


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss



##############
class CharbonnierLoss2(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss2, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss
