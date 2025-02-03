# losses.py

import torch


class MaskedMSELoss(torch.nn.Module):
    """
    Creates a criterion that measures the mean squared error
    (squared L2 norm) between each element in the input x and target y
    either considering the masked footprint or the full image.
    """

    def __init__(self, mask: bool, nodata: int):
        super(MaskedMSELoss, self).__init__()
        self.mask = mask
        self.nodata = nodata

    def forward(self, output, target):
        if self.mask:
            mask = target != self.nodata
            output = output[:, 0, :, :].unsqueeze(1)
            output = output[mask]
            target = target[mask]

        loss = torch.mean((output - target) ** 2)
        return loss


class MaskedRMSELoss(torch.nn.Module):
    """
    Creates a criterion that measures the root mean squared error
    either considering the masked footprint or the full image.
    """

    def __init__(self, mask: bool, nodata: int):
        super(MaskedRMSELoss, self).__init__()
        self.mask = mask
        self.nodata = nodata

    def forward(self, output, target):
        if self.mask:
            mask = target != self.nodata
            output = output[:, 0, :, :].unsqueeze(1)
            output = output[mask]
            target = target[mask]

        loss = torch.sqrt(torch.mean((output - target) ** 2))
        return loss


class MaskedGaussianNLLLoss(torch.nn.Module):
    """
    Gaussian negative log likelihood loss either considering the masked footprint or the full image.
    https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
    """

    def __init__(self, mask: bool, nodata: int, neg_penalty: float = 0):
        super(MaskedGaussianNLLLoss, self).__init__()
        self.mask = mask
        self.nodata = nodata
        self.neg_penalty = neg_penalty
        self.loss = torch.nn.GaussianNLLLoss()

    def forward(self, output, target):
        if output.shape[1] != 2:
            raise ValueError(
                "Output must have two channels: mean and variance. Set 'output_variance' to 'true' in the config file."
            )

        mean = output[:, 0, :, :].unsqueeze(1)
        var = output[:, 1, :, :].unsqueeze(1)
        if self.mask:
            mask = target != self.nodata
            mean = mean[mask]
            var = var[mask]
            target = target[mask]

        loss = self.loss(mean, target, var)

        if self.neg_penalty:
            penalty = torch.mean(torch.relu(-mean) ** 2)
            return loss + penalty * self.neg_penalty
        else:
            return loss
