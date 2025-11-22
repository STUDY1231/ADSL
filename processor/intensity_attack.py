import torch
import torch.nn as nn
import random


class IntensityAttacker(nn.Module):
    def __init__(self, n_points=20, data_range=(-1, 1)):
        super().__init__()
        self.n = n_points
        self.rho = nn.Parameter(torch.zeros(n_points + 1))
        self.x_min = data_range[0]
        self.x_max = data_range[1]

    def intensity_mapping(self, x):
        exp_diff = torch.exp(self.rho - self.rho[0])
        cumsum = torch.cumsum(exp_diff, dim=0)
        total = cumsum[-1]
        mapping_points = (cumsum - 1) / (total - 1 + 1e-8)

        x_norm = (x - self.x_min) / (self.x_max - self.x_min + 1e-8)

        indices = x_norm * self.n
        indices_floor = indices.floor().long().clamp(0, self.n - 1)
        indices_ceil = (indices_floor + 1).clamp(0, self.n)
        weights = (indices - indices_floor.float()).clamp(0, 1)

        y_floor = mapping_points[indices_floor]
        y_ceil = mapping_points[indices_ceil]
        y_mapped = y_floor + weights * (y_ceil - y_floor)

        x_mapped = (self.x_max - self.x_min) * y_mapped + self.x_min
        return x_mapped

    def forward(self, x, mask=None):
        x_attacked = self.intensity_mapping(x)
        if mask is not None:
            x_out = mask * x_attacked + (1 - mask) * x
        else:
            x_out = x_attacked
        return x_out

    def reset_parameters(self):
        with torch.no_grad():
            self.rho.zero_()


def generate_part_aware_mask(batch_size, img_height=256, img_width=128,
                             patch_size=16, selected_part=None, device='cuda'):
    num_patches_h = img_height // patch_size  # 16

    part_row_ranges = {
        0: (0, num_patches_h // 2),
        1: (num_patches_h // 4, num_patches_h // 4 + num_patches_h // 2),
        2: (num_patches_h // 2, num_patches_h),
    }

    if selected_part is None:
        selected_part = random.randint(0, 2)
    start_row, end_row = part_row_ranges[selected_part]
    start_pixel = start_row * patch_size
    end_pixel = end_row * patch_size

    mask = torch.zeros(batch_size, 1, img_height, img_width, device=device)
    mask[:, :, start_pixel:end_pixel, :] = 1.0

    return mask, selected_part