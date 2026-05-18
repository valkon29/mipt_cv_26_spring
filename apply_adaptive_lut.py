import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenFeatureExtractor(nn.Module):
    def __init__(self, feature_dim: int = 256):
        super(FrozenFeatureExtractor, self).__init__()

        def conv_gn_relu(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
                nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
                nn.ReLU(inplace=True),
            )

        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
                self.gn1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
                self.relu1 = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
                self.gn2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
                self.relu2 = nn.ReLU(inplace=True)

            def forward(self, x):
                identity = x
                out = self.conv1(x)
                out = self.gn1(out)
                out = self.relu1(out)
                out = self.conv2(out)
                out = self.gn2(out)
                out = out + identity
                out = self.relu2(out)
                return out

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                with torch.no_grad():
                    m.weight.mul_(0.5)

        self.features = nn.Sequential(
            conv_gn_relu(3, 32, kernel_size=7, stride=2, padding=3),
            conv_gn_relu(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64),
            conv_gn_relu(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128),
            conv_gn_relu(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256),
            nn.AdaptiveAvgPool2d(1),
        )

        self.features.apply(init_weights)

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        x_normalized = x * 2.0 - 1.0
        with torch.no_grad():
            features = self.features(x_normalized)
        features = features.view(features.size(0), -1)
        features = F.normalize(features, p=2, dim=1)
        features = features * 5.0
        return features


class ParameterGenerator(nn.Module):
    def __init__(self, feature_dim: int = 256, n_params: int = 32):
        super(ParameterGenerator, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, n_params)

        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.01)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.ln1(self.fc1(features))
        x = F.relu(x)
        x = self.ln2(self.fc2(x))
        x = F.relu(x)
        x = self.fc3(x)
        return torch.tanh(x) * 0.5


class LUTGenerator(nn.Module):
    def __init__(self, lut_size: int = 33, n_params: int = 32):
        super(LUTGenerator, self).__init__()
        self.lut_size = lut_size
        self.n_params = n_params
        coords = torch.linspace(0, 1, self.lut_size)
        r, g, b = torch.meshgrid(coords, coords, coords, indexing='ij')
        base_lut = torch.stack([r, g, b], dim=-1)
        self.register_buffer('base_lut', base_lut)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        batch_size = params.size(0)
        lut = self.base_lut.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

        brightness = params[:, :3].view(batch_size, 1, 1, 1, 3) * 0.2
        contrast = 1.0 + params[:, 3:6].view(batch_size, 1, 1, 1, 3) * 0.3
        contrast = torch.clamp(contrast, 0.5, 2.0)
        gamma = 1.0 + params[:, 6:9].view(batch_size, 1, 1, 1, 3) * 0.3
        gamma = torch.clamp(gamma, 0.5, 2.0)
        color_balance = params[:, 9:12].view(batch_size, 1, 1, 1, 3) * 0.1
        saturation = 1.0 + params[:, 12:13].view(batch_size, 1, 1, 1, 1) * 0.3
        saturation = torch.clamp(saturation, 0.5, 1.5)

        lut = lut + color_balance
        lut = lut + brightness
        lut = (lut - 0.5) * contrast + 0.5
        lum = 0.299 * lut[..., 0:1] + 0.587 * lut[..., 1:2] + 0.114 * lut[..., 2:3]
        lut = lum + (lut - lum) * saturation
        lut = torch.clamp(lut, 1e-3, 1.0)
        gamma_inv = 1.0 / gamma
        gamma_inv = torch.clamp(gamma_inv, 0.5, 2.0)
        lut = torch.pow(lut, gamma_inv)
        lut = torch.clamp(lut, 0, 1)
        return lut


class LUTApplier(nn.Module):
    def __init__(self, lut_size: int = 33):
        super(LUTApplier, self).__init__()
        self.lut_size = lut_size

    def forward(self, images: torch.Tensor, luts: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = images.shape
        scaled = images * (self.lut_size - 1)
        r, g, b = scaled[:, 0], scaled[:, 1], scaled[:, 2]

        r0 = torch.floor(r).long().clamp(0, self.lut_size - 1)
        r1 = torch.ceil(r).long().clamp(0, self.lut_size - 1)
        g0 = torch.floor(g).long().clamp(0, self.lut_size - 1)
        g1 = torch.ceil(g).long().clamp(0, self.lut_size - 1)
        b0 = torch.floor(b).long().clamp(0, self.lut_size - 1)
        b1 = torch.ceil(b).long().clamp(0, self.lut_size - 1)

        fr = (r - r0.floor()).unsqueeze(1)
        fg = (g - g0.floor()).unsqueeze(1)
        fb = (b - b0.floor()).unsqueeze(1)

        device = images.device

        def gather_lut(r_idx, g_idx, b_idx):
            batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1, 1, 1)
            batch_indices = batch_indices.expand(-1, height, width, 1).contiguous()
            r_idx = r_idx.view(batch_size, height, width, 1)
            g_idx = g_idx.view(batch_size, height, width, 1)
            b_idx = b_idx.view(batch_size, height, width, 1)
            indices = torch.cat([batch_indices, r_idx, g_idx, b_idx], dim=-1)
            lut_values = luts[indices[..., 0], indices[..., 1], indices[..., 2], indices[..., 3]]
            return lut_values.permute(0, 3, 1, 2)

        c000 = gather_lut(r0, g0, b0)
        c001 = gather_lut(r0, g0, b1)
        c010 = gather_lut(r0, g1, b0)
        c011 = gather_lut(r0, g1, b1)
        c100 = gather_lut(r1, g0, b0)
        c101 = gather_lut(r1, g0, b1)
        c110 = gather_lut(r1, g1, b0)
        c111 = gather_lut(r1, g1, b1)

        c00 = c000 * (1 - fr) + c100 * fr
        c01 = c001 * (1 - fr) + c101 * fr
        c10 = c010 * (1 - fr) + c110 * fr
        c11 = c011 * (1 - fr) + c111 * fr

        c0 = c00 * (1 - fg) + c10 * fg
        c1 = c01 * (1 - fg) + c11 * fg

        result = c0 * (1 - fb) + c1 * fb
        return result


class AdaptiveLUT(nn.Module):
    def __init__(self, lut_size: int = 33, n_params: int = 32):
        super(AdaptiveLUT, self).__init__()
        self.feature_extractor = FrozenFeatureExtractor()
        self.parameter_generator = ParameterGenerator(feature_dim=256, n_params=n_params)
        self.lut_generator = LUTGenerator(lut_size=lut_size, n_params=n_params)
        self.lut_applier = LUTApplier(lut_size=lut_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        params = self.parameter_generator(features)
        luts = self.lut_generator(params)
        enhanced = self.lut_applier(images, luts)
        return enhanced


def load_model(checkpoint_path: str, device: torch.device) -> AdaptiveLUT:
    model = AdaptiveLUT(lut_size=33, n_params=32).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def apply_model(model: AdaptiveLUT, img: np.ndarray, device: torch.device) -> np.ndarray:
    h, w = img.shape[:2]
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        enhanced = model(tensor)
    result = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result = np.clip(result, 0, 1)
    if result.shape[:2] != (h, w):
        result = cv2.resize(result, (w, h))
    return result


def main():
    parser = argparse.ArgumentParser(description='Apply Adaptive LUT model to an image')
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('output', type=str, help='Path to save enhanced image')
    parser.add_argument('--checkpoint', type=str,
                        default='adaptive_lut_results/best_adaptive_lut.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Image size for model input (default: 256, used with --resize)')
    parser.add_argument('--resize', action='store_true',
                        help='Resize image to --img-size before processing')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found")
        return 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully")

    print(f"Loading image: {args.input}")
    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        print(f"Error: Could not read image '{args.input}'")
        return 1
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    print(f"Applying enhancement...")
    if args.resize:
        print(f"  Resizing to {args.img_size}x{args.img_size}")
        img_input = cv2.resize(img_rgb, (args.img_size, args.img_size))
    else:
        img_input = img_rgb
    result = apply_model(model, img_input, device)

    result_bgr = (result * 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_bgr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, result_bgr)
    print(f"Enhanced image saved: {args.output}")


if __name__ == '__main__':
    exit(main())
