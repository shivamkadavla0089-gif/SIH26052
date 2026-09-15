import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0)):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, xr, xi):
        out_r = self.conv_r(xr) - self.conv_i(xi)
        out_i = self.conv_r(xi) + self.conv_i(xr)
        return out_r, out_i


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0), output_padding=(1, 0)):
        super(ComplexConvTranspose2d, self).__init__()
        self.deconv_r = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding)
        self.deconv_i = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding)

    def forward(self, xr, xi):
        out_r = self.deconv_r(xr) - self.deconv_i(xi)
        out_i = self.deconv_r(xi) + self.deconv_i(xr)
        return out_r, out_i


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(ComplexBatchNorm2d, self).__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, xr, xi):
        return self.bn_r(xr), self.bn_i(xi)


class ComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(ComplexLSTM, self).__init__()
        self.lstm_r = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm_i = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, xr, xi):
        xr = xr.transpose(1, 2)
        xi = xi.transpose(1, 2)
        out_rr, _ = self.lstm_r(xr)
        out_ii, _ = self.lstm_i(xi)
        out_ri, _ = self.lstm_r(xi)
        out_ir, _ = self.lstm_i(xr)
        out_r = out_rr - out_ii
        out_i = out_ri + out_ir
        return out_r.transpose(1, 2), out_i.transpose(1, 2)


class DCCRN(nn.Module):
    def __init__(self, n_fft=512, hop_length=256):
        super(DCCRN, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.enc_conv1 = ComplexConv2d(1, 16, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0))
        self.enc_bn1 = ComplexBatchNorm2d(16)
        self.enc_conv2 = ComplexConv2d(16, 32, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0))
        self.enc_bn2 = ComplexBatchNorm2d(32)
        self.enc_conv3 = ComplexConv2d(32, 64, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0))
        self.enc_bn3 = ComplexBatchNorm2d(64)

        self.bottleneck_dim = 64 * 33
        self.lstm = ComplexLSTM(input_size=self.bottleneck_dim, hidden_size=self.bottleneck_dim, num_layers=2)

        self.dec_conv3 = ComplexConvTranspose2d(128, 32, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0), output_padding=(1, 0))
        self.dec_bn3 = ComplexBatchNorm2d(32)
        self.dec_conv2 = ComplexConvTranspose2d(64, 16, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0), output_padding=(0, 0))
        self.dec_bn2 = ComplexBatchNorm2d(16)
        self.dec_conv1 = ComplexConvTranspose2d(32, 1, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0), output_padding=(0, 0))

    def forward(self, real, imag):
        b, c, f, t = real.size()

        e1_r, e1_i = F.prelu(self.enc_bn1(*self.enc_conv1(real, imag))[0], torch.tensor(0.25).to(real.device)), \
                     F.prelu(self.enc_bn1(*self.enc_conv1(real, imag))[1], torch.tensor(0.25).to(real.device))
        e2_r, e2_i = F.prelu(self.enc_bn2(*self.enc_conv2(e1_r, e1_i))[0], torch.tensor(0.25).to(real.device)), \
                     F.prelu(self.enc_bn2(*self.enc_conv2(e1_r, e1_i))[1], torch.tensor(0.25).to(real.device))
        e3_r, e3_i = F.prelu(self.enc_bn3(*self.enc_conv3(e2_r, e2_i))[0], torch.tensor(0.25).to(real.device)), \
                     F.prelu(self.enc_bn3(*self.enc_conv3(e2_r, e2_i))[1], torch.tensor(0.25).to(real.device))

        b, c3, f3, t3 = e3_r.size()
        lstm_in_r = e3_r.reshape(b, c3 * f3, t3)
        lstm_in_i = e3_i.reshape(b, c3 * f3, t3)
        lstm_out_r, lstm_out_i = self.lstm(lstm_in_r, lstm_in_i)
        lstm_out_r = lstm_out_r.reshape(b, c3, f3, t3)
        lstm_out_i = lstm_out_i.reshape(b, c3, f3, t3)

        d3_in_r = torch.cat([lstm_out_r, e3_r], dim=1)
        d3_in_i = torch.cat([lstm_out_i, e3_i], dim=1)
        d3_r, d3_i = F.prelu(self.dec_bn3(*self.dec_conv3(d3_in_r, d3_in_i))[0], torch.tensor(0.25).to(real.device)), \
                     F.prelu(self.dec_bn3(*self.dec_conv3(d3_in_r, d3_in_i))[1], torch.tensor(0.25).to(real.device))

        d2_in_r = torch.cat([d3_r, e2_r], dim=1)
        d2_in_i = torch.cat([d3_i, e2_i], dim=1)
        d2_r, d2_i = F.prelu(self.dec_bn2(*self.dec_conv2(d2_in_r, d2_in_i))[0], torch.tensor(0.25).to(real.device)), \
                     F.prelu(self.dec_bn2(*self.dec_conv2(d2_in_r, d2_in_i))[1], torch.tensor(0.25).to(real.device))

        d1_in_r = torch.cat([d2_r, e1_r], dim=1)
        d1_in_i = torch.cat([d2_i, e1_i], dim=1)
        mask_r, mask_i = self.dec_conv1(d1_in_r, d1_in_i)

        if mask_r.size(-1) != t:
            mask_r = F.interpolate(mask_r, size=(f, t), mode='bilinear', align_corners=False)
            mask_i = F.interpolate(mask_i, size=(f, t), mode='bilinear', align_corners=False)

        mask_r = torch.tanh(mask_r)
        mask_i = torch.tanh(mask_i)
        clean_r = real * mask_r - imag * mask_i
        clean_i = real * mask_i + imag * mask_r
        return clean_r, clean_i