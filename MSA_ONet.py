import torch
from torch import nn
from torch.nn import init

class SEAttention(nn.Module):
    def __init__(self,  in_ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEAttentionInception(nn.Module):
    def __init__(self, in_ch, out_ch, di=4):
        super(SEAttentionInception, self).__init__()
        self.se=SEAttention(in_ch)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//di, kernel_size=1),
            nn.Conv2d(in_ch//di, in_ch//di, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch//di),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//di, kernel_size=1),
            nn.Conv2d(in_ch//di, in_ch//di, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(in_ch//di),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//di, kernel_size=1),
            nn.Conv2d(in_ch//di, in_ch//di, kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(in_ch//di),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Conv2d(in_ch, in_ch // di, kernel_size=1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        cat = torch.cat((branch1, branch2, branch3, branch4), dim=1)
        out = self.conv(cat)
        out1= out
        seout = self.se(out1)

        return seout


class Unet_Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_Down, self).__init__()

        self.SEI = SEAttentionInception(in_ch, out_ch)

        self.Down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_i = self.SEI(x)
        out = self.Down(x_i)
        return out


class Unet_Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_Up, self).__init__()
        self.Up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.Up(x)

class Simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.act(y)

class AISL(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AISL, self).__init__()
        self.SimAM=Simam_module(out_ch)
        self.convc = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x, f):
        x_an = self.SimAM(x)

        cat = torch.cat([x_an, f], 1)
        out = self.convc(cat)

        return out


class NestedUNet(nn.Module):
    def __init__(self, in_ch):#
        super().__init__()
        self.conv0_0 = nn.Conv2d(in_ch, 128, kernel_size=1)
        self.down0_1 = Unet_Down(128, 256)
        self.down1_2 = Unet_Down(256, 512)
        self.down2_3 = Unet_Down(512, 1024)

        self.up3_0 = Unet_Up(1024, 512)
        self.up2_1 = Unet_Up(512, 256)
        self.up1_2 = Unet_Up(256, 128)

        self.conv2_1 = AISL(512 * 2, 512)
        self.conv1_2 = AISL(256 * 2, 256)
        self.conv0_3 = AISL(128 * 2, 128)


    def forward(self, input):
        x0_0 = self.conv0_0(input)

        x1_0 = self.down0_1(x0_0)
        x2_0 = self.down1_2(x1_0)
        x3_0 = self.down2_3(x2_0)

        x2_1 = self.conv2_1(x2_0, self.up3_0(x3_0))
        x1_2 = self.conv1_2(x1_0, self.up2_1(x2_1))
        x0_3 = self.conv0_3(x0_0,self.up1_2(x1_2))

        f_out5 = x2_1 + x2_0
        f_out7 = x1_2 + x1_0
        f_out9 = x0_3 + x0_0

        return f_out5, f_out7, f_out9

def convD(in_ch, out_ch, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

def convT(in_ch, out_ch, kernel_size):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv3_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv3_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)


        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn_3 = self.conv3_1(attn)
        attn_3 = self.conv3_2(attn_3)

        attn = attn + attn_0 + attn_1 + attn_2 + attn_3

        attn = self.conv3(attn)

        return attn * u

class MSA_CD(nn.Module):
    def __init__(self):
        super().__init__()

        self.down39 = nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.msa_cd=AttentionModule(256)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, d9, d7, d5):
        d7f9 = self.down39(d9)

        dc = d7f9 + d5 + d7

        atten_img = self.msa_cd(dc)

        return atten_img

class MSA_ONet(nn.Module):
    def __init__(self,in_ch):
        super(MSA_ONet, self).__init__()
        self.fe = NestedUNet(in_ch)

        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.down1 = convD(256, 256, 3)

        self.convfuse = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.msa_cd = MSA_CD()
        self.down2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3,stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 256, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2, bias=True),
        )
        self.softmax = nn.Softmax(dim=-1)

    def count_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params
    def forward(self, img_before, img_after):
        f1_out5, f1_out7, f1_out9 = self.fe(img_before)
        f2_out5, f2_out7, f2_out9 = self.fe(img_after)

        d_map7 = self.conv1(f2_out7 - f1_out7)
        d_map9 = self.conv2(f2_out9 - f1_out9)

        D9= self.down1(d_map9)

        input71 = d_map9
        input72 = self.convfuse(torch.cat((D9, d_map7), dim=1))
        input73 = d_map7

        atten_map = self.msa_cd(input71, input72, input73)
        fe_out = atten_map.clone()
        out = self.down2(fe_out)

        out1 = torch.flatten(out, 1, 3)
        out2 = self.fc(out1)
        final_out = self.softmax(out2)
        return final_out