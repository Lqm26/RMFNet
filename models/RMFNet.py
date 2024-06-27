from .submodules import *


class Multi_Branch(nn.Module):
    def __init__(self, n_c, n_b, scale, in_c=3*3, branch_c=3):
        super(Multi_Branch, self).__init__()
        pad = (1, 1)

        self.aux_feature = nn.Conv2d(branch_c, n_c, 3, 1, padding=pad)
        basic_block = functools.partial(ResidualBlock_noBN, nf=n_c)
        self.ResBlock1 = make_layer(basic_block, n_b)
        self.ResBlock2 = make_layer(basic_block, n_b)

        self.main_feature = nn.Conv2d(scale ** 2 * 3 + n_c + in_c, n_c, 3, 1, padding=pad)
        self.conv_h_m = nn.Conv2d(n_c, n_c, 3, 1, padding=pad)
        self.conv_o_m = nn.Conv2d(n_c * 2, scale ** 2 * 3, 3, 1, padding=pad)

        self.FFM_pos = Feature_fusion_module(in_channel=n_c * 2, out_channel=n_c)
        self.FFM_neg = Feature_fusion_module(in_channel=n_c * 2, out_channel=n_c)
        self.FEM = Feature_Exchange_Module(n_c)

        initialize_weights([self.main_feature, self.conv_h_m, self.conv_o_m,
                            self.aux_feature], 0.1)

    def forward(self, xs, hs, os):
        x_m, x_pos, x_neg = xs

        x_m = torch.cat((x_m, hs, os), dim=1)
        x_m = F.relu(self.main_feature(x_m))

        x_pos = F.relu(self.aux_feature(x_pos))
        x_pos = self.ResBlock1(self.FFM_pos(x_m, x_pos))

        x_neg = F.relu(self.aux_feature(x_neg))
        x_neg = self.ResBlock1(self.FFM_neg(x_m, x_neg))

        x_pos, x_neg = self.FEM(x_pos, x_neg, 1, 1)
        x_pos = self.ResBlock2(x_pos)
        x_neg = self.ResBlock2(x_neg)

        x_h_m = F.relu(self.conv_h_m(x_m))
        x_o_m = self.conv_o_m(torch.cat((x_neg, x_neg), dim=1))

        return x_h_m, x_o_m

class RMFNet(nn.Module):
    def __init__(self, scale, n_c, n_b, repeat=1):
        super(RMFNet, self).__init__()
        self.Multi_Branch = Multi_Branch(n_c, n_b, scale, in_c=3*3)
        self.scale = scale
        self.down = PixelUnShuffle(scale)
        self.repeat = repeat

    def forward(self, x, x_h, x_o, init):

        _, _, T, _, _ = x.shape
        f1 = x[:, :, 0, :, :]
        f2 = x[:, :, 1, :, :]
        f3 = x[:, :, 2, :, :]

        x_input_main = torch.cat((f1, f2, f3), dim=1)
        x_input_aux1 = torch.cat(
            (f1[:, 0:1, :, :].repeat(1, self.repeat, 1, 1), f2[:, 0:1, :, :].repeat(1, self.repeat, 1, 1),
             f3[:, 0:1, :, :].repeat(1, self.repeat, 1, 1)), dim=1)  # pos
        x_input_aux2 = torch.cat(
            (f1[:, 1:2, :, :].repeat(1, self.repeat, 1, 1), f2[:, 1:2, :, :].repeat(1, self.repeat, 1, 1),
             f3[:, 1:2, :, :].repeat(1, self.repeat, 1, 1)), dim=1)  # neg

        if init:
            x_h, x_o = self.Multi_Branch([x_input_main, x_input_aux1, x_input_aux2], x_h, x_o)
        else:
            x_o = self.down(x_o)
            x_h, x_o = self.Multi_Branch([x_input_main, x_input_aux1, x_input_aux2], x_h, x_o)
        x_o = F.pixel_shuffle(x_o, self.scale) + F.interpolate(f2, scale_factor=self.scale, mode='nearest')
        
        return x_h, x_o
