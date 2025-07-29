import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SRGAN3D(nn.Module):
    def __init__(self, input_depth=32, input_height=32, input_width=32, input_channels=1, batch_size=8,
                 kernel_size=(3, 3, 3)):
        super(SRGAN3D, self).__init__()
        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.num_points = kernel_size[0] * kernel_size[1] * kernel_size[2]
        self.num_channels = input_channels
        self.extend_scope = 2.0

        # Initial convolution
        self.conv1 = nn.Conv3d(input_channels, 64, 3, padding=1)

        # First block
        self.conv2 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv6_1 = nn.Conv3d(64, 64, 3, padding=1)
        self.ca_at1 = nn.Sequential(
            nn.Conv3d(256, 256 // 16, 1),
            nn.Conv3d(256 // 16, 256, 1)
        )
        self.conv7 = nn.Conv3d(256, 64, 1)

        # Second block
        self.conv8 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv9 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv10 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv11 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv12 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv12_1 = nn.Conv3d(64, 64, 3, padding=1)
        self.ca_at2 = nn.Sequential(
            nn.Conv3d(256, 256 // 16, 1),
            nn.Conv3d(256 // 16, 256, 1)
        )
        self.conv13 = nn.Conv3d(256, 64, 1)

        # Third block
        self.conv14 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv15 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv16 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv17 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv18 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv18_1 = nn.Conv3d(64, 64, 3, padding=1)
        self.ca_at3 = nn.Sequential(
            nn.Conv3d(256, 256 // 16, 1),
            nn.Conv3d(256 // 16, 256, 1)
        )
        self.conv19 = nn.Conv3d(256, 64, 1)

        # Fourth block
        self.conv20 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv21 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv22 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv23 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv24 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv24_1 = nn.Conv3d(64, 64, 3, padding=1)
        self.ca_at4 = nn.Sequential(
            nn.Conv3d(256, 256 // 16, 1),
            nn.Conv3d(256 // 16, 256, 1)
        )
        self.conv25 = nn.Conv3d(256, 64, 1)

        # Fifth block
        self.conv26 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv27 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv28 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv29 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv30 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv30_1 = nn.Conv3d(64, 64, 3, padding=1)
        self.ca_at5 = nn.Sequential(
            nn.Conv3d(256, 256 // 16, 1),
            nn.Conv3d(256 // 16, 256, 1)
        )
        self.conv31 = nn.Conv3d(256, 64, 1)

        # Final layers
        self.ca_at6 = nn.Sequential(
            nn.Conv3d(384, 384 // 16, 1),
            nn.Conv3d(384 // 16, 384, 1)
        )
        self.conv32 = nn.Conv3d(384, 1, 3, padding=1)
        self.sa_conv32 = nn.Sequential(
            nn.Conv3d(1, 1, 3, stride=2, padding=1),
            nn.ConvTranspose3d(1, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Conv3d(1, 1, 3, padding=1)
        )

        # Deformable convolution layers
        self.deform_offset = nn.Conv3d(64, 3 * self.num_points, 3, padding=1)
        self.deform_bn = nn.BatchNorm3d(3 * self.num_points)
        self.deform_conv = nn.Conv3d(64, 64, 3, padding=1)
        self.deform_bn_out = nn.BatchNorm3d(64)

    def _coordinate_map_3d(self, offset_field):
        # offset_field shape: [batch, 3*num_points, depth, height, width]
        offset_field = offset_field.view(self.batch_size, 3, self.num_points, self.input_depth, self.input_height,
                                         self.input_width)
        z_offset, y_offset, x_offset = offset_field[:, 0], offset_field[:, 1], offset_field[:, 2]

        # Create center coordinates
        z_center = torch.arange(self.input_depth).repeat(self.input_height * self.input_width).view(self.input_depth,
                                                                                                    self.input_height,
                                                                                                    self.input_width)
        z_center = z_center.repeat(self.num_points, 1, 1, 1).permute(1, 2, 3, 0)
        z_center = z_center.unsqueeze(0).repeat(self.batch_size, 1, 1, 1, 1).float()

        y_center = torch.arange(self.input_height).repeat(self.input_width * self.input_depth).view(self.input_height,
                                                                                                    self.input_width,
                                                                                                    self.input_depth)
        y_center = y_center.repeat(self.num_points, 1, 1, 1).permute(2, 1, 3, 0)
        y_center = y_center.unsqueeze(0).repeat(self.batch_size, 1, 1, 1, 1).float()

        x_center = torch.arange(self.input_width).repeat(self.input_depth * self.input_height).view(self.input_width,
                                                                                                    self.input_depth,
                                                                                                    self.input_height)
        x_center = x_center.repeat(self.num_points, 1, 1, 1).permute(1, 3, 2, 0)
        x_center = x_center.unsqueeze(0).repeat(self.batch_size, 1, 1, 1, 1).float()

        # Create grid coordinates
        z = torch.linspace(-(self.kernel_size[0] - 1) / 2, (self.kernel_size[0] - 1) / 2, self.kernel_size[0])
        y = torch.linspace(-(self.kernel_size[1] - 1) / 2, (self.kernel_size[1] - 1) / 2, self.kernel_size[1])
        x = torch.linspace(-(self.kernel_size[2] - 1) / 2, (self.kernel_size[2] - 1) / 2, self.kernel_size[2])
        z, y, x = torch.meshgrid(z, y, x, indexing='ij')

        z_grid = z.flatten().repeat(self.input_depth * self.input_height * self.input_width).view(self.input_depth,
                                                                                                  self.input_height,
                                                                                                  self.input_width,
                                                                                                  self.num_points)
        y_grid = y.flatten().repeat(self.input_depth * self.input_height * self.input_width).view(self.input_depth,
                                                                                                  self.input_height,
                                                                                                  self.input_width,
                                                                                                  self.num_points)
        x_grid = x.flatten().repeat(self.input_depth * self.input_height * self.input_width).view(self.input_depth,
                                                                                                  self.input_height,
                                                                                                  self.input_width,
                                                                                                  self.num_points)

        z_grid = z_grid.unsqueeze(0).repeat(self.batch_size, 1, 1, 1, 1)
        y_grid = y_grid.unsqueeze(0).repeat(self.batch_size, 1, 1, 1, 1)
        x_grid = x_grid.unsqueeze(0).repeat(self.batch_size, 1, 1, 1, 1)

        # Calculate final coordinates
        z_coord = z_center + z_grid + self.extend_scope * z_offset
        y_coord = y_center + y_grid + self.extend_scope * y_offset
        x_coord = x_center + x_grid + self.extend_scope * x_offset

        return x_coord, y_coord, z_coord

    def _trilinear_interpolate(self, input_feature, x, y, z):
        x = x.view(-1)
        y = y.view(-1)
        z = z.view(-1)

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        z0 = torch.floor(z).long()
        z1 = z0 + 1

        x0 = torch.clamp(x0, 0, self.input_width - 1)
        x1 = torch.clamp(x1, 0, self.input_width - 1)
        y0 = torch.clamp(y0, 0, self.input_height - 1)
        y1 = torch.clamp(y1, 0, self.input_height - 1)
        z0 = torch.clamp(z0, 0, self.input_depth - 1)
        z1 = torch.clamp(z1, 0, self.input_depth - 1)

        input_feature_flat = input_feature.view(-1, self.num_channels)

        dimension_3 = self.input_width
        dimension_2 = self.input_width * self.input_height
        dimension_1 = self.input_width * self.input_height * self.input_depth

        base = torch.arange(self.batch_size) * dimension_1
        repeat = torch.ones(self.num_points * self.input_depth * self.input_height * self.input_width).unsqueeze(0).t()
        base = base.view(-1, 1).matmul(repeat.t()).view(-1)

        base_z0 = base + z0 * dimension_2
        base_z1 = base + z1 * dimension_2

        base_z0_y0 = base_z0 + y0 * dimension_3
        base_z0_y1 = base_z0 + y1 * dimension_3
        base_z1_y0 = base_z1 + y0 * dimension_3
        base_z1_y1 = base_z1 + y1 * dimension_3

        index_a = (base_z0_y0 + x0).long()
        index_b = (base_z0_y1 + x0).long()
        index_c = (base_z0_y0 + x1).long()
        index_d = (base_z0_y1 + x1).long()
        index_e = (base_z1_y0 + x0).long()
        index_f = (base_z1_y1 + x0).long()
        index_g = (base_z1_y0 + x1).long()
        index_h = (base_z1_y1 + x1).long()

        value_a = input_feature_flat[index_a]
        value_b = input_feature_flat[index_b]
        value_c = input_feature_flat[index_c]
        value_d = input_feature_flat[index_d]
        value_e = input_feature_flat[index_e]
        value_f = input_feature_flat[index_f]
        value_g = input_feature_flat[index_g]
        value_h = input_feature_flat[index_h]

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()
        z0_float = z0.float()
        z1_float = z1.float()

        wa = ((x1_float - x) * (y1_float - y) * (z1_float - z)).unsqueeze(1)
        wb = ((x1_float - x) * (y - y0_float) * (z1_float - z)).unsqueeze(1)
        wc = ((x - x0_float) * (y1_float - y) * (z1_float - z)).unsqueeze(1)
        wd = ((x - x0_float) * (y - y0_float) * (z1_float - z)).unsqueeze(1)
        we = ((x1_float - x) * (y1_float - y) * (z - z0_float)).unsqueeze(1)
        wf = ((x1_float - x) * (y - y0_float) * (z - z0_float)).unsqueeze(1)
        wg = ((x - x0_float) * (y1_float - y) * (z - z0_float)).unsqueeze(1)
        wh = ((x - x0_float) * (y - y0_float) * (z - z0_float)).unsqueeze(1)

        outputs = (value_a * wa + value_b * wb + value_c * wc + value_d * wd +
                   value_e * we + value_f * wf + value_g * wg + value_h * wh)

        outputs = outputs.view(self.batch_size,
                               self.kernel_size[0] * self.input_depth,
                               self.kernel_size[1] * self.input_height,
                               self.kernel_size[2] * self.input_width,
                               self.num_channels)
        return outputs

    def deform_conv3d(self, inputs, offset):
        x, y, z = self._coordinate_map_3d(offset)
        deformed_feature = self._trilinear_interpolate(inputs, x, y, z)
        return deformed_feature

    def deform_conv3d_block(self, inputs):
        offset = self.deform_offset(inputs)
        offset = torch.tanh(self.deform_bn(offset))
        deformed_feature = self.deform_conv3d(inputs, offset)
        outputs = self.deform_conv(deformed_feature)
        outputs = F.relu(self.deform_bn_out(outputs))
        return outputs

    def spatial_attention_3d(self, x, sa_module):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        avg_pool = sa_module(avg_pool)
        avg_pool = torch.sigmoid(avg_pool)
        return x * avg_pool

    def channel_attention_3d(self, input_feature, ca_module):
        avg_pool = torch.mean(input_feature, dim=[2, 3, 4], keepdim=True)
        avg_pool = ca_module(avg_pool)
        avg_pool = torch.sigmoid(avg_pool)
        return input_feature * avg_pool

    def generator(self, input_x):
        # Initial convolution
        conv1 = F.relu(self.conv1(input_x))

        # First block
        conv2 = F.relu(self.conv2(conv1))
        conv3 = self.conv3(conv2)
        conv3 = conv1 + conv3
        conv4 = F.relu(self.conv4(conv3))
        conv5 = self.conv5(conv4)
        conv5 = conv3 + conv5
        conv6 = F.relu(self.conv6(conv5))
        conv6 = self.conv6_1(conv6)
        conv6 = conv6 + conv5
        conv6 = torch.cat([conv6, conv5, conv3, conv1], dim=1)
        at1 = self.channel_attention_3d(conv6, self.ca_at1)
        conv7 = F.relu(self.conv7(at1))
        conv7 = conv7 + conv1

        # Second block
        conv8 = F.relu(self.conv8(conv7))
        conv9 = self.conv9(conv8)
        conv9 = conv9 + conv7
        conv10 = F.relu(self.conv10(conv9))
        conv11 = self.conv11(conv10)
        conv11 = conv11 + conv9
        conv12 = F.relu(self.conv12(conv11))
        conv12 = self.conv12_1(conv12)
        conv12 = conv12 + conv11
        conv12 = torch.cat([conv12, conv11, conv9, conv7], dim=1)
        at2 = self.channel_attention_3d(conv12, self.ca_at2)
        conv13 = F.relu(self.conv13(at2))
        conv13 = conv13 + conv7

        # Third block
        conv14 = F.relu(self.conv14(conv13))
        conv15 = self.conv15(conv14)
        conv15 = conv15 + conv13
        conv16 = F.relu(self.conv16(conv15))
        conv17 = self.conv17(conv16)
        conv17 = conv17 + conv15
        conv18 = F.relu(self.conv18(conv17))
        conv18 = self.conv18_1(conv18)
        conv18 = conv18 + conv17
        conv18 = torch.cat([conv18, conv17, conv15, conv13], dim=1)
        at3 = self.channel_attention_3d(conv18, self.ca_at3)
        conv19 = F.relu(self.conv19(at3))
        conv19 = conv19 + conv13

        # Fourth block
        conv20 = F.relu(self.conv20(conv19))
        conv21 = self.conv21(conv20)
        conv21 = conv21 + conv19
        conv22 = F.relu(self.conv22(conv21))
        conv23 = self.conv23(conv22)
        conv23 = conv23 + conv21
        conv24 = F.relu(self.conv24(conv23))
        conv24 = self.conv24_1(conv24)
        conv24 = conv24 + conv23
        conv24 = torch.cat([conv24, conv23, conv21, conv19], dim=1)
        at4 = self.channel_attention_3d(conv24, self.ca_at4)
        conv25 = F.relu(self.conv25(at4))
        conv25 = conv25 + conv19

        # Fifth block
        conv26 = F.relu(self.conv26(conv25))
        conv27 = self.conv27(conv26)
        conv27 = conv27 + conv25
        conv28 = F.relu(self.conv28(conv27))
        conv29 = self.conv29(conv28)
        conv29 = conv29 + conv27
        conv30 = F.relu(self.conv30(conv29))
        conv30 = self.conv30_1(conv30)
        conv30 = conv30 + conv29
        conv30 = torch.cat([conv27, conv29, conv30, conv25], dim=1)
        at5 = self.channel_attention_3d(conv30, self.ca_at5)
        conv31 = F.relu(self.conv31(at5))
        conv31 = conv31 + conv25

        # Final concatenation and output
        conv32 = torch.cat([conv31, conv25, conv19, conv13, conv7, conv1], dim=1)
        conv32 = self.channel_attention_3d(conv32, self.ca_at6)
        conv32 = self.conv32(conv32)
        conv32 = self.spatial_attention_3d(conv32, self.sa_conv32)
        out = conv32 + input_x
        return out

    def forward(self, input_source):
        fake = self.generator(input_source)
        return fake