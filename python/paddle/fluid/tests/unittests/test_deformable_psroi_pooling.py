#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import unittest
import numpy as np
from op_test import OpTest

class TestDeformablePSROIPoolOp(OpTest):
    def set_data(self):
        self.init_test_case()
        self.make_rois()
        self.calc_deformable_psroi_pooling()
        self.inputs = {'Input': self.input, "ROIs": (self.rois[:, 1:5],
                                                     self.rois_lod), "Trans": self.trans}
        self.attrs = {
            'no_trans': self.no_trans,
            'spatial_scale': self.spatial_scale,
            'output_dim': self.output_channels,
            'group_size': self.group_size,
            'pooled_height': self.pooled_height,
            'pooled_width': self.pooled_width,
            'part_size': self.part_size,
            'sample_per_part': self.sample_per_part,
            'trans_std': self.trans_std
        }

        self.outputs = {'Output': self.out.astype('float32'),
                        'TopCount': self.top_count.astype('float32')}

    def init_test_case(self):
        self.batch_size = 3
        self.channels = 3 * 2 * 2
        self.height = 12
        self.width = 12
        self.input_dim = [self.batch_size, self.channels,
                          self.height, self.width]
        self.no_trans = 1
        self.spatial_scale = 1.0 / 4.0
        self.output_channels = 12
        self.group_size = [1, 1]
        self.pooled_height = 4
        self.pooled_width = 4
        #self.pooled_size=4
        self.part_size = [4, 4]
        self.sample_per_part = 2
        self.trans_std = 0.1
        self.input = np.random.random(self.input_dim).astype('float32')

    def make_rois(self):
        rois = []
        self.rois_lod = [[]]
        for bno in range(self.batch_size):
            self.rois_lod[0].append(bno + 1)
            for i in range(bno + 1):
                x_1 = np.random.random_integers(
                    0, self.width // self.spatial_scale - self.pooled_width)
                y_1 = np.random.random_integers(
                    0, self.height // self.spatial_scale - self.pooled_height)
                x_2 = np.random.random_integers(x_1 + self.pooled_width,
                                                self.width // self.spatial_scale)
                y_2 = np.random.random_integers(
                    y_1 + self.pooled_height, self.height // self.spatial_scale)
                roi = [bno, x_1, y_1, x_2, y_2]
                rois.append(roi)
        self.rois_num = len(rois)
        self.rois = np.array(rois).astype("float32")

    def dmc_bilinear(self, data_im, p_h, p_w):
        h_low = int(np.floor(p_h))
        w_low = int(np.floor(p_w))
        h_high = h_low + 1
        w_high = w_low + 1
        l_h = p_h - h_low
        l_w = p_w - w_low
        h_h = 1 - l_h
        h_w = 1 - l_w
        v_1 = 0
        if h_low >= 0 and w_low >= 0:
            v_1 = data_im[h_low, w_low]
        v_2 = 0
        if h_low >= 0 and w_high <= self.width - 1:
            v_2 = data_im[h_low, w_high]
        v_3 = 0
        if h_high <= self.height - 1 and w_low >= 0:
            v_3 = data_im[h_high, w_low]
        v_4 = 0
        if h_high <= self.height - 1 and w_high <= self.width - 1:
            v_4 = data_im[h_high, w_high]
        w_1, w_2, w_3, w_4 = h_h * h_w, h_h * l_w, l_h * h_w, l_h * l_w
        val = w_1 * v_1 + w_2 * v_2 + w_3 * v_3 + w_4 * v_4
        return val
 
    def calc_deformable_psroi_pooling(self):
        output_shape = (self.rois_num, self.output_channels,
                        self.pooled_height, self.pooled_width)
        self.out = np.zeros(output_shape)
        self.trans = np.random.rand(self.rois_num, 2, self.part_size[0],
                                    self.part_size[1]).astype('float32')
        self.top_count = np.random.random((output_shape)).astype('float32')
        count = self.rois_num * self.output_channels * self.pooled_height * self.pooled_width
        for index in range(count):
            p_w = int(index % self.pooled_width)
            p_h = int(index / self.pooled_width % self.pooled_height)
            ctop = int(index / self.pooled_width / self.pooled_height % self.output_channels)
            n_out = int(index / self.pooled_width / self.pooled_height / self.output_channels)
            roi = self.rois[n_out]
            roi_batch_id = int(roi[0])
            roi_start_w = int(np.round(roi[1])) * self.spatial_scale - 0.5
            roi_start_h = int(np.round(roi[2])) * self.spatial_scale - 0.5
            roi_end_w = int(np.round(roi[3]+1)) * self.spatial_scale - 0.5
            roi_end_h = int(np.round(roi[4]+1)) * self.spatial_scale - 0.5
            roi_width = max(roi_end_w - roi_start_w, 0.1)
            roi_height = max(roi_end_h - roi_start_h, 0.1)
            bin_size_h = float(roi_height) / float(self.pooled_height)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            sub_bin_size_h = bin_size_h / self.sample_per_part
            sub_bin_size_w = bin_size_w / self.sample_per_part
            part_h = int(np.floor(p_h) / self.pooled_height * self.part_size[0])
            part_w = int(np.floor(p_w) / self.pooled_width * self.part_size[1])
            if self.no_trans:
                trans_x = 0
                trans_y = 0
            else:
                trans_x = self.trans[n_out][0][part_h][part_w] * self.trans_std
                trans_y = self.trans[n_out][1][part_h][part_w] * self.trans_std
            wstart = p_w * bin_size_w + roi_start_w
            wstart = wstart + trans_x * roi_width
            hstart = p_h * bin_size_h + roi_start_h
            hstart = hstart + trans_y * roi_height
            sum = 0
            num_sample = 0
            g_w = np.floor(p_w * self.group_size[0] / self.pooled_height)
            g_h = np.floor(p_h * self.group_size[1] / self.pooled_width)
            g_w = min(max(g_w, 0), self.group_size[0] - 1)
            g_h = min(max(g_h, 0), self.group_size[1] - 1)
            #print(input[n, 1])
            input_i = self.input[roi_batch_id]
            for i_w in range(self.sample_per_part):
                for i_h in range(self.sample_per_part):
                    w_sample = wstart + i_w * sub_bin_size_w
                    h_sample = hstart + i_h * sub_bin_size_h
                    if w_sample < -0.5 or w_sample > self.width - 0.5 or \
                    h_sample < -0.5 or h_sample > self.height - 0.5:
                        continue
                    w_sample = min(max(w_sample, 0.), self.width - 1.)
                    h_sample = min(max(h_sample, 0.), self.height - 1.)
                    c_sample = int((ctop * self.group_size[0] + g_h) * self.group_size[1] + g_w)
                    val = self.dmc_bilinear(input_i[c_sample], h_sample, w_sample)
                    sum = sum + val
                    num_sample = num_sample + 1
            if num_sample == 0:
                self.out[n_out][ctop][p_h][p_w] = 0
            else:
                self.out[n_out][ctop][p_h][p_w] = sum / num_sample
            self.top_count[n_out][ctop][p_h][p_w] = num_sample
 
    def setUp(self):
        self.op_type = "deformable_psroi_pooling"
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Input'], 'Output')

if __name__ == '__main__':
    unittest.main()
