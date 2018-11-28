import os
import scipy
import scipy.misc

import torch
import numpy as np

from carla.agent import Agent
from carla.carla_server_pb2 import Control
from agents.imitation.modules.carla_net import CarlaNet
from agents.imitation.modules.gan import define_G


class ImitationLearning(Agent):

    def __init__(self, city_name,
                 avoid_stopping=True,
                 model_path="model/policy.pth",
                 vrg_transfer=False,
                 vrg_model_path="model/transfer.pth",
                 trans_direction="B2A",
                 visualize=False,
                 image_cut=[115, 510]):

        super(ImitationLearning, self).__init__()
        # Agent.__init__(self)

        self._image_size = (88, 200, 3)
        self._avoid_stopping = avoid_stopping

        dir_path = os.path.dirname(__file__)
        self._models_path = os.path.join(dir_path, model_path)
        self.model = CarlaNet()
        if torch.cuda.is_available():
            self.model.cuda()
        self.load_model()
        self.model.eval()

        self.vrg_transfer = vrg_transfer
        if vrg_transfer:
            self.trans_direction = trans_direction
            self._vrg_models_path = os.path.join(
                dir_path, vrg_model_path)
            dtype = torch.cuda.FloatTensor \
                if torch.cuda.is_available() else torch.FloatTensor
            self.transfer_model = define_G(
                3, 3, 64, "resnet_9blocks", norm="instance", use_dropout=False,
                enable_progressive=False, progress_start=1,
                progress_chas=[256, 128, 64, 32, 16], progress_inc=2,
                progress_kernel=3, enable_lstm=False, flo_len=0,
                flo_mode="max", flo_indices=[], batch_size=1, lstm_cha=256,
                lstm_hei=64, lstm_wid=64, dtype=dtype)
            if torch.cuda.is_available():
                self.transfer_model.cuda()
            self.load_transfer_model()
            self.transfer_model.eval()

            self.visualize = visualize
            if self.visualize:
                from tensorboardX import SummaryWriter
                self.writer = SummaryWriter(
                    os.path.join(
                        dir_path+'/runs/',
                        "carla_transfer_visualize"))
                self.step = 0

        self._image_cut = image_cut

    def load_model(self):
        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path: %s'%self._models_path)
        checkpoint = torch.load(self._models_path)
        self.model.load_state_dict(checkpoint['state_dict'])

    def load_transfer_model(self):
        if not os.path.exists(self._vrg_models_path):
            raise RuntimeError('failed to find the models path: %s'%self._vrg_models_path)
        pretrained_dict = torch.load(self._vrg_models_path)
        partial_dict = {}
        if self.trans_direction == 'B2A':
            for k, v in pretrained_dict.items():
                if 'netG_B' in k:
                    partial_dict[k[7:]] = pretrained_dict[k]
        if self.trans_direction == 'A2B':
            for k, v in pretrained_dict.items():
                if 'netG_A' in k:
                    partial_dict[k[7:]] = pretrained_dict[k]
        self.transfer_model.load_state_dict(partial_dict)

    def run_step(self, measurements, sensor_data, directions, target):

        control = self._compute_action(
            sensor_data['CameraRGB'].data,
            measurements.player_measurements.forward_speed,
            directions)

        return control

    def _compute_action(self, rgb_image, speed, direction=None):

        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]

        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0],
                                                      self._image_size[1]])

        image_input = image_input.astype(np.float32)
        image_input = np.expand_dims(
            np.transpose(image_input, (2, 0, 1)),
            axis=0)

        image_input = np.multiply(image_input, 1.0 / 255.0)
        speed = np.array([[speed]]).astype(np.float32) / 25.0
        direction = int(direction-2)

        steer, acc, brake = self._control_function(image_input,
                                                   speed,
                                                   direction)

        # This a bit biased, but is to avoid fake breaking

        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        # We limit speed to 35 km/h to avoid
        if speed > 10.0 and brake == 0.0:
            acc = 0.0

        if np.abs(steer) > 0.15:
            acc = acc * 0.4

        control = Control()
        control.steer = steer
        control.throttle = acc
        control.brake = brake

        control.hand_brake = 0
        control.reverse = 0

        return control

    def _control_function(self, image_input, speed, control_input):

        img_ts = torch.from_numpy(image_input).cuda()
        speed_ts = torch.from_numpy(speed).cuda()

        with torch.no_grad():
            if self.vrg_transfer:
                img_ts_trans = img_ts * 2 - 1
                img_ts_trans = self.transfer_model.forward_seq(img_ts_trans)
                img_ts_trans = (img_ts_trans + 1) / 2.0
                if self.visualize:
                    self.writer.add_image("carla_visual/origin",
                                          img_ts[0],
                                          self.step)
                    self.writer.add_image("carla_visual/trans",
                                          img_ts_trans[0],
                                          self.step)
                    self.step += 1
                branches, pred_speed = self.model(img_ts_trans, speed_ts)
            else:
                branches, pred_speed = self.model(img_ts, speed_ts)

        pred_result = branches[0][
            3*control_input:3*(control_input+1)].cpu().numpy()

        predicted_steers = (pred_result[0])

        predicted_acc = (pred_result[1])

        predicted_brake = (pred_result[2])

        if self._avoid_stopping:
            predicted_speed = pred_speed.squeeze().item()
            real_speed = speed * 25.0

            real_predicted = predicted_speed * 25.0
            if real_speed < 2.0 and real_predicted > 3.0:

                predicted_acc = 1 * (5.6 / 25.0 - speed) + predicted_acc

                predicted_brake = 0.0

                predicted_acc = predicted_acc

        return predicted_steers, predicted_acc, predicted_brake
