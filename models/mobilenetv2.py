from torch import nn
import numpy as np


class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=None, groups=1):
        """
        Args:
            in_channels: the number of a input channel
            out_channels: the number of an output channel
            kernel_size: the kernal size
            stride: the stride size
            groups: ???
        """
        if padding is None:
            padding = (kernel_size - 1) // 2
            
        super(ConvBNReLU6, self).__init__(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio,
                 use_res_connect=True, linear_bottleneck=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        if not use_res_connect:
            self.use_res_connect = False
        else:
            self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        
        # Expand the number of a channel
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_channels, hidden_dim, kernel_size=1))

        # Depthwise + Pointwise Convolution
        # There is no ReLU after Pointwise Convolution
        if linear_bottleneck:
            layers.extend([ConvBNReLU6(hidden_dim, hidden_dim,
                                       stride=stride, groups=hidden_dim),
                           nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels,
                                     kernel_size=1, stride=1, padding=0, bias=False),
                           nn.BatchNorm2d(out_channels)])
        else:
            layers.extend([ConvBNReLU6(hidden_dim, hidden_dim,
                                       stride=stride, groups=hidden_dim),
                           ConvBNReLU6(hidden_dim, out_channels, kernel_size=1,
                                       stride=1, padding=0)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DepthProjExpBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, stride, expand_ratio,
                 use_res_connect=True, linear_bottleneck=True):
        super(DepthProjExpBlock, self).__init__()
        self.stride = stride
        if not use_res_connect:
            self.use_res_connect = False
        else:
            self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []

        # Depthwise + Pointwise Convolution
        # There is no ReLU after Pointwise Convolution
        if linear_bottleneck:
            layers.extend([ConvBNReLU6(hidden_dim, hidden_dim,
                                       stride=stride, groups=hidden_dim),
                           nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels,
                                     kernel_size=1, stride=1, padding=0, bias=False),
                           nn.BatchNorm2d(out_channels)])
        else:
            layers.extend([ConvBNReLU6(hidden_dim, hidden_dim,
                                       stride=stride, groups=hidden_dim),
                           ConvBNReLU6(hidden_dim, out_channels, kernel_size=1,
                                       stride=1, padding=0)])

        # Expand the number of a 
        in_channels = out_channels
        hidden_dim = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_channels, hidden_dim, kernel_size=1))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ProjExpDepthBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, stride, expand_ratio,
                 use_res_connect=True, linear_bottleneck=True):
        super(ProjExpDepthBlock, self).__init__()
        self.stride = stride
        if not use_res_connect:
            self.use_res_connect = False
        else:
            self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []

        # Depthwise + Pointwise Convolution
        # There is no ReLU after Pointwise Convolution
        if linear_bottleneck:
            layers.extend([ConvBNReLU6(hidden_dim, hidden_dim,
                                       stride=stride, groups=hidden_dim),
                           nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels,
                                     kernel_size=1, stride=1, padding=0, bias=False),
                           nn.BatchNorm2d(out_channels)])
        else:
            layers.extend([ConvBNReLU6(hidden_dim, hidden_dim,
                                       stride=stride, groups=hidden_dim),
                           ConvBNReLU6(hidden_dim, out_channels, kernel_size=1,
                                       stride=1, padding=0)])

        # Expand the number of a 
        in_channels = out_channels
        hidden_dim = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_channels, hidden_dim, kernel_size=1))

        ConvBNReLU6(hidden_dim, hidden_dim, stride=1, groups=hidden_dim)

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, round_nearest=8,
                 use_res_connect=True, linear_bottleneck=True, res_loc=0,
                 inverted_residual_setting=[], first_layer_stride=2):
                 
        def _make_divisible(v, divisor, min_value=None):
            """It ensures that all layers have a channel number that is divisible by a divisor (e.g. 8).
               This function is implemented original repository:
               https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

               For example, if v = 109, divisor = 8, min_value = 16,
               new_v = int(109.5 + 8 / 2) // 8 * 8 = 113.5 // 8 * 8 = 112.
               
               This result shows that this method return the bigger number than v
               which is divisable by 8 if v is close to that number.

               If the number is more close to smaller number than v
               which is divisable by 8, the method return that number.

            Args:
                v: the number of a current channel
                divsor: 
                min_value: 
            Returns:
                new_v: 
            """
            # Make the number of channel diviable by a divisor
            new_v = max(min_value if min_value is not None else divisor,
                        int(v + divisor / 2) // divisor * divisor)

            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        super(MobileNetV2, self).__init__()
        BOTTLENECKS, EXPLOSIONS, DEPTHWISE = 0, 1, 2

        # Build a first layer.
        # I don't know why max function was used for last_channel.
        # I think it decreases performance to reduce the number of last channel with width_mult.
        input_channel, last_channel = 32, 1280
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU6(in_channels=3,
                                out_channels=input_channel,
                                stride=first_layer_stride)]

        if res_loc == BOTTLENECKS:
            # Build inverted residual blocks.
            for t, c, n, s in inverted_residual_setting:
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1 # stride is applied only in the first block.
                    features.append(InvertedResidual(input_channel, output_channel,
                                                     stride, expand_ratio=t,
                                                     use_res_connect=use_res_connect,
                                                     linear_bottleneck=linear_bottleneck))
                    input_channel = output_channel
        elif res_loc == EXPLOSIONS:
            for t, c, n, s in inverted_residual_setting:
                # Expand the number of a channel (first)
                hidden_dim = int(round(input_channel * t))
                if t != 1:
                    features.append(ConvBNReLU6(input_channel, hidden_dim, kernel_size=1))

                # DepthProjExpBlocks
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n-1):
                    stride = s if i == 0 else 1 # stride is applied only in the first block.
                    features.append(DepthProjExpBlock(input_channel, hidden_dim, output_channel,
                                                      stride, expand_ratio=t,
                                                      use_res_connect=use_res_connect,
                                                      linear_bottleneck=linear_bottleneck))
                    input_channel = output_channel
                    hidden_dim = int(round(input_channel * t))
                
                # Depthwise + Pointwise Convolution
                stride = s if (n-1) == 0 else stride
                if linear_bottleneck:
                    features.extend([ConvBNReLU6(hidden_dim, hidden_dim,
                                                 stride=stride, groups=hidden_dim),
                                     nn.Conv2d(in_channels=hidden_dim, out_channels=output_channel,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(output_channel)])
                else:
                    features.extend([ConvBNReLU6(hidden_dim, hidden_dim,
                                                 stride=stride, groups=hidden_dim),
                                     ConvBNReLU6(hidden_dim, output_channel, kernel_size=1,
                                                 stride=1, padding=0)])
                input_channel = output_channel
        elif res_loc == DEPTHWISE:
            for t, c, n, s in inverted_residual_setting:
                # Expand the number of a channel (first)
                hidden_dim = int(round(input_channel * t))
                if t != 1:
                    features.append(ConvBNReLU6(input_channel, hidden_dim, kernel_size=1))

                # Depthwise Layer
                ConvBNReLU6(hidden_dim, hidden_dim, stride=s, groups=hidden_dim)

                # ProjExpDepthBlock
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n-1):
                    stride = s if i == 0 else 1 # stride is applied only in the first block.
                    features.append(ProjExpDepthBlock(input_channel, hidden_dim, output_channel,
                                                      stride, expand_ratio=t,
                                                      use_res_connect=use_res_connect,
                                                      linear_bottleneck=linear_bottleneck))
                    input_channel = output_channel
                    hidden_dim = int(round(input_channel * t))
                
                # Pointwise Convolution
                if linear_bottleneck:
                    features.extend([nn.Conv2d(in_channels=hidden_dim, out_channels=output_channel,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(output_channel)])
                else:
                    features.append([ConvBNReLU6(hidden_dim, output_channel, kernel_size=1,
                                                 stride=1, padding=0)])
                input_channel = output_channel
        else:
            assert False, 'You choose a wrong location for residual connections.'

        # Build a last layer.
        features.append(ConvBNReLU6(input_channel, self.last_channel, kernel_size=1))
        
        # Convert list to nn.Sequential.
        self.features = nn.Sequential(*features)

        # Build a classifier.
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(self.last_channel, num_classes))

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # the shape of adaptive_avg_pool2d output: (B, C, 1, 1)
        # the shape of reshape result: (B, C)
        # Reference: https://wikidocs.net/52846
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        # x = nn.functional.avg_pool2d(x, 4).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        # I don't know why we should implement _forward_impl function...
        return self._forward_impl(x)

    def get_param_num(self):
        """https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        param_num = sum([np.prod(p.size()) for p in model_parameters])
        return param_num
