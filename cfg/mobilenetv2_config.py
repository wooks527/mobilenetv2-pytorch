class Config:
    def __init__(self, dataset):
        if dataset == 'cifar10':
            self.inverted_residual_setting = [
                # t: expand ratio, c: output channel,
                # n: repeat times, s: stride
                # -----------------------------------
                # t, c, n, s      Input (Base)
                [1, 16, 1, 1],  #  32 x  32 x 32
                [6, 24, 2, 1],  #  32 x  32 x 16, NOTE: change stride 2 -> 1 for CIFAR10
                [6, 32, 3, 2],  #  56 x  56 x 24
                [6, 64, 4, 2],  #  28 x  28 x 32
                [6, 96, 3, 1],  #  14 x  14 x 64
                [6, 160, 3, 2], #  14 x  14 x 96
                [6, 320, 1, 1], #   7 x   7 x 160
            ]
            self.num_cls = 10
            self.first_layer_stride = 1
        elif dataset == 'cifar100':
            self.inverted_residual_setting = [
                # t: expand ratio, c: output channel,
                # n: repeat times, s: stride
                # -----------------------------------
                # t, c, n, s      Input (Base)
                [1, 16, 1, 1],  #  32 x  32 x 32
                [6, 24, 2, 1],  #  32 x  32 x 16, NOTE: change stride 2 -> 1 for CIFAR10
                [6, 32, 3, 2],  #  56 x  56 x 24
                [6, 64, 4, 2],  #  28 x  28 x 32
                [6, 96, 3, 1],  #  14 x  14 x 64
                [6, 160, 3, 2], #  14 x  14 x 96
                [6, 320, 1, 1], #   7 x   7 x 160
            ]
            self.num_cls = 100
            self.first_layer_stride = 1
        elif dataset == 'imagenet':
            self.inverted_residual_setting = [
                # t: expand ratio, c: output channel,
                # n: repeat times, s: stride
                # -----------------------------------
                # t, c, n, s      Input (Base)
                [1, 16, 1, 1],  #  32 x  32 x 32
                [6, 24, 2, 2],  #  32 x  32 x 16
                [6, 32, 3, 2],  #  56 x  56 x 24
                [6, 64, 4, 2],  #  28 x  28 x 32
                [6, 96, 3, 1],  #  14 x  14 x 64
                [6, 160, 3, 2], #  14 x  14 x 96
                [6, 320, 1, 1], #   7 x   7 x 160
            ]
            self.num_cls = 1000
            self.first_layer_stride = 2
        else:
            assert False, 'You choose wrong datset.'