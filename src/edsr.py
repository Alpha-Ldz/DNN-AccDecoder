import torch

import EDSR.src.utility as utility
import EDSR.src.data as data
import EDSR.src.model as model
import EDSR.src.loss as loss
import attr
from EDSR.src.trainer import Trainer

@attr.s
class Args:
    debug = attr.ib(default=False)
    template = attr.ib(default='.')

    # Hardware specifications
    n_threads = attr.ib(default=6)
    cpu = attr.ib(default=False)
    n_GPUs = attr.ib(default=1)
    seed = attr.ib(default=1)

    # Data specifications
    dir_data = attr.ib(default='../../../dataset')
    dir_demo = attr.ib(default='../test')
    data_train = attr.ib(default='DIV2K')
    data_test = attr.ib(default='DIV2K')
    data_range = attr.ib(default='1-800/801-810')
    ext = attr.ib(default='sep')
    scale = attr.ib(default=4)
    patch_size = attr.ib(default=192)
    rgb_range = attr.ib(default=255)
    n_colors = attr.ib(default=3)
    chop = attr.ib(default=False)
    no_augment = attr.ib(default=False)

    # Model specifications
    model = attr.ib(default='EDSR')

    act = attr.ib(default='relu')
    pre_train = attr.ib(default='')
    extend = attr.ib(default='.')
    n_resblocks = attr.ib(default=16)
    n_feats = attr.ib(default=64)
    res_scale = attr.ib(default=1)
    shift_mean = attr.ib(default=True)
    dilation = attr.ib(default=False)
    precision = attr.ib(default='single')

    # Option for Residual dense network (RDN)
    G0 = attr.ib(default=64)
    RDNkSize = attr.ib(default=3)
    RDNconfig = attr.ib(default='B')

    # Option for Residual channel attention network (RCAN)
    n_resgroups = attr.ib(default=10)
    reduction = attr.ib(default=16)

    # Training specifications
    reset = attr.ib(default=False)
    test_every = attr.ib(default=1000)
    epochs = attr.ib(default=300)
    batch_size = attr.ib(default=16)
    split_batch = attr.ib(default=1)
    self_ensemble = attr.ib(default=False)
    test_only = attr.ib(default=False)
    gan_k = attr.ib(default=1)

    # Optimization specifications
    lr = attr.ib(default=1e-4)
    decay = attr.ib(default=200)
    gamma = attr.ib(default=0.5)
    optimizer = attr.ib(default='ADAM')
    momentum = attr.ib(default=0.9)
    betas = attr.ib(default=(0.9, 0.999))
    epsilon = attr.ib(default=1e-8)
    weight_decay = attr.ib(default=0)
    gclip = attr.ib(default=0)

    # Loss specifications
    loss = attr.ib(default='1*L1')
    skip_threshold = attr.ib(default=1e8)

    # Log specifications
    save = attr.ib(default='test')
    load = attr.ib(default='')
    resume = attr.ib(default=0)
    save_models = attr.ib(default=True)
    print_every = attr.ib(default=100)
    save_results = attr.ib(default=False)
    save_gt = attr.ib(default=False)

args = Args()


torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def EDSR():
    global model
    if args.data_test == ['video']:
        from EDSR.src.videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()