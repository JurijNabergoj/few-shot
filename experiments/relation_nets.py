import math

from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
import argparse

import sys

sys.path.append("/few_shot/")
from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.relation import relation_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH
from few_shot.models import RelationNetwork


setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='omniglot')
parser.add_argument('--n-train', default=1, type=int)
parser.add_argument('--n-test', default=1, type=int)
parser.add_argument('--k-train', default=5, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=15, type=int)
parser.add_argument('--q-test', default=15, type=int)
parser.add_argument('--preload_model', default=False, type=bool)
args = parser.parse_args()

evaluation_episodes = 1000
episodes_per_epoch = 100


def load_network(model, param_string):
    """Loads best previous model weights."""
    model_file = str(PATH + "/models/relation_nets/" + param_string + ".pth")
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
        print("Model loaded.")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


if __name__ == '__main__':
    if args.dataset == 'omniglot':
        n_epochs = 40
        dataset_class = OmniglotDataset
        num_input_channels = 1
        drop_lr_every = 20
    elif args.dataset == 'miniImageNet':
        n_epochs = 80
        dataset_class = MiniImageNet
        num_input_channels = 3
        drop_lr_every = 40
    else:
        raise (ValueError, 'Unsupported dataset')

    param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
                f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

    print(param_str)

    ###################
    # Create datasets #
    ###################
    background = dataset_class('background')
    background_taskloader = DataLoader(
        background,
        batch_sampler=NShotTaskSampler(background, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
        num_workers=4
    )
    evaluation = dataset_class('evaluation')
    evaluation_taskloader = DataLoader(
        evaluation,
        batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
        num_workers=4
    )

    #########
    # Model #
    #########
    model = RelationNetwork(args.n_train,
                            args.k_train,
                            args.q_train,
                            num_input_channels,
                            device=device)
    model.encoder.apply(weights_init)
    model.relation.apply(weights_init)
    if args.preload_model:
        load_network(model, param_str)
    model.to(device, dtype=torch.double)

    ############
    # Training #
    ############
    print(f'Training Relation network on {args.dataset}...')
    encoder_optimiser = Adam(model.encoder.parameters(), lr=1e-3)
    relation_optimiser = Adam(model.relation.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()


    class MultiModelOptimizer:

        def __init__(self, params1, params2):
            self.opt1 = Adam(params1, lr=1e-3)
            self.opt2 = Adam(params2, lr=1e-3)

        def zero_grad(self):
            self.opt1.zero_grad()
            self.opt2.zero_grad()

        def step(self):
            self.opt1.step()
            self.opt2.step()

        @property
        def param_groups(self):
            return self.opt1.param_groups + self.opt2.param_groups


    def lr_schedule(epoch, lr):
        # Drop lr every 2000 episodes
        if epoch % drop_lr_every == 0:
            return lr / 2
        else:
            return lr


    callbacks = [
        EvaluateFewShot(
            eval_fn=relation_net_episode,
            num_tasks=evaluation_episodes,
            n_shot=args.n_test,
            k_way=args.k_test,
            q_queries=args.q_test,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
        ),
        ModelCheckpoint(
            filepath=(PATH + f'/models/relation_nets/{param_str}.pth').replace("\\", "/"),
            monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'
        ),
        LearningRateScheduler(schedule=lr_schedule),
        CSVLogger((PATH + f'/logs/relation_nets/{param_str}.csv').replace("\\", "/")),
    ]
    fit(
        model,
        optimiser=MultiModelOptimizer(model.encoder.parameters(), model.relation.parameters()),
        loss_fn=loss_fn,
        epochs=n_epochs,
        dataloader=background_taskloader,
        prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        fit_function=relation_net_episode,
        fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True},
    )
