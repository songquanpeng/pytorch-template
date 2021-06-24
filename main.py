from munch import Munch
from config import load_cfg, save_cfg
from utils.misc import basic_setup
from solver.solver import Solver
from data.dataloader import get_train_loader, get_test_loader


def main(args):
    print(args.__dict__)
    basic_setup(args)
    if args.mode == 'train':
        solver = Solver(args)
        loaders = Munch(train=get_train_loader(args),
                        test=get_test_loader(args))
        solver.train(loaders)
    elif args.mode == 'sample':
        pass
    elif args.mode == 'eval':
        pass
    else:
        assert False


if __name__ == '__main__':
    cfg = load_cfg()
    save_cfg(cfg)
    main(cfg)
