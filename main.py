from munch import Munch

from config import setup_cfg, validate_cfg, load_cfg, save_cfg, print_cfg
from data.loader import get_train_loader, get_test_loader, get_selected_loader
from solver.solver import Solver


def main(args):
    solver = Solver(args)
    if args.mode == 'train':
        loaders = Munch(train=get_train_loader(**args), test=get_test_loader(**args))
        if args.selected_path:
            loaders.selected = get_selected_loader(**args)
        solver.train(loaders)
    elif args.mode == 'sample':
        solver.sample()
    elif args.mode == 'eval':
        solver.evaluate()
    else:
        assert False, f"Unimplemented mode: {args.mode}"


if __name__ == '__main__':
    cfg = load_cfg()
    setup_cfg(cfg)
    validate_cfg(cfg)
    save_cfg(cfg)
    print_cfg(cfg)
    main(cfg)
