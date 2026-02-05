from graphgps.optuna.args import TrainArgs
from graphgps.data.data_molalkit import get_data
from graphgps.optuna.graphgps import GraphGPS
from graphgps.optuna.evaluator import Evaluator
from graphgps.run.utils import *


def graphgps_cv(arguments=None):
    from torch_geometric.graphgym.config import cfg
    args = TrainArgs().parse_args(arguments)
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    cfg.out_dir = args.save_dir
    cfg.dataset.task_type = args.task_type
    if args.task_type == 'binary':
        cfg.model.loss_fun = 'cross_entropy'
    else:
        cfg.model.loss_fun = 'mse'
    dump_cfg(cfg)
    auto_select_device()
    # Set Pytorch environment
    torch.set_num_threads(args.n_jobs)
    args.dataset.compute_posenc_stats()
    model = GraphGPS(save_dir=args.save_dir,
                     cfg_path=args.cfg_file,
                     n_features=len(args.features_columns) if args.features_columns is not None else 0,
                     features_generators_name=args.features_generators_name,
                     number_of_molecules=len(args.smiles_columns),
                     ensemble_size=args.ensemble_size,
                     n_jobs=args.n_jobs,
                     seed=args.seed,
                     features_scaling=args.features_scaling)
    evaluator = Evaluator(save_dir=args.save_dir,
                          dataset=args.dataset,
                          model=model,
                          task_type=cfg.dataset.task_type,
                          metrics=args.metrics,
                          cross_validation=args.cross_validation,
                          n_splits=args.n_splits,
                          split_type=args.split_type,
                          split_sizes=args.split_sizes,
                          num_folds=args.num_folds,
                          seed=args.seed)
    if args.separate_test_path is not None:
        args.dataset_test.compute_posenc_stats()
        evaluator.run_external(args.dataset_test)
    else:
        evaluator.run_cross_validation()
