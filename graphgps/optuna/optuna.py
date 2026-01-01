import optuna
from optuna.samplers import TPESampler
from graphgps.optuna.args import OptunaArgs
from graphgps.data.data_molalkit import get_data
from graphgps.optuna.graphgps import GraphGPS
from graphgps.optuna.evaluator import Evaluator
from graphgps.run.utils import *


def graphgps_optuna(arguments=None):
    from torch_geometric.graphgym.config import cfg
    args = OptunaArgs().parse_args(arguments)
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    cfg.out_dir = args.save_dir
    # modify cfg based on the arguments
    if args.features_generators_name is not None:
        cfg.gnn.use_features = True
        n_features = 0
        for fg in args.features_generators_name:
            if fg in ['rdkit_2d', 'rdkit_2d_normalized']:
                n_features += 200
            elif fg in ['morgan', 'morgan_count']:
                n_features += 2048
            else:
                raise ValueError(f"Unknown features generator: {fg}")
        cfg.gnn.n_features = n_features * len(args.smiles_columns)
    cfg.dataset.task_type = args.dataset_type
    if args.dataset_type == 'classification':
        cfg.model.loss_fun = 'cross_entropy'
    else:
        cfg.model.loss_fun = 'mse'
    dump_cfg(cfg)
    auto_select_device()
    # Set Pytorch environment
    torch.set_num_threads(args.n_jobs)

    def objective(trial):
        PE_type = trial.suggest_categorical('PE_type', ['', 'RWSE', 'LapPE', 'SignNet']) #  'EquivStableLapPE',
        cfg.dataset.node_encoder_name = 'Atom' if PE_type == '' else 'Atom+%s' % PE_type
        if PE_type == 'RWSE':
            cfg.posenc_RWSE.enable = True
        elif PE_type == 'LapPE':
            cfg.posenc_LapPE.enable = True
        # elif PE_type == 'EquivStableLapPE':
        #     cfg.posenc_EquivStableLapPE.enable = True
        elif PE_type == 'SignNet':
            cfg.posenc_SignNet.enable = True
        params = {
            'model.graph_pooling': trial.suggest_categorical('graph_pooling', ['mean', 'add']),
            'gt.layer_type': trial.suggest_categorical('layer_type', ['GINE+Transformer', 'CustomGatedGCN+Transformer', 'GCN+Transformer', 'CustomGatedGCN+Performer']),
            'gt.layers': trial.suggest_int('layers', 2, 10),
            'gt.n_heads': trial.suggest_int('n_heads', 4, 8, step=4),
            'gt.dropout': trial.suggest_float('dropout', 0.0, 0.4, step=0.05),
            'gt.attn_dropout': trial.suggest_float('attn_dropout', 0.0, 0.5, step=0.25),
            'gt.dim_hidden': trial.suggest_categorical('dim_hidden', [64, 128, 256, 512]),
            'gnn.layers_post_mp': trial.suggest_int('layers_post_mp', 1, 5),
            'gnn.agg': trial.suggest_categorical('gnn_aggr', ['mean', 'add']),
            'optim.weight_decay': trial.suggest_categorical('weight_decay', [0.0, 1e-5, 1e-4, 1e-3]),
            'optim.base_lr': trial.suggest_categorical('base_lr', [1e-4, 5e-4, 1e-3]),
            'optim.max_epoch': trial.suggest_categorical('max_epoch', [50, 100, 200]),
            'train.batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        }
        params['optim.num_warmup_epochs'] = int(0.1 * params['optim.max_epoch'])
        params['gnn.dropout'] = params['gt.dropout']
        params['gnn.dim_inner'] = params['gt.dim_hidden']

        params_list = []
        for key, value in params.items():
            params_list.append(key)
            params_list.append(value)
        cfg.merge_from_list(params_list)

        args.dataset.compute_posenc_stats()
        model = GraphGPS(save_dir='%s/trial-%d' % (args.save_dir, trial.number),
                         cfg_path=None,
                         features_generators_name=args.features_generators_name,
                         number_of_molecules=1,
                         n_jobs=args.n_jobs,
                         seed=args.seed)
        evaluator = Evaluator(save_dir='%s/trial-%d' % (args.save_dir, trial.number),
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
        if args.separate_val_path is not None:
            args.dataset_val.compute_posenc_stats()
            if args.separate_test_path is not None:
                args.dataset_train_val.compute_posenc_stats()
                args.dataset_test.compute_posenc_stats()
                evaluator1 = Evaluator(save_dir='%s/trial-%d' % (args.save_dir, trial.number),
                                        dataset=args.dataset_train_val,
                                        model=model,
                                        task_type=cfg.dataset.task_type,
                                        metrics=args.metrics,
                                        cross_validation=args.cross_validation,
                                        n_splits=args.n_splits,
                                        split_type=args.split_type,
                                        split_sizes=args.split_sizes,
                                        num_folds=args.num_folds,
                                        seed=args.seed)
                evaluator1.run_external(args.dataset_test, name='test')
                evaluator.run_external(args.dataset_test, name='test_train_only')
            return evaluator.run_external(args.dataset_val, name='val')
        else:
            if args.separate_test_path is not None:
                args.dataset_test.compute_posenc_stats()
                evaluator.run_external(args.dataset_test, name='test')
            return evaluator.run_cross_validation()

    study = optuna.create_study(
        study_name="optuna-study",
        sampler=TPESampler(seed=args.seed),
        storage="sqlite:///%s/optuna.db" % args.save_dir,
        load_if_exists=True,
        direction='minimize' if args.dataset_type == 'regression' else 'maximize'
    )
    n_to_run = args.n_trials - len(study.trials)
    if n_to_run > 0:
        study.optimize(objective, n_trials=n_to_run)
