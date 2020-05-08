import argparse
import sys

import numpy as np

import obtain_data
import experiment
import BTree
import MoEInexactIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id')  # only used to identify standard output
    parser.add_argument('dataset_filename')
    parser.add_argument('--dataset_dtype_str')
    parser.add_argument('--dataset_md5')
    # Sort precedes shuffle
    parser.add_argument('--sort_data', action='store_true')
    parser.add_argument('--shuffle_data', action='store_true')
    parser.add_argument('--btree_node_capacity', type=int)
    parser.add_argument('--moe_index_units', type=int, default=20)
    parser.add_argument('--moe_index_n_experts', type=int, default=8)
    parser.add_argument('--moe_index_epochs', type=int, default=10)
    parser.add_argument('--moe_index_learning_rate', type=float, default=0.1)
    parser.add_argument('--moe_index_batch_size', type=int, default=10)
    parser.add_argument('--moe_index_decay', type=float, default=1e-6)
    parser.add_argument('--moe_index_cache_max_size', type=int, default=81)
    parser.add_argument('--moe_experiment_lookup_pattern',
                        default='random_uniform')
    parser.add_argument('--moe_experiment_lookup_beta_a', type=float)
    parser.add_argument('--moe_experiment_lookup_beta_b', type=float)
    parser.add_argument('--moe_experiment_insertion_lookup_interspersion',
                        type=float, default=0.5)
    parser.add_argument('--moe_experiment_show_graph', action='store_true')
    parser.add_argument('--moe_experiment_graph_filename')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if any((i is not None and i < 0) for i in [
        args.btree_node_capacity,
        args.moe_index_units,
        args.moe_index_n_experts,
        args.moe_index_epochs,
        args.moe_index_learning_rate,
        args.moe_index_batch_size,
        args.moe_index_decay,
        args.moe_index_cache_max_size,
        args.moe_experiment_lookup_beta_a,
        args.moe_experiment_lookup_beta_b,
        args.moe_experiment_insertion_lookup_interspersion
    ]):
        raise ValueError('Numeric option values must be nonnegative')

    data = obtain_data.decompress_array(
        args.dataset_filename, args.dataset_md5, args.dataset_dtype_str, verbose=args.verbose)

    if args.sort_data:
        data = np.sort(data)
    elif args.shuffle_data:
        np.random.default_rng().shuffle(data)

    size_if_full = None
    depth = None
    if args.btree_node_capacity is not None:
        btree_index_params = {
            'node_capacity': args.btree_node_capacity
        }
        btree_index = BTree.BTree(**btree_index_params)
        size_if_full, depth = experiment.run_exact_index_experiment(
            data, btree_index, args.verbose)
        if args.verbose:
            print(btree_index)

    moe_index_params = {
        'units': args.moe_index_units,
        'n_experts': args.moe_index_n_experts,
        'epochs': args.moe_index_epochs,
        'learning_rate': args.moe_index_learning_rate,
        'batch_size': args.moe_index_batch_size,
        'decay': args.moe_index_decay,
        'cache_max_size': args.moe_index_cache_max_size
    }
    moe_index = MoEInexactIndex.MoEInexactIndex(**moe_index_params)

    moe_experiment_params = {
        'lookup_pattern': args.moe_experiment_lookup_pattern,
        'lookup_beta_a': args.moe_experiment_lookup_beta_a,
        'lookup_beta_b': args.moe_experiment_lookup_beta_b,
        'insertion_lookup_interspersion': args.moe_experiment_insertion_lookup_interspersion,
        'show_graph': args.moe_experiment_show_graph,
        'graph_filename': args.moe_experiment_graph_filename
    }
    training_loss, evaluation_loss, _, _ = experiment.run_inexact_index_experiment(
        data, moe_index, **moe_experiment_params, verbose=args.verbose)

    moe_count_params = moe_index.count_params()

    print(','.join(map(str, [
        '"' + ' '.join(sys.argv[1:]) + '"',
        (size_if_full if not None else '""'),
        (depth if not None else '""'),
        moe_count_params,
        training_loss, evaluation_loss
    ])))


if __name__ == "__main__":
    main()
