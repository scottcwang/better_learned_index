import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.lines as lines


def run_exact_index_experiment(data, index, verbose=False):
    """
    Adds every key in the dataset to the index.

    Arguments:
        data {numpy.float} -- Vector of key values in insertion order.
        index {Index} -- Index to train.

    Returns:
        [int] -- Size of the index if it were full.
        [int] -- Maximum depth of the index.
    """
    for key in data:
        index.insert(key)
    return index.size_if_full(), index.depth()


def run_inexact_index_experiment(data, index, lookup_pattern=None, lookup_beta_a=None, lookup_beta_b=None, insertion_lookup_interspersion='random_uniform', verbose=False, show_graph=False, graph_filename=None):
    """
    Runs an experiment: trains the index on possibly interspersed sequence of insertions and lookups of keys in a dataset; evaluates the index by looking up every key in the dataset; returns the squared loss between the looked-up positions and their true positions during training and evaluation; optionally, displays a plot of the true position and the looked-up position for each key.

    Arguments:
        data {numpy.float} -- Vector of key values in insertion order.
        index {Index} -- Index to train.
        lookup_pattern {string} -- Lookup pattern during training: 'most_recent' to always look up the most recently inserted key, 'sequential' to look up the earliest inserted key not already looked up, or 'random_uniform' to look up a random key that is already inserted. Ignored if both lookup_beta_a and lookup_beta_b are specified.
        lookup_beta_a {float} -- the a parameter of the Beta distribution governing the lookup pattern distribution in the sorted data. Will supersede lookup_pattern if and only if lookup_beta_b is also specified.
        lookup_beta_b {float} -- the b parameter of the Beta distribution governing the lookup pattern distribution in the sorted data. Will supersede lookup_pattern if and only if lookup_beta_a is also specified.
        insertion_lookup_interspersion {float} in [0, 1) -- Probability during training that a lookup occurs instead of an insertion. If lookup_beta_a and lookup_beta_b are specified, a datum may be drawn that is not already inserted in the index, in which case the lookup will be skipped.
        verbose {bool} -- Whether to print the key and position for each trained and looked-up key (default: {False}).
        show_graph {bool} -- Whether to display a plot of the true position and the looked-up position against the value of each key (default: {True}).
        graph_filename {string} -- If not None, the filename of the graph image to save (default: {None}).

    Returns:
        [int] -- Mean training squared loss between looked-up positions and true positions, averaged over the number of training operations
        [int] -- Mean evaluation squared loss between looked-up positions and true positions, averaged over the number of data
        [numpy.int_] -- Predicted position for each key in {data}
        [numpy.int_] -- Number of times each key in {data} was trained
    """

    # Training
    position_insert = 0
    position_lookup = 0
    training_loss = 0

    argsorted_data = np.argsort(data)

    train_count = np.zeros_like(data)
    while True:
        is_insert = np.random.random_sample() > insertion_lookup_interspersion
        if is_insert:
            # insertion
            if position_insert >= len(data):
                break
            position = position_insert
            position_insert += 1
        else:
            # lookup
            if position_insert == 0:
                continue
            if lookup_beta_a is not None and lookup_beta_b is not None:
                rank = int(np.random.beta(
                    lookup_beta_a, lookup_beta_b)*len(data))
                if rank < len(data):
                    position = argsorted_data[rank]
                else:
                    continue
            elif lookup_pattern == 'most_recent':
                position = position_insert - 1
            elif lookup_pattern == 'sequential':
                position = position_lookup
                position_lookup += 1
            elif lookup_pattern == 'random_uniform':
                position = np.random.randint(0, position_insert)
            else:
                continue
            if position >= len(data) or position >= position_insert:
                continue
        true_position = np.nonzero(np.argsort(
            data[:position_insert]) == position)[0].item()
        key = data[position].item()
        predicted_position = index.lookup(key, verbose)
        loss = abs(predicted_position - true_position)
        training_loss += loss
        index.train(key, true_position, is_insert, verbose)
        if verbose:
            print('loss: ' + str(loss))
        train_count[position] += 1

    # Evaluation
    evaluation_predicted_position = np.empty_like(data, dtype=np.int_)
    evaluation_loss = 0
    for true_position, lookup_index in enumerate(argsorted_data):
        evaluation_predicted_position[lookup_index] = index.lookup(
            data[lookup_index], verbose)
        loss = abs(
            evaluation_predicted_position[lookup_index] - true_position)
        evaluation_loss += loss
        if verbose:
            print('loss: ' + str(loss))

    # Graphing
    if show_graph or graph_filename is not None:
        true_scatter = plt.scatter(
            data[argsorted_data],
            range(len(data)),
            s=train_count[argsorted_data] * 10,
            alpha=0.5,
            label='True position'
        )
        predicted_scatter = plt.scatter(
            data[argsorted_data],
            evaluation_predicted_position[argsorted_data],
            marker='x',
            edgecolors=None,
            label='Predicted position'
        )
        plt.xlabel('Key')
        plt.ylabel('Position')
        legend_lines, legend_labels = true_scatter.legend_elements(
            prop="sizes",
            func=lambda s: s / 10,
            color=true_scatter.get_facecolor().flatten(),
            num=5)
        plt.legend(
            [
                *legend_lines,
                lines.Line2D(
                    [], [],
                    color=predicted_scatter.get_facecolor().flatten(),
                    marker='x',
                    linestyle='None')
            ],
            [*legend_labels, 'Predicted position'],
            title='True position;\ntraining access frequency')
        if graph_filename is not None:
            plt.savefig(graph_filename)
        if show_graph:
            plt.show()

    return (
        training_loss / sum(train_count),
        evaluation_loss / len(data),
        evaluation_predicted_position,
        train_count)
