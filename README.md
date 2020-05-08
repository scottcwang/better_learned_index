# Constant-space tunable-accuracy insertion-robust learned range indices

### UW-Madison CS 839 001 Spring 2020

1. Generate a key space dataset:
```
python obtain_data.py <filename to save generated dataset> \
--generated_dataset_size <size of dataset> \
--generated_data_distribution <uniform | integer number of Gaussians in a mixture of Gaussians> \
--generated_dataset_std_coefficient <if Gaussians, scaling factor for each Gaussian's standard deviation>
```
Or specify the name of a dataset from [SOSD](https://github.com/learnedsystems/SOSD):
```
python obtain_data.py <name of an SOSD dataset>
```

2. Run an experiment:
```
python . \
<name of the experiment> \
<dataset filename> \
[--sort_data] [--shuffle_data] \
[--btree_node_capacity <capacity of each B-tree nodes; omit to skip B-tree experiment>] \
[--moe_index_n_experts <number of experts in the mixture of experts model. Default 8>] \
[--moe_index_cache_max_size <capacity of the training cache for the learned index. Default 81>] \
[--moe_experiment_lookup_pattern <random_uniform | sequential | most_recent. Default random_uniform>] \
[--moe_experiment_lookup_beta_a <a-parameter> --moe_experiment_lookup_beta_b <b-parameter of beta distribution defining the lookup pattern. Overrides --moe_experiment_lookup_pattern>] \
[--moe_experiment_insertion_lookup_interspersion <probability of lookup instead of insertion>] \
[--moe_experiment_graph_filename <filename to which to save training visualisation>]
```

3. Read the output:
```
[provided input options],
[capacity of B-tree],
[depth of B-tree],
[learned index space consumption],
[mean absolute error during training],
[mean absolute error during evaluation]
```