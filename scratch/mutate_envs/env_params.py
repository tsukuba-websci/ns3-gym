# bottleneck_bandwidthはMbps, bottleneck_delayはms

env_params = {
    'n_leaf': {
        'min_value': 1,
        'max_value': 10,
        'mutation_step': 1,
    },
    'error_rate': {
        'min_value': 0.0,
        'max_value': 0.8,
        'mutation_step': 0.1,
    },
    'bottleneck_bandwidth': {
        'min_value': 2,
        'max_value': 20,
        'mutation_step': 2,
    },
    'bottleneck_delay': {
        'min_value': 10,
        'max_value': 100,
        'mutation_step': 5,
    },
}