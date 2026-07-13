from __future__ import annotations

import copy

from vetta.settings.data import TreeDatasetSettings


# ``TreeDatasetConfig`` is a typed Pydantic settings model. The alias keeps the
# established import path and typed dict-conversion API (``from_dict``/``to_dict``/
# ``from_defaults``) available to callers.
TreeDatasetConfig = TreeDatasetSettings


_AUG_CONFIG_KEYS = {
    'pos_noise',
    'probs',
    'max_degrees',
    'min_zoom',
    'max_zoom',
    'max_delta',
    'margin_w',
}
_AUG_PROB_KEYS = {'zoom', 'rotate', 'translate', 'jitter'}
_AUG_PROB_OPTIONAL_KEYS = {'pos_jitter', 'rad_jitter'}
_CAND_STATS_KEYS = {
    'ncand_gen',
    'mdist_range',
    'uniform_rnd_bounds',
    'std_fctr_range',
    'offset_fctr_0',
    'offset_trunc_0',
    'n_cells',
    'k_noise',
    'gnoise_size',
    'gnoise_n_std',
    'n_timesteps',
    'zoom_base',
    'mdist',
}
_RECEIVE_MODES = {'tree'}
_TARGET_MODES = {'single', 'multi'}


def default_tree_dataset_config() -> dict:
    return TreeDatasetConfig().to_dict()


def copy_tree_dataset_config(config: dict | None = None) -> dict:
    if config is None:
        config = default_tree_dataset_config()
    return copy.deepcopy(config)


def normalise_tree_dataset_config(config: dict | None = None) -> dict:
    if config is None:
        config = default_tree_dataset_config()
    config = TreeDatasetConfig.from_dict(copy.deepcopy(config)).to_dict()
    validate_tree_dataset_config(config)
    return config


def validate_tree_dataset_config(config: dict) -> None:
    config = TreeDatasetConfig.from_dict(copy.deepcopy(config)).to_dict()
    _validate_nested_keys(config['aug_config'], _AUG_CONFIG_KEYS, 'aug_config')
    _validate_nested_keys(
        config['aug_config']['probs'],
        _AUG_PROB_KEYS,
        'aug_config.probs',
        optional_keys=_AUG_PROB_OPTIONAL_KEYS,
    )
    _validate_nested_keys(config['cand_stats'], _CAND_STATS_KEYS, 'cand_stats')
    _validate_pair(config['norm_domain'], 'norm_domain')
    _validate_pair(config['wrap_domain'], 'wrap_domain')
    _validate_pair(config['cand_time_range'], 'cand_time_range')
    _validate_non_negative(config['target_queue_length'], 'target_queue_length')
    _validate_non_negative(config['pad_size'], 'pad_size')
    _validate_non_negative(config['tgt_seg_size'], 'tgt_seg_size')
    _validate_non_negative(config['poll_time'], 'poll_time')
    _validate_non_negative(config['tpn_timeout'], 'tpn_timeout')
    _validate_non_negative(config['n_candidates'], 'n_candidates')
    _validate_non_negative(config['max_targets'], 'max_targets')
    _validate_optional_non_negative(config['seed'], 'seed')
    _validate_optional_non_negative(config['rank'], 'rank')
    _validate_optional_non_negative(config['start_port'], 'start_port')
    _validate_optional_non_negative(config['n_servers'], 'n_servers')
    _validate_optional_non_negative(config['static_cand_seed'], 'static_cand_seed')
    _validate_optional_non_negative(config['dynamic_cand_seed'], 'dynamic_cand_seed')
    _validate_optional_non_negative(config['order_seed'], 'order_seed')
    _validate_probability(config['init_prob'], 'init_prob')
    _validate_mode(config['mode'])
    _validate_choice(config['receive_mode'], _RECEIVE_MODES, 'receive_mode')
    _validate_choice(config['target_mode'], _TARGET_MODES, 'target_mode')
    for name, value in config['aug_config']['probs'].items():
        _validate_probability(value, f"aug_config.probs.{name}")
    if config['normalise'] and config['load_segmentation']:
        raise ValueError(
            "normalise and load_segmentation cannot be True at the same time."
        )
    if config['load_segmentation']:
        if config['aug_config']['probs']['translate'] > 0.0:
            raise ValueError(
                "augmentation with translation is not supported when loading segmentation."
            )
        if config['aug_config']['probs']['zoom'] > 0.0:
            raise ValueError(
                "augmentation with zoom is not supported when loading segmentation."
            )
    if config['pad_before_aug'] and not config['load_segmentation']:
        raise ValueError(
            "pad_before_aug is only supported when load_segmentation is True."
        )
    if config['add_semi'] and not config['add_splits']:
        raise ValueError("add_semi requires add_splits to be True.")
    if config['add_candidates']:
        raise ValueError("add_candidates is no longer supported.")


def resolve_tree_dataset_seed(config: dict, worker_id: int | None = None) -> int:
    if worker_id is not None and worker_id < 0:
        raise ValueError("worker_id must be non-negative.")
    seed = config['seed']
    rank = config['rank']
    _validate_optional_non_negative(seed, 'seed')
    _validate_optional_non_negative(rank, 'rank')

    if seed is None:
        seed = 1
        if rank is not None:
            seed *= 3 ** rank
        if worker_id is not None:
            seed *= 5 ** worker_id

    return int(seed)


def _validate_nested_keys(
    value: dict,
    expected_keys: set[str],
    name: str,
    *,
    optional_keys: set[str] | None = None,
) -> None:
    optional_keys = set() if optional_keys is None else optional_keys
    received = set(value.keys())
    unknown = sorted(received - expected_keys - optional_keys)
    missing = sorted(expected_keys - received)
    if unknown:
        raise ValueError(f"{name} received unknown config keys: {unknown}")
    if missing:
        raise ValueError(f"{name} missing required config keys: {missing}")


def _validate_pair(value: list[float], name: str) -> None:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values.")


def _validate_non_negative(value: int | float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_optional_non_negative(value: int | None, name: str) -> None:
    if value is not None:
        _validate_non_negative(value, name)


def _validate_probability(value: float, name: str) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0.")


def _validate_mode(mode: str) -> None:
    if mode not in {'train', 'test'}:
        raise ValueError("mode must be one of ['test', 'train'].")


def _validate_choice(value: str, choices: set[str], name: str) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}.")
