import argparse
import json
from pathlib import Path
from typing import Any
from typing import Sequence

import numpy as np
import torch

from vetta import Tree
from vetta import VesselTreeAutoencoder
from vetta import VesselTreeAutoencoderConfig
from vetta import make_full_encoder_config
from vetta import make_partial_encoder_config
from vetta import dice_score
from vetta import evaluate
from vetta import infer_tree
from vetta import train
from vetta import tree_to_segmentation
from vetta.data import TreeChunk


POS_OCTAVES = [1, 2]
POSDIMS = 2
TOPOLOGY_SIZE = 3
DEFAULT_SEG_SIZE = 250
DEFAULT_N_INTERPOLATE = 100


def _coerce_checkpoint_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _coerce_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_checkpoint_value(item) for item in value]
    if not isinstance(value, str):
        return value

    if value == "True":
        return True
    if value == "False":
        return False
    if value == "None":
        return None
    try:
        if any(marker in value for marker in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_checkpoint_config(
    checkpoint_path: Path,
    *,
    backwards_compatibility_paper: bool | None,
) -> dict[str, Any]:
    with (checkpoint_path / "config.json").open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    checkpoint_config = _coerce_checkpoint_value(raw_config)
    raw_model_config = checkpoint_config["model_config"]

    model_config = VesselTreeAutoencoder.default_config()
    model_config.update(
        {
            key: value
            for key, value in raw_model_config.items()
            if key in model_config
        }
    )
    # Existing proposed checkpoints predate this key. Default to the paper
    # compatibility path so their state dicts and inference behavior line up.
    if backwards_compatibility_paper is None:
        backwards_compatibility_paper = raw_model_config.get(
            "backwards_compatibility_paper",
            True,
        )
    model_config["backwards_compatibility_paper"] = backwards_compatibility_paper
    checkpoint_config["model_config"] = model_config
    return checkpoint_config


def load_checkpoint_model(
    checkpoint_path: Path,
    *,
    device: str,
    backwards_compatibility_paper: bool | None,
) -> tuple[VesselTreeAutoencoder, dict[str, Any]]:
    checkpoint_config = load_checkpoint_config(
        checkpoint_path,
        backwards_compatibility_paper=backwards_compatibility_paper,
    )
    model = VesselTreeAutoencoder(
        VesselTreeAutoencoderConfig.from_mapping(checkpoint_config["model_config"])
    )
    state_dict = torch.load(
        checkpoint_path / "model" / "vt_autoencoder",
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(state_dict)
    model = model.to(torch.device(device)).eval()
    return model, checkpoint_config


def iter_dataset_trees(
    data_root: Path,
    *,
    dataset: str,
    split: str,
    n_trees: int,
):
    loaded = 0
    split_dir = data_root / dataset / split
    for chunk_path in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        chunk = TreeChunk.load(str(chunk_path))
        for tree_idx, tree in enumerate(chunk.trees):
            yield chunk_path, tree_idx, tree
            loaded += 1
            if loaded >= n_trees:
                return
    if loaded < n_trees:
        raise RuntimeError(f"requested {n_trees} trees but only found {loaded} in {split_dir}")


def render_tree_segmentation(tree: Tree) -> np.ndarray:
    if tree.radius is None:
        raise ValueError("checkpoint evaluation requires tree.radius for segmentation rendering")
    return tree_to_segmentation(
        seg_size=DEFAULT_SEG_SIZE,
        n_interpolate=DEFAULT_N_INTERPOLATE,
        edges=tree.edges.astype(np.int64),
        edges_mask=tree.edges_mask.astype(np.float32),
        pos=tree.pos.astype(np.float32),
        radius=tree.radius.astype(np.float32),
    )


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    model, checkpoint_config = load_checkpoint_model(
        args.checkpoint,
        device=args.device,
        backwards_compatibility_paper=args.backwards_compatibility_paper,
    )
    if args.mode == "classic":
        checkpoint_config["model_config"]["use_vae"] = False
    elif args.mode == "variational":
        checkpoint_config["model_config"]["use_vae"] = True

    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with torch.no_grad():
        for chunk_path, tree_idx, tree in iter_dataset_trees(
            args.data_root,
            dataset=args.dataset,
            split=args.split,
            n_trees=args.n_trees,
        ):
            tree_id = f"{chunk_path.name}/tree_{tree_idx:05d}"
            output_tree = infer_tree(
                model,
                tree,
                checkpoint_config=checkpoint_config,
                device=args.device,
                source_chunk=str(chunk_path),
                source_tree_index=tree_idx,
                checkpoint_path=str(args.checkpoint),
                order_seed=args.order_seed,
            )
            pairs[tree_id] = (
                render_tree_segmentation(tree),
                render_tree_segmentation(output_tree),
            )

    result = evaluate(pairs)
    return {
        "config": {
            "checkpoint": str(args.checkpoint),
            "data_root": str(args.data_root),
            "dataset": args.dataset,
            "split": args.split,
            "n_trees": args.n_trees,
            "device": args.device,
            "mode": args.mode,
            "order_seed": args.order_seed,
            "backwards_compatibility_paper": args.backwards_compatibility_paper,
        },
        "metrics": result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="SSA_0.2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-trees", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("auto", "classic", "variational"), default="auto")
    parser.add_argument("--order-seed", type=int, default=7)
    parser.add_argument(
        "--backwards-compatibility-paper",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Force the legacy paper behaviors needed to reproduce released paper "
            "checkpoints exactly. Defaults to the checkpoint value, or True for "
            "legacy checkpoints that predate the flag."
        ),
    )
    return parser


def build_tiny_model(out_mode: str = "octaves") -> VesselTreeAutoencoder:
    config = VesselTreeAutoencoder.default_config()
    config.update(
        {
            "n_slots": 8,
            "z_dim": 6,
            # full-encoder raw feature width:
            # 2*(posdims*len(octaves)*2) + 2*topology_size
            "data_size": 2 * (POSDIMS * len(POS_OCTAVES) * 2) + 2 * TOPOLOGY_SIZE,
            "enc_size": 6,
            "mlp_hidden_dims": 16,
            "n_heads": 2,
            "vdim": 4,
            "dropout": 0.0,
            "n_encoder_layers": 1,
            "n_decoder_layers": 1,
            "dim_feedforward_transformer": 16,
            "pos_octaves": POS_OCTAVES,
            "posdims": POSDIMS,
            "topology_size": TOPOLOGY_SIZE,
            "out_mode": out_mode,
            "use_vae": False,
            "include_rad": False,
        }
    )
    model = VesselTreeAutoencoder(VesselTreeAutoencoderConfig.from_dict(config))
    model.eval()
    return model


def synthetic_tree() -> Tree:
    max_nodes, max_children, max_edges = 6, 3, 8
    # Give branch arrays headroom for any decoded topology (a single chain can be
    # up to max_nodes long, a star up to max_nodes branches).
    max_branches, max_branch_len = 6, 6
    pos = np.zeros((max_nodes, 2), dtype=np.float32)
    node_mask = np.zeros((max_nodes,), dtype=np.float32)
    depth = -np.ones((max_nodes,), dtype=np.int32)
    edges = -np.ones((max_edges, 2), dtype=np.int32)
    edges_mask = np.zeros((max_edges,), dtype=np.float32)
    topology = -np.ones((max_nodes, TOPOLOGY_SIZE), dtype=np.int32)
    children = -np.ones((max_nodes, max_children), dtype=np.int32)
    n_children = -np.ones((max_nodes,), dtype=np.int32)
    root_mask = np.zeros((max_nodes,), dtype=np.float32)
    parents = np.zeros((max_nodes, max_nodes), dtype=np.int32)
    branches = -np.ones((max_branches, max_branch_len), dtype=np.int32)
    branch_mask = np.zeros((max_branches,), dtype=np.float32)

    for i, (x, y) in enumerate([(10.0, 10.0), (20.0, 25.0), (35.0, 15.0)]):
        pos[i] = (x, y)
        node_mask[i] = 1.0
        depth[i] = i
    topology[0] = [0, 1, 0]
    topology[1] = [0, 1, 0]
    topology[2] = [1, 0, 0]
    n_children[0], n_children[1], n_children[2] = 1, 1, 0
    children[0, 0] = 1
    children[1, 0] = 2
    root_mask[0] = 1.0
    parents[1, 0] = 1
    parents[2, 1] = 1
    edges[0] = (0, 1)
    edges[1] = (1, 2)
    edges_mask[0] = 1.0
    edges_mask[1] = 1.0
    branches[0, :3] = [0, 1, 2]
    branch_mask[0] = 1.0

    return Tree(
        branch_mask=branch_mask,
        branches=branches,
        children=children,
        depth=depth,
        n_children=n_children,
        node_mask=node_mask,
        parents=parents,
        pos=pos,
        root_mask=root_mask,
        topology=topology,
        edges=edges,
        edges_mask=edges_mask,
        metadata={"id": 0},
        world_to_image=None,
    )


def test_model_reparameterize() -> None:
    config = VesselTreeAutoencoder.default_config()
    config.update(
        {
            "n_slots": 3,
            "z_dim": 5,
            "enc_size": 5,
            "mlp_hidden_dims": 16,
            "n_heads": 2,
            "vdim": 4,
            "dropout": 0.0,
            "n_encoder_layers": 1,
            "n_decoder_layers": 1,
            "dim_feedforward_transformer": 16,
            "pos_octaves": [1],
            "out_mode": "default",
        }
    )
    config_obj = VesselTreeAutoencoderConfig.from_dict(config)
    model = VesselTreeAutoencoder(config_obj)
    z_mu = torch.zeros((2, config["z_dim"]), dtype=torch.float32)
    z_logvar = torch.zeros((2, config["z_dim"]), dtype=torch.float32)
    sample = model.reparameterize(z_mu, z_logvar)
    assert tuple(sample.shape) == (2, config["z_dim"])


def test_public_encoder_config_builders() -> None:
    config = build_tiny_model(out_mode="default").settings
    full = make_full_encoder_config(config)
    partial = make_partial_encoder_config(config)

    assert full.pool_final is True
    assert full.apply_mlp_b is True
    assert full.output_dims == config.z_dim
    assert partial.pool_final is False
    assert partial.include_query is True
    assert partial.st_size == config.data_size + 1


def test_infer_tree_and_segmentation() -> None:
    model = build_tiny_model()
    tree = synthetic_tree()
    checkpoint_config = {
        "model_config": {
            "pos_octaves": POS_OCTAVES,
            "posdims": POSDIMS,
            "use_vae": False,
            "include_rad": False,
        }
    }
    with torch.no_grad():
        out = infer_tree(model, tree, checkpoint_config=checkpoint_config, device="cpu")

    assert out.pos.shape == tree.pos.shape
    assert out.edges.shape == tree.edges.shape
    assert int(out.node_mask.sum()) >= 1

    radius = np.full((out.pos.shape[0],), 0.02, dtype=np.float32)
    seg = tree_to_segmentation(
        seg_size=32,
        n_interpolate=20,
        edges=out.edges,
        edges_mask=out.edges_mask,
        pos=np.clip(out.pos / 50.0, 0.0, 1.0).astype(np.float32),
        radius=radius,
    )
    assert seg.shape == (32, 32, 3)
    assert seg.dtype == np.uint8


def test_evaluation_dice() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    other = np.zeros((16, 16), dtype=np.uint8)
    other[4:12, 4:12] = 255
    disjoint = np.zeros((16, 16), dtype=np.uint8)
    disjoint[0:2, 0:2] = 255

    assert dice_score(mask, other) == 1.0
    assert dice_score(mask, disjoint) == 0.0
    assert dice_score(np.zeros((4, 4)), np.zeros((4, 4))) == 1.0

    result = evaluate([(mask, other), (mask, disjoint)])
    assert result["dice"]["mean"] == 0.5
    assert set(result["dice"]) == {"mean", "p5", "p50", "p95"}


def test_training_runs() -> None:
    import math

    model = build_tiny_model(out_mode="default")
    trees = [synthetic_tree() for _ in range(3)]
    history = train(
        model,
        trees,
        n_steps=3,
        batch_size=2,
        matching_k=1,
        lr=1e-4,
        device="cpu",
    )
    assert len(history) == 3
    assert all(math.isfinite(value) for value in history)


def test_octaves_domain_wired() -> None:
    import math

    from vetta.inference import resolve_position_decode_domain

    # The octave-target domain defaults from the model config (the checkpoints'
    # (-3, 3)); octaves training must use it without raising KeyError('domain').
    model = build_tiny_model(out_mode="octaves")
    assert tuple(model.config["domain"]) == (-3.0, 3.0)
    trees = [synthetic_tree() for _ in range(3)]
    history = train(model, trees, n_steps=3, batch_size=2, matching_k=1, lr=1e-4, device="cpu")
    assert len(history) == 3 and all(math.isfinite(v) for v in history)

    # Inference must decode with the same domain: `domain` is authoritative and
    # takes precedence over the encoder's `wrap_domain`, and an explicit
    # infer_node_config override wins over the model config.
    full_config = {"domain": [-3.0, 3.0], "wrap_domain": [-0.5, 0.5]}
    assert resolve_position_decode_domain(model_config=full_config) == (-3.0, 3.0)
    assert resolve_position_decode_domain(model_config={"wrap_domain": [-0.5, 0.5]}) == (-0.5, 0.5)
    assert (
        resolve_position_decode_domain(
            infer_node_config={"wrap_domain": [-3.0, 3.0]},
            model_config={"domain": [-0.25, 0.25]},
        )
        == (-3.0, 3.0)
    )


def test_backwards_compatibility_paper_flag() -> None:
    # The umbrella flag defaults to the fixed behaviour and can be turned on to
    # reproduce the paper/legacy quirks. Under the legacy flag the edge encoder
    # gathers topology_b from indices_a, so topology_a == topology_b; the fixed
    # default gathers from indices_b, so they can differ.
    from vetta.model import EdgeBatch
    from vetta.model import VesselEdgesEncoder
    from vetta.model import VesselEdgesEncoderConfig

    def _gather(flag: bool):
        config = VesselEdgesEncoderConfig(posdims=2, backwards_compatibility_paper=flag)
        model = VesselEdgesEncoder(config)
        pos = torch.zeros((1, 3, 2), dtype=torch.float32)
        depth = torch.zeros((1, 3), dtype=torch.float32)
        topology = torch.tensor([[[0, 1, 0], [1, 0, 0], [0, 0, 1]]], dtype=torch.float32)
        edges = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.int64)
        edges_mask = torch.ones((1, 2), dtype=torch.float32)
        batch = EdgeBatch(pos=pos, depth=depth, topology=topology, edges=edges, edges_mask=edges_mask)
        return model._gather_edge_parts(batch)

    legacy = _gather(True)
    fixed = _gather(False)
    assert torch.equal(legacy.topology_b, legacy.topology_a)
    assert not torch.equal(fixed.topology_b, fixed.topology_a)


def run_smoke_tests() -> None:
    test_model_reparameterize()
    test_public_encoder_config_builders()
    test_infer_tree_and_segmentation()
    test_evaluation_dice()
    test_training_runs()
    test_octaves_domain_wired()
    test_backwards_compatibility_paper_flag()
    print("public smoke test passed")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        run_smoke_tests()
        return
    result = evaluate_checkpoint(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
