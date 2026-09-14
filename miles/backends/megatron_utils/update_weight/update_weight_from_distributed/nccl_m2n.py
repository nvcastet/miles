"""Hybrid nccl-rl reshard plus broadcast weight updates."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import socket
import time
import uuid
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import requests
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle

from miles.backends.megatron_utils.sglang import per_block_cast_to_fp8
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group, init_process_group
from miles.utils.fp8_kernel import blockwise_cast_to_fp8_triton

from ..common import end_weight_update
from .broadcast import (
    UpdateWeightFromDistributed,
    connect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_FP8_BLOCK_SIZE = (128, 128)
_FP8_MANIFEST_QUANTIZATION = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": list(_FP8_BLOCK_SIZE),
    "weight_dtype": "float8_e4m3fn",
    "scale_dtype": "float32",
    "scale_format": "canonical",
}
_DENSE_RE = re.compile(r"module\.module\.decoder\.layers\.(\d+)\.mlp\.linear_fc([12])\.weight$")
_EXPERT_RE = re.compile(r"module\.module\.decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc([12])\.weight(\d+)$")


def _new_m2n_group_name() -> str:
    return f"miles-m2n-{uuid.uuid4().hex}"


def _validated_engine_gpu_counts(
    args: Namespace,
    engine_count: int,
    engine_gpu_counts: Sequence[int] | None,
) -> list[int]:
    expected_size = int(args.rollout_num_gpus_per_engine)
    total_gpus = int(args.rollout_num_gpus)
    if expected_size <= 0 or total_gpus <= 0 or total_gpus % expected_size:
        raise ValueError(
            "NCCL M2N requires positive rollout GPU counts with the total "
            f"divisible by the per-engine count, got total={total_gpus}, per_engine={expected_size}"
        )
    expected_engines = total_gpus // expected_size
    if engine_count != expected_engines:
        raise ValueError(
            f"NCCL M2N expected {expected_engines} rollout engine handles for "
            f"{total_gpus} GPUs at {expected_size} GPUs per engine, got {engine_count}"
        )

    counts = [expected_size] * engine_count if engine_gpu_counts is None else list(engine_gpu_counts)
    if len(counts) != engine_count:
        raise ValueError(
            "NCCL M2N requires one GPU count per rollout engine handle, "
            f"got {engine_count} handles and {len(counts)} counts"
        )
    if counts != [expected_size] * engine_count:
        raise ValueError(
            f"NCCL M2N requires homogeneous {expected_size}-GPU rollout engines, got {counts}"
        )
    return counts


def _is_unreachable_engine_error(error: Exception) -> bool:
    cause: BaseException = error
    as_instanceof_cause = getattr(error, "as_instanceof_cause", None)
    if callable(as_instanceof_cause):
        try:
            cause = as_instanceof_cause()
        except Exception:
            pass
    return isinstance(
        cause,
        (ray.exceptions.RayActorError, requests.exceptions.ConnectionError),
    )


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if getattr(torch, name, None) is not dtype:
        raise ValueError(f"Unsupported nccl-rl dtype {dtype}")
    return name


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported nccl-rl dtype {name!r}")
    return dtype


def _tensor_bytes(shape: Sequence[int], dtype_name: str) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel * torch.empty((), dtype=_dtype_from_name(dtype_name)).element_size()


def _fp8_manifest_quantization(
    quantization_config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if quantization_config is None:
        return None
    if not isinstance(quantization_config, Mapping):
        raise ValueError("NCCL M2N quantization_config must be a mapping")
    try:
        json.dumps(quantization_config, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("NCCL M2N quantization_config must be JSON serializable") from exc
    if (
        quantization_config.get("quant_method") != "fp8"
        or quantization_config.get("fmt", "e4m3") != "e4m3"
        or quantization_config.get("activation_scheme") != "dynamic"
        or list(quantization_config.get("weight_block_size") or ()) != list(_FP8_BLOCK_SIZE)
    ):
        raise ValueError(
            "NCCL M2N supports only block FP8 rollout weights with "
            "quant_method='fp8', fmt='e4m3', activation_scheme='dynamic', "
            "weight_block_size=[128, 128]"
        )
    # Checkpoint scale_fmt does not determine the M2N wire format: source
    # weights are requantized and scales are transferred as canonical FP32.
    return dict(_FP8_MANIFEST_QUANTIZATION)


def _fp8_scale_shape(
    weight_shape: Sequence[int],
    description: str,
) -> list[int]:
    shape = list(weight_shape)
    if len(shape) < 2 or any(dim % block for dim, block in zip(shape[-2:], _FP8_BLOCK_SIZE, strict=True)):
        raise ValueError(f"{description} shape {shape} is not aligned to FP8 blocks " f"{list(_FP8_BLOCK_SIZE)}")
    shape[-2] //= _FP8_BLOCK_SIZE[0]
    shape[-1] //= _FP8_BLOCK_SIZE[1]
    return shape


def _quantize_canonical_block_fp8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.dtype != torch.bfloat16 or weight.dim() != 3:
        raise ValueError(
            "NCCL M2N FP8 expert sources must be 3-D BF16 tensors, "
            f"got shape={tuple(weight.shape)} dtype={weight.dtype}"
        )
    scale_shape = _fp8_scale_shape(weight.shape, "FP8 expert source")
    flat = weight.contiguous().view(-1, weight.shape[-1])
    if per_block_cast_to_fp8 is not None:
        qweight, scale = per_block_cast_to_fp8(flat)
    else:
        qweight, scale = blockwise_cast_to_fp8_triton(flat, list(_FP8_BLOCK_SIZE))
    qweight = qweight.view_as(weight).contiguous()
    scale = scale.view(scale_shape).to(torch.float32).contiguous()
    if qweight.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"Canonical FP8 quantizer returned unexpected dtype {qweight.dtype}")
    return qweight, scale


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _one_owner(candidates: list[dict[str, Any]], description: str) -> int:
    if len(candidates) != 1:
        ranks = sorted(item["world_rank"] for item in candidates)
        raise ValueError(f"NCCL M2N requires exactly one canonical {description}; " f"found world ranks {ranks}")
    return candidates[0]["world_rank"]


def _build_rank_layout(
    topologies: Sequence[dict[str, Any]],
    *,
    need_dense: bool,
    need_expert: bool,
) -> dict[str, Any]:
    """Select canonical dense and expert owners independently for every PP stage."""

    if not topologies:
        raise ValueError("Cannot construct NCCL M2N rank layout without trainer ranks")
    if len({item["world_rank"] for item in topologies}) != len(topologies):
        raise ValueError("Trainer topology contains duplicate world ranks")

    sizes: dict[str, int] = {}
    for field in ("pp", "tp", "cp", "dense_dp", "ep", "etp", "expert_dp", "independent_dp"):
        values = {int(item[f"{field}_size"]) for item in topologies}
        if len(values) != 1:
            raise ValueError(f"Inconsistent trainer {field} sizes: {sorted(values)}")
        sizes[field] = values.pop()
        for item in topologies:
            rank = int(item[f"{field}_rank"])
            if not 0 <= rank < sizes[field]:
                raise ValueError(
                    f"Invalid trainer {field} rank {rank} for size {sizes[field]} "
                    f"on world rank {item['world_rank']}"
                )
    if sizes["etp"] != 1:
        raise ValueError(f"NCCL M2N requires trainer ETP=1, got {sizes['etp']}")

    coordinate_views = {
        "dense": ("pp", "tp", "cp", "dense_dp", "independent_dp"),
        "expert": ("pp", "ep", "etp", "expert_dp", "independent_dp"),
    }
    for family, fields in coordinate_views.items():
        expected = 1
        for field in fields:
            expected *= sizes[field]
        coordinates = {
            tuple(int(item[f"{field}_rank"]) for field in fields)
            for item in topologies
        }
        if len(topologies) != expected or len(coordinates) != expected:
            raise ValueError(
                f"NCCL M2N {family} topology describes {expected} ranks across "
                f"{fields}, but received {len(topologies)} trainer ranks with "
                f"{len(coordinates)} unique coordinates"
            )

    canonical = [item for item in topologies if item["independent_dp_rank"] == 0]
    dense_world_by_pp: dict[str, list[int]] = {}
    expert_world_by_pp: dict[str, list[int]] = {}
    for pp_rank in range(sizes["pp"]):
        if need_dense:
            dense_world_by_pp[str(pp_rank)] = [
                _one_owner(
                    [
                        item
                        for item in canonical
                        if item["pp_rank"] == pp_rank
                        and item["dense_dp_rank"] == 0
                        and item["cp_rank"] == 0
                        and item["tp_rank"] == tp_rank
                    ],
                    f"dense owner for PP={pp_rank}, TP={tp_rank}",
                )
                for tp_rank in range(sizes["tp"])
            ]
        if need_expert:
            expert_world_by_pp[str(pp_rank)] = [
                _one_owner(
                    [
                        item
                        for item in canonical
                        if item["pp_rank"] == pp_rank
                        and item["expert_dp_rank"] == 0
                        and item["ep_rank"] == ep_rank
                        and item["etp_rank"] == etp_rank
                    ],
                    f"expert owner for PP={pp_rank}, EP={ep_rank}, ETP={etp_rank}",
                )
                for ep_rank in range(sizes["ep"])
                for etp_rank in range(sizes["etp"])
            ]

    ordered_world = sorted(
        {rank for meshes in (dense_world_by_pp, expert_world_by_pp) for mesh in meshes.values() for rank in mesh}
    )
    if not ordered_world:
        raise ValueError("No trainer rank owns an NCCL M2N-routable tensor")
    world_to_comm = {world_rank: comm_rank for comm_rank, world_rank in enumerate(ordered_world)}
    dense_meshes = {pp: [world_to_comm[rank] for rank in ranks] for pp, ranks in dense_world_by_pp.items()}
    expert_meshes = {pp: [world_to_comm[rank] for rank in ranks] for pp, ranks in expert_world_by_pp.items()}
    for family, meshes in (("dense", dense_meshes), ("expert", expert_meshes)):
        for pp_rank, mesh in meshes.items():
            if mesh != list(range(mesh[0], mesh[0] + len(mesh))):
                raise ValueError(
                    f"NCCL M2N {family} source mesh for PP={pp_rank} must be "
                    f"a contiguous communicator interval, got {mesh}"
                )

    return {
        "source_world_ranks": ordered_world,
        "trainer_world_to_comm_rank": {str(key): value for key, value in world_to_comm.items()},
        "dense_world_by_pp": dense_world_by_pp,
        "expert_world_by_pp": expert_world_by_pp,
        "dense_source_mesh_by_pp": dense_meshes,
        "expert_source_mesh_by_pp": expert_meshes,
        "sizes": sizes,
    }


def _local_source_spec(
    name: str,
    tensor: torch.Tensor,
) -> dict[str, Any] | None:
    dense_match = _DENSE_RE.fullmatch(name)
    expert_match = _EXPERT_RE.fullmatch(name)
    if dense_match:
        layer, projection = dense_match.groups()
        family = "dense"
        expert_id = None
    elif expert_match:
        layer, projection, expert_id = expert_match.groups()
        family = "routed_expert"
        expert_id = int(expert_id)
    else:
        return None
    partition_dim = int(getattr(tensor, "partition_dim", -1))
    partition_stride = int(getattr(tensor, "partition_stride", 1))
    if family == "dense" and projection == "1":
        # Fused SwiGLU FC1 stores each local TP shard as [gate, up]. Older
        # Megatron/TE versions do not consistently expose partition_stride=2.
        partition_dim = 0
        partition_stride = 2
    elif family == "dense" and projection == "2":
        # Match the existing residual gather workaround for TE row-parallel
        # projections that incorrectly report partition_dim=0.
        partition_dim = 1
        partition_stride = 1
    return {
        "name": name,
        "family": family,
        "layer": int(layer),
        "projection": f"fc{projection}",
        "expert_id": expert_id,
        "dtype": _dtype_name(tensor.dtype),
        "local_shape": list(tensor.shape),
        "partition_dim": partition_dim,
        "partition_stride": partition_stride,
    }


def _entry(
    *,
    name: str,
    family: str,
    pp_rank: int,
    source_names_by_rank: Mapping[int, Sequence[str]],
    dtype: str,
    global_shape: Sequence[int],
    source_mesh: Sequence[int],
    source_shard_dim: int,
    source_local_shape: Sequence[int],
    destination_mesh: Sequence[Sequence[int]],
    destination_shard_dim: int,
    destination_local_shape: Sequence[int],
    source_recipe: str,
    destination_recipe: str,
    destination_parameter: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "pp_rank": pp_rank,
        "dtype": dtype,
        "global_shape": list(global_shape),
        "source": {
            "mesh": [list(source_mesh)],
            "placements": [
                {"type": "replicate"},
                {"type": "shard", "dim": source_shard_dim},
            ],
            "local_shape": list(source_local_shape),
            "names_by_rank": {str(rank): list(names) for rank, names in sorted(source_names_by_rank.items())},
            "recipe": source_recipe,
        },
        "destination": {
            "mesh": [list(row) for row in destination_mesh],
            "placements": [
                {"type": "replicate"},
                {"type": "shard", "dim": destination_shard_dim},
            ],
            "local_shape": list(destination_local_shape),
            "parameter": destination_parameter,
            "recipe": destination_recipe,
        },
    }


def _fp8_expert_pair(weight: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if weight["family"] != "routed_expert" or weight["dtype"] != "bfloat16":
        raise ValueError(
            "NCCL M2N FP8 routes require routed-expert BF16 sources, "
            f"got family={weight['family']!r} dtype={weight['dtype']!r}"
        )
    pair_id = weight["name"]
    if not pair_id.endswith(".weight"):
        raise ValueError(f"Invalid FP8 weight name {pair_id!r}")

    source = weight["source"]
    destination = weight["destination"]
    scale = _entry(
        name=f"{pair_id.removesuffix('.weight')}.weight_scale_inv",
        family=weight["family"],
        pp_rank=weight["pp_rank"],
        source_names_by_rank={int(rank): names for rank, names in source["names_by_rank"].items()},
        dtype="float32",
        global_shape=_fp8_scale_shape(weight["global_shape"], f"{pair_id} global"),
        source_mesh=source["mesh"][0],
        source_shard_dim=source["placements"][1]["dim"],
        source_local_shape=_fp8_scale_shape(source["local_shape"], f"{pair_id} source"),
        destination_mesh=destination["mesh"],
        destination_shard_dim=destination["placements"][1]["dim"],
        destination_local_shape=_fp8_scale_shape(destination["local_shape"], f"{pair_id} destination"),
        source_recipe=f"{source['recipe']}_scale",
        destination_recipe=f"{destination['recipe']}_scale",
        destination_parameter=f"{destination['parameter']}_scale_inv",
    )
    weight["dtype"] = "float8_e4m3fn"
    weight["pair_id"] = pair_id
    weight["tensor_role"] = "weight"
    scale["pair_id"] = pair_id
    scale["tensor_role"] = "scale"
    return weight, scale


def _local_shape(
    global_shape: Sequence[int],
    shard_dim: int,
    shard_count: int,
) -> list[int]:
    shape = list(global_shape)
    if shape[shard_dim] % shard_count:
        raise ValueError(f"Shape {shape} cannot shard dimension {shard_dim} over {shard_count} ranks")
    shape[shard_dim] //= shard_count
    return shape


def _expert_destination_shard_dim(
    projection: str,
    destination_tp_size: int,
    destination_ep_size: int | None,
) -> int:
    """Return the SGLang routed-expert shard dimension for an engine.

    SGLang consumes the engine's tensor-parallel ranks either entirely as
    expert parallelism (EP=TP, MoE-TP=1) or entirely as MoE tensor
    parallelism (EP=1, MoE-TP=TP). The former shards the expert axis. The
    latter replicates experts and shards each expert's intermediate axis.

    A missing EP size preserves the original manifest-builder behavior for
    callers that do not have rollout topology available.
    """

    ep_size = destination_tp_size if destination_ep_size is None else destination_ep_size
    if ep_size <= 0 or destination_tp_size % ep_size:
        raise ValueError(
            "NCCL M2N requires rollout TP to be divisible by rollout EP; "
            f"got TP={destination_tp_size}, EP={ep_size}"
        )
    if ep_size == destination_tp_size:
        return 0
    if ep_size == 1:
        return 1 if projection == "fc1" else 2
    raise ValueError(
        "NCCL M2N routed-expert transfer currently supports rollout EP=1 "
        "or EP=TP; "
        f"got TP={destination_tp_size}, EP={ep_size}"
    )


def _entry_source_names(entry: Mapping[str, Any]) -> set[str]:
    return {name for names in entry["source"]["names_by_rank"].values() for name in names}


def _fp8_expert_module(pair_id: str) -> tuple[str, str]:
    for component in ("gate", "up", "down"):
        suffix = f".{component}_proj.weight"
        if pair_id.endswith(suffix):
            return pair_id.removesuffix(suffix), component
    raise ValueError(f"Invalid NCCL M2N FP8 expert pair ID {pair_id!r}")


def _retain_complete_fp8_modules(
    entries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    modules: dict[str, set[tuple[str, str]]] = {}
    entry_modules: list[str] = []
    for entry in entries:
        module, component = _fp8_expert_module(entry["pair_id"])
        modules.setdefault(module, set()).add((component, entry["tensor_role"]))
        entry_modules.append(module)
    complete = {
        module
        for module, members in modules.items()
        if members == {(component, role) for component in ("gate", "up", "down") for role in ("weight", "scale")}
    }
    return [entry for entry, module in zip(entries, entry_modules, strict=True) if module in complete]


def _validate_fp8_pairs(entries: Sequence[Mapping[str, Any]]) -> None:
    pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    modules: dict[str, set[str]] = {}
    for entry in entries:
        pair_id = entry.get("pair_id")
        role = entry.get("tensor_role")
        if (
            entry.get("family") != "routed_expert"
            or not isinstance(pair_id, str)
            or role not in ("weight", "scale")
            or role in pairs.setdefault(pair_id, {})
        ):
            raise ValueError(f"Invalid NCCL M2N FP8 pair metadata for {entry['name']}")
        pairs[pair_id][role] = entry
        module, component = _fp8_expert_module(pair_id)
        modules.setdefault(module, set()).add(component)

    for pair_id, pair in pairs.items():
        if set(pair) != {"weight", "scale"}:
            raise ValueError(f"NCCL M2N FP8 pair {pair_id!r} is incomplete: {sorted(pair)}")
        weight, scale = pair["weight"], pair["scale"]
        expected_scale_name = f"{pair_id.removesuffix('.weight')}.weight_scale_inv"
        if (
            weight["name"] != pair_id
            or scale["name"] != expected_scale_name
            or weight["dtype"] != "float8_e4m3fn"
            or scale["dtype"] != "float32"
            or weight["pp_rank"] != scale["pp_rank"]
            or _entry_source_names(weight) != _entry_source_names(scale)
            or weight["source"]["names_by_rank"] != scale["source"]["names_by_rank"]
            or weight["source"]["mesh"] != scale["source"]["mesh"]
            or weight["source"]["placements"] != scale["source"]["placements"]
            or weight["destination"]["mesh"] != scale["destination"]["mesh"]
            or weight["destination"]["placements"] != scale["destination"]["placements"]
            or scale["global_shape"] != _fp8_scale_shape(weight["global_shape"], f"{pair_id} global")
            or scale["source"]["local_shape"] != _fp8_scale_shape(weight["source"]["local_shape"], f"{pair_id} source")
            or scale["destination"]["local_shape"]
            != _fp8_scale_shape(weight["destination"]["local_shape"], f"{pair_id} destination")
            or scale["source"]["recipe"] != f"{weight['source']['recipe']}_scale"
            or scale["destination"]["recipe"] != f"{weight['destination']['recipe']}_scale"
            or scale["destination"]["parameter"] != f"{weight['destination']['parameter']}_scale_inv"
        ):
            raise ValueError(f"Inconsistent NCCL M2N FP8 pair {pair_id!r}")
    for module, components in modules.items():
        if components != {"gate", "up", "down"}:
            raise ValueError(f"NCCL M2N FP8 expert module {module!r} is incomplete: " f"{sorted(components)}")


def _same_specs(specs: Sequence[dict[str, Any]], fields: Sequence[str], description: str) -> None:
    for field in fields:
        values = {json.dumps(spec[field], sort_keys=True) for spec in specs}
        if len(values) != 1:
            raise ValueError(f"Inconsistent {description} {field}: {values}")


def _build_manifest(
    trainer_payloads: Sequence[dict[str, Any]],
    engine_gpu_counts: Sequence[int],
    quantization_config: Mapping[str, Any] | None = None,
    destination_ep_size: int | None = None,
) -> dict[str, Any]:
    quantization = _fp8_manifest_quantization(quantization_config)
    if not engine_gpu_counts or any(count <= 0 for count in engine_gpu_counts):
        raise ValueError(f"nccl-rl requires positive rollout engine GPU counts, got {engine_gpu_counts}")
    if len(set(engine_gpu_counts)) != 1:
        raise ValueError("nccl-rl requires homogeneous rollout engine parallelism, " f"got {list(engine_gpu_counts)}")
    destination_count = sum(engine_gpu_counts)

    all_specs = [spec for payload in trainer_payloads for spec in payload["specs"]]
    need_dense = quantization is None and any(spec["family"] == "dense" for spec in all_specs)
    need_expert = any(spec["family"] == "routed_expert" for spec in all_specs)
    destination_tp_size = engine_gpu_counts[0]
    if need_expert:
        # Validate the rollout expert layout before building rank ownership.
        # Projection-specific calls below select the corresponding shard axis.
        _expert_destination_shard_dim("fc1", destination_tp_size, destination_ep_size)
    layout = _build_rank_layout(
        [payload["topology"] for payload in trainer_payloads],
        need_dense=need_dense,
        need_expert=need_expert,
    )
    payload_by_world = {payload["topology"]["world_rank"]: payload for payload in trainer_payloads}
    world_to_comm = {int(world): comm for world, comm in layout["trainer_world_to_comm_rank"].items()}
    source_count = len(layout["source_world_ranks"])
    destination_mesh: list[list[int]] = []
    cursor = source_count
    for count in engine_gpu_counts:
        destination_mesh.append(list(range(cursor, cursor + count)))
        cursor += count
    entries: list[dict[str, Any]] = []
    layer_owners: dict[int, int] = {}

    def record_layer_owner(layer: int, pp_rank: int) -> None:
        previous = layer_owners.setdefault(layer, pp_rank)
        if previous != pp_rank:
            raise ValueError(f"Global decoder layer {layer} is reported by PP stages " f"{previous} and {pp_rank}")

    for pp_rank_text, dense_world in layout["dense_world_by_pp"].items():
        pp_rank = int(pp_rank_text)
        source_mesh = layout["dense_source_mesh_by_pp"][pp_rank_text]
        specs_by_world = {
            world: {
                (spec["layer"], spec["projection"]): spec
                for spec in payload_by_world[world]["specs"]
                if spec["family"] == "dense"
            }
            for world in dense_world
        }
        dense_keys = set.intersection(*(set(specs) for specs in specs_by_world.values()))
        if any(set(specs) != dense_keys for specs in specs_by_world.values()):
            raise ValueError(f"Dense FFN source specs differ across selected TP ranks for PP={pp_rank}")
        for layer, projection in sorted(dense_keys):
            record_layer_owner(layer, pp_rank)
            specs = [specs_by_world[world][(layer, projection)] for world in dense_world]
            _same_specs(
                specs,
                (
                    "name",
                    "dtype",
                    "local_shape",
                    "partition_dim",
                    "partition_stride",
                ),
                f"dense layer {layer} {projection}",
            )
            spec = specs[0]
            rows, columns = spec["local_shape"]
            source_names_by_rank = {
                world_to_comm[world]: [specs_by_world[world][(layer, projection)]["name"]] for world in dense_world
            }
            if projection == "fc1":
                if rows % 2 or spec["partition_dim"] != 0 or spec["partition_stride"] != 2:
                    raise ValueError(
                        f"{spec['name']} is not a supported fused gate/up TP shard: "
                        f"shape={spec['local_shape']}, partition_dim={spec['partition_dim']}, "
                        f"partition_stride={spec['partition_stride']}"
                    )
                local_rows = rows // 2
                global_shape = [local_rows * len(dense_world), columns]
                if global_shape[0] % destination_tp_size:
                    raise ValueError(f"{spec['name']} cannot shard evenly over rollout " f"TP={destination_tp_size}")
                for component, index in (("gate", 0), ("up", 1)):
                    entries.append(
                        _entry(
                            name=f"model.layers.{layer}.mlp.{component}_proj.weight",
                            family="dense",
                            pp_rank=pp_rank,
                            source_names_by_rank=source_names_by_rank,
                            dtype=spec["dtype"],
                            global_shape=global_shape,
                            source_mesh=source_mesh,
                            source_shard_dim=0,
                            source_local_shape=[local_rows, columns],
                            destination_mesh=destination_mesh,
                            destination_shard_dim=0,
                            destination_local_shape=_local_shape(global_shape, 0, destination_tp_size),
                            source_recipe=f"dense_fc1_{index}",
                            destination_recipe=f"dense_{component}",
                            destination_parameter=f"model.layers.{layer}.mlp.gate_up_proj.weight",
                        )
                    )
            else:
                if spec["partition_stride"] != 1 or spec["partition_dim"] != 1:
                    raise ValueError(
                        f"{spec['name']} is not a supported row-parallel down projection: "
                        f"partition_dim={spec['partition_dim']}, partition_stride={spec['partition_stride']}"
                    )
                global_shape = [rows, columns * len(dense_world)]
                if global_shape[1] % destination_tp_size:
                    raise ValueError(f"{spec['name']} cannot shard evenly over rollout " f"TP={destination_tp_size}")
                entries.append(
                    _entry(
                        name=f"model.layers.{layer}.mlp.down_proj.weight",
                        family="dense",
                        pp_rank=pp_rank,
                        source_names_by_rank=source_names_by_rank,
                        dtype=spec["dtype"],
                        global_shape=global_shape,
                        source_mesh=source_mesh,
                        source_shard_dim=1,
                        source_local_shape=[rows, columns],
                        destination_mesh=destination_mesh,
                        destination_shard_dim=1,
                        destination_local_shape=_local_shape(global_shape, 1, destination_tp_size),
                        source_recipe="dense_fc2",
                        destination_recipe="dense_down",
                        destination_parameter=f"model.layers.{layer}.mlp.down_proj.weight",
                    )
                )

    for pp_rank_text, expert_world in layout["expert_world_by_pp"].items():
        pp_rank = int(pp_rank_text)
        source_mesh = layout["expert_source_mesh_by_pp"][pp_rank_text]
        specs_by_world: dict[int, dict[tuple[int, str], list[dict[str, Any]]]] = {}
        for world in expert_world:
            grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
            for spec in payload_by_world[world]["specs"]:
                if spec["family"] == "routed_expert":
                    grouped.setdefault((spec["layer"], spec["projection"]), []).append(spec)
            for specs in grouped.values():
                specs.sort(key=lambda item: item["expert_id"])
            specs_by_world[world] = grouped
        expert_keys = set.intersection(*(set(specs) for specs in specs_by_world.values()))
        if any(set(specs) != expert_keys for specs in specs_by_world.values()):
            raise ValueError(f"Expert FFN source specs differ across selected EP ranks for PP={pp_rank}")
        for layer, projection in sorted(expert_keys):
            record_layer_owner(layer, pp_rank)
            per_world = [specs_by_world[world][(layer, projection)] for world in expert_world]
            counts = {len(specs) for specs in per_world}
            if len(counts) != 1 or not counts:
                raise ValueError(f"Uneven local expert counts for layer {layer} " f"{projection}: {counts}")
            local_experts = counts.pop()
            for shard, specs in enumerate(per_world):
                actual_ids = [spec["expert_id"] for spec in specs]
                expected_ids = list(
                    range(
                        shard * local_experts,
                        (shard + 1) * local_experts,
                    )
                )
                if actual_ids != expected_ids:
                    raise ValueError(
                        f"Layer {layer} {projection} source shard {shard} must "
                        f"own expert IDs {expected_ids}, got {actual_ids}"
                    )
            flat_specs = [spec for specs in per_world for spec in specs]
            expert_ids = sorted(spec["expert_id"] for spec in flat_specs)
            if expert_ids != list(range(len(expert_ids))):
                raise ValueError(
                    f"Layer {layer} {projection} expert IDs must be complete and contiguous, got {expert_ids}"
                )
            _same_specs(
                flat_specs,
                (
                    "dtype",
                    "local_shape",
                    "partition_dim",
                    "partition_stride",
                ),
                f"expert layer {layer} {projection}",
            )
            spec = flat_specs[0]
            rows, columns = spec["local_shape"]
            source_names_by_rank = {
                world_to_comm[world]: [item["name"] for item in specs_by_world[world][(layer, projection)]]
                for world in expert_world
            }
            num_experts = local_experts * len(expert_world)
            destination_shard_dim = _expert_destination_shard_dim(
                projection,
                destination_tp_size,
                destination_ep_size,
            )
            if projection == "fc1":
                if rows % 2:
                    raise ValueError(f"Expert fused gate/up rows must be even for {spec['name']}")
                intermediate = rows // 2
                global_shape = [num_experts, intermediate, columns]
                for component, index in (("gate", 0), ("up", 1)):
                    weight = _entry(
                        name=f"model.layers.{layer}.mlp.experts.{component}_proj.weight",
                        family="routed_expert",
                        pp_rank=pp_rank,
                        source_names_by_rank=source_names_by_rank,
                        dtype=spec["dtype"],
                        global_shape=global_shape,
                        source_mesh=source_mesh,
                        source_shard_dim=0,
                        source_local_shape=[
                            local_experts,
                            intermediate,
                            columns,
                        ],
                        destination_mesh=destination_mesh,
                        destination_shard_dim=destination_shard_dim,
                        destination_local_shape=_local_shape(
                            global_shape,
                            destination_shard_dim,
                            destination_tp_size,
                        ),
                        source_recipe=f"expert_fc1_{index}",
                        destination_recipe=f"expert_{component}",
                        destination_parameter=f"model.layers.{layer}.mlp.experts.w13_weight",
                    )
                    entries.extend(_fp8_expert_pair(weight) if quantization is not None else (weight,))
            else:
                global_shape = [num_experts, rows, columns]
                weight = _entry(
                    name=f"model.layers.{layer}.mlp.experts.down_proj.weight",
                    family="routed_expert",
                    pp_rank=pp_rank,
                    source_names_by_rank=source_names_by_rank,
                    dtype=spec["dtype"],
                    global_shape=global_shape,
                    source_mesh=source_mesh,
                    source_shard_dim=0,
                    source_local_shape=[local_experts, rows, columns],
                    destination_mesh=destination_mesh,
                    destination_shard_dim=destination_shard_dim,
                    destination_local_shape=_local_shape(
                        global_shape,
                        destination_shard_dim,
                        destination_tp_size,
                    ),
                    source_recipe="expert_fc2",
                    destination_recipe="expert_down",
                    destination_parameter=f"model.layers.{layer}.mlp.experts.w2_weight",
                )
                entries.extend(_fp8_expert_pair(weight) if quantization is not None else (weight,))

    if not entries:
        raise ValueError("The selected model has no FFN update unit supported by NCCL M2N")
    entries.sort(key=lambda item: (item["pp_rank"], item["name"]))
    all_units = {tuple(unit) for payload in trainer_payloads for unit in payload["update_units"]}
    while True:
        covered_names = {name for item in entries for name in _entry_source_names(item)}
        routed_units = sorted(unit for unit in all_units if unit and set(unit).issubset(covered_names))
        routed_names = {name for unit in routed_units for name in unit}
        retained_entries = [item for item in entries if _entry_source_names(item).issubset(routed_names)]
        if quantization is not None:
            retained_entries = _retain_complete_fp8_modules(retained_entries)
        if len(retained_entries) == len(entries):
            break
        entries = retained_entries
    if not entries:
        raise ValueError("No complete atomic FFN update unit can be routed through NCCL M2N")
    if quantization is not None:
        _validate_fp8_pairs(entries)
    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "source_world_ranks": layout["source_world_ranks"],
        "trainer_world_to_comm_rank": layout["trainer_world_to_comm_rank"],
        "communicator_world_size": source_count + destination_count,
        "routed_update_units": [list(unit) for unit in routed_units],
        "entries": entries,
    }
    if quantization is not None:
        manifest["quantization"] = quantization
    manifest["manifest_hash"] = _manifest_digest(manifest)
    return manifest


def _process_group_options() -> Any:
    options = dist.ProcessGroupNCCL.Options()
    options.config.blocking = 1
    return options


def _nccl_rl() -> Any:
    try:
        from nccl import m2n
    except Exception as exc:
        raise RuntimeError(
            "nccl-rl was selected, but its nccl.m2n package or native library " "is unavailable"
        ) from exc
    return m2n


def _warm_and_borrow_nccl_comm(pg: dist.ProcessGroup, device: torch.device) -> int:
    if device.type != "cuda":
        raise RuntimeError(f"nccl-rl requires CUDA, got {device}")
    torch.cuda.set_device(device)
    dist.all_reduce(torch.zeros(1, device=device), group=pg)
    torch.cuda.synchronize(device)
    comm_ptr = int(pg._get_backend(device)._comm_ptr())
    if not comm_ptr:
        raise RuntimeError("ProcessGroupNCCL returned a null communicator pointer")
    return comm_ptr


def _check_engine_results(results: Sequence[Any], operation: str) -> None:
    for result in results:
        if result is None:
            continue
        success = result.get("success") if isinstance(result, Mapping) else getattr(result, "success", None)
        message = result.get("message", "") if isinstance(result, Mapping) else getattr(result, "message", "")
        if success is not True:
            raise RuntimeError(
                f"SGLang {operation} returned an unsuccessful or unsupported " f"response {result!r}: {message}"
            )


def _collect_errors(error: str | None) -> list[str]:
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error, group=get_gloo_group())
    return [item for item in errors if item]


class UpdateWeightFromNcclM2N(UpdateWeightFromDistributed):
    """Send supported FFN weights with nccl-rl and broadcast the rest."""

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        is_lora: bool = False,
    ) -> None:
        self._m2n_quantization = _fp8_manifest_quantization(quantization_config)
        if not torch.cuda.is_available():
            raise RuntimeError("nccl-rl requires CUDA in the trainer process")
        _nccl_rl()

        super().__init__(
            args,
            model,
            weights_getter,
            model_name=model_name,
            quantization_config=quantization_config,
            is_lora=is_lora,
        )
        self._m2n_group_name: str | None = None
        self._m2n_pg: dist.ProcessGroup | None = None
        self._m2n_comm_ptr: int | None = None
        self._m2n_manifest: dict[str, Any] | None = None
        self._m2n_comm_rank: int | None = None
        self._m2n_local_tensors: dict[str, torch.Tensor] = {}
        self._m2n_fp8_pair_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._m2n_routed_units: set[tuple[str, ...]] = set()
        self._residual_pp_rank = get_parallel_state().pp.rank
        self._residual_group_started = False
        self._deferred_update_error: str | None = None

    def update_weights(self) -> None:
        self._deferred_update_error = None
        try:
            super().update_weights()
        except Exception:
            self._connection_stale = True
            raise

    def _pause_and_prepare_engines(self) -> None:
        error: str | None = None
        try:
            super()._pause_and_prepare_engines()
        except Exception as exc:
            error = f"trainer rank {dist.get_rank()} rollout prepare: " f"{type(exc).__name__}: {exc}"
        failures = _collect_errors(error)
        if failures:
            self._connection_stale = True
            raise RuntimeError("NCCL M2N rollout preparation failed: " + " | ".join(failures))

    @property
    def _is_source(self) -> bool:
        ps = get_parallel_state()
        return ps.tp.rank == 0 and ps.cp.rank == 0 and ps.intra_dp.rank == 0 and ps.indep_dp.rank == 0

    def _trainer_payload(self) -> dict[str, Any]:
        ps = get_parallel_state()
        local_tensors: dict[str, torch.Tensor] = {}
        specs: list[dict[str, Any]] = []
        update_units: list[list[str]] = []
        for is_expert in (False, True):
            for unit in super()._get_weight_transfer_update_units(is_expert):
                names = tuple(name for name, _tensor in unit)
                update_units.append(list(names))
                for name, tensor in unit:
                    local_tensors[name] = tensor
                    spec = _local_source_spec(name, tensor)
                    if spec is not None:
                        specs.append(spec)
        self._m2n_local_tensors = local_tensors
        return {
            "topology": {
                "world_rank": dist.get_rank(),
                "tp_rank": ps.tp.rank,
                "tp_size": ps.tp.size,
                "pp_rank": ps.pp.rank,
                "pp_size": ps.pp.size,
                "cp_rank": ps.cp.rank,
                "cp_size": ps.cp.size,
                "dense_dp_rank": ps.intra_dp.rank,
                "dense_dp_size": ps.intra_dp.size,
                "ep_rank": ps.ep.rank,
                "ep_size": ps.ep.size,
                "etp_rank": ps.etp.rank,
                "etp_size": ps.etp.size,
                "expert_dp_rank": mpu.get_expert_data_parallel_rank(),
                "expert_dp_size": mpu.get_expert_data_parallel_world_size(),
                "independent_dp_rank": ps.indep_dp.rank,
                "independent_dp_size": ps.indep_dp.size,
            },
            "specs": specs,
            "update_units": update_units,
        }

    def _negotiate_manifest(self, engine_gpu_counts: Sequence[int]) -> dict[str, Any]:
        try:
            record = {"payload": self._trainer_payload(), "error": None}
        except Exception as exc:
            record = {
                "payload": None,
                "error": (f"trainer rank {dist.get_rank()}: " f"{type(exc).__name__}: {exc}"),
            }
        gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, record, group=get_gloo_group())
        objects: list[dict[str, Any] | None] = [None]
        if dist.get_rank() == 0:
            failures = [item["error"] for item in gathered if item is not None and item["error"] is not None]
            if failures:
                objects[0] = {
                    "manifest": None,
                    "error": " | ".join(failures),
                }
            else:
                payloads = [item["payload"] for item in gathered if item is not None and item["payload"] is not None]
                try:
                    manifest = _build_manifest(
                        payloads,
                        engine_gpu_counts,
                        quantization_config=self._m2n_quantization,
                        destination_ep_size=self.args.sglang_ep_size,
                    )
                    objects[0] = {
                        "manifest": manifest,
                        "error": None,
                    }
                except Exception as exc:
                    objects[0] = {
                        "manifest": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        dist.broadcast_object_list(objects, src=0, group=get_gloo_group())
        result = objects[0]
        if result is None:
            raise RuntimeError("Trainer rank zero did not produce an NCCL M2N manifest")
        if result["error"] is not None:
            raise RuntimeError(f"NCCL M2N manifest negotiation failed: {result['error']}")
        manifest = result["manifest"]
        if manifest is None:
            raise RuntimeError("NCCL M2N manifest negotiation returned no manifest")
        if manifest.get("manifest_hash") != _manifest_digest(manifest):
            raise RuntimeError("NCCL M2N manifest hash validation failed")
        self._m2n_manifest = manifest
        self._m2n_routed_units = {tuple(unit) for unit in manifest["routed_update_units"]}
        return manifest

    def _teardown_local_m2n(self) -> None:
        if getattr(self, "_m2n_comm_ptr", None) is not None:
            torch.cuda.synchronize()
            _nccl_rl().finalize()
            self._m2n_comm_ptr = None
        if getattr(self, "_m2n_pg", None) is not None:
            dist.destroy_process_group(self._m2n_pg)
            self._m2n_pg = None
        self._m2n_comm_rank = None

    def _disconnect_existing(
        self,
        replacement_engines: Sequence[ActorHandle] | None = None,
    ) -> None:
        old_engines = self.rollout_engines
        if old_engines is None:
            return

        def record_destroy_error(
            errors: list[str],
            engine: ActorHandle,
            exc: Exception,
        ) -> None:
            retired = replacement_engines is not None and engine not in replacement_engines
            if retired and _is_unreachable_engine_error(exc):
                logger.info(
                    "Ignoring teardown from an unreachable retired rollout "
                    "engine; it must restart before reuse: %s",
                    exc,
                )
            else:
                errors.append(f"{type(exc).__name__}: {exc}")

        def launch_remote_destroy(
            group_name: str,
            *,
            current_rank: bool = False,
        ) -> tuple[list[tuple[ActorHandle, Any]], list[str]]:
            if not current_rank and dist.get_rank() != 0:
                return [], []
            refs: list[tuple[ActorHandle, Any]] = []
            errors: list[str] = []
            for engine in old_engines:
                try:
                    refs.append(
                        (
                            engine,
                            engine.destroy_weights_update_group.remote(group_name),
                        )
                    )
                except Exception as exc:
                    record_destroy_error(errors, engine, exc)
            return refs, errors

        def finish_remote_destroy(
            refs: Sequence[tuple[ActorHandle, Any]],
            errors: list[str],
            operation: str,
        ) -> str | None:
            for engine, ref in refs:
                try:
                    _check_engine_results([ray.get(ref)], operation)
                except Exception as exc:
                    record_destroy_error(errors, engine, exc)
            return " | ".join(errors) or None

        def raise_collective(error: str | None, message: str) -> None:
            failures = _collect_errors(error)
            if failures:
                self._connection_stale = True
                raise RuntimeError(f"{message}: {' | '.join(failures)}")

        m2n_refs, m2n_errors = (
            launch_remote_destroy(self._m2n_group_name)
            if self._m2n_manifest is not None and self._m2n_group_name is not None
            else ([], [])
        )
        local_m2n_error: str | None = None
        try:
            self._teardown_local_m2n()
        except Exception as exc:
            local_m2n_error = f"trainer rank {dist.get_rank()} local M2N teardown: " f"{type(exc).__name__}: {exc}"
        m2n_error = finish_remote_destroy(m2n_refs, m2n_errors, "M2N teardown")
        if local_m2n_error is not None:
            m2n_error = local_m2n_error if m2n_error is None else f"{local_m2n_error} | {m2n_error}"
        raise_collective(
            m2n_error,
            f"Failed to destroy nccl-rl connection {self._m2n_group_name!r}",
        )

        pp_size = getattr(
            getattr(self, "args", None),
            "pipeline_model_parallel_size",
            1,
        )
        for pp_rank in range(pp_size):
            residual_error: str | None = None
            if (
                self._is_source
                and getattr(self, "_residual_pp_rank", 0) == pp_rank
                and self._m2n_manifest is not None
                and (getattr(self, "_residual_group_started", False) or self._model_update_groups is not None)
            ):
                residual_refs, residual_errors = launch_remote_destroy(
                    self._group_name,
                    current_rank=True,
                )
                if self._model_update_groups is not None:
                    try:
                        dist.destroy_process_group(self._model_update_groups)
                        self._model_update_groups = None
                    except Exception as exc:
                        residual_error = (
                            f"trainer rank {dist.get_rank()} local residual teardown: " f"{type(exc).__name__}: {exc}"
                        )
                remote_error = finish_remote_destroy(
                    residual_refs,
                    residual_errors,
                    "residual broadcast teardown",
                )
                if remote_error is not None:
                    residual_error = remote_error if residual_error is None else f"{residual_error} | {remote_error}"
                if residual_error is None:
                    self._residual_group_started = False

            raise_collective(
                residual_error,
                "Failed to destroy the previous residual broadcast connection",
            )

        self._m2n_manifest = None
        self._m2n_routed_units.clear()
        self._m2n_group_name = None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        del engine_gpu_offsets
        engine_gpu_counts = _validated_engine_gpu_counts(
            self.args,
            len(rollout_engines),
            engine_gpu_counts,
        )
        self._disconnect_existing(rollout_engines)

        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = list(engine_gpu_counts)
        self._residual_pp_rank = get_parallel_state().pp.rank
        self._group_name = f"miles-pp_{self._residual_pp_rank}"
        manifest = self._negotiate_manifest(engine_gpu_counts)

        connection: list[dict[str, Any] | None] = [None]
        rendezvous_world_rank = manifest["source_world_ranks"][0]
        if dist.get_rank() == rendezvous_world_rank:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            connection[0] = {
                "master_address": master_address,
                "master_port": master_port,
                "group_name": _new_m2n_group_name(),
            }
        dist.broadcast_object_list(
            connection,
            src=rendezvous_world_rank,
            group=get_gloo_group(),
        )
        if connection[0] is None:
            raise RuntimeError(f"Trainer rank {rendezvous_world_rank} did not publish NCCL M2N " "rendezvous metadata")
        self._m2n_group_name = connection[0]["group_name"]

        refs = []
        if dist.get_rank() == 0:
            rank_cursor = len(manifest["source_world_ranks"])
            for engine, count in zip(rollout_engines, engine_gpu_counts, strict=True):
                refs.append(
                    engine.init_weights_update_group.remote(
                        connection[0]["master_address"],
                        connection[0]["master_port"],
                        rank_cursor,
                        manifest["communicator_world_size"],
                        self._m2n_group_name,
                        backend="nccl",
                        m2n_manifest=manifest,
                    )
                )
                rank_cursor += count

        local_error: str | None = None
        world_to_comm = manifest["trainer_world_to_comm_rank"]
        comm_rank = world_to_comm.get(str(dist.get_rank()))
        if comm_rank is not None:
            try:
                self._m2n_comm_rank = comm_rank
                device = torch.device("cuda", torch.cuda.current_device())
                self._m2n_pg = init_process_group(
                    backend="nccl",
                    init_method=(f"tcp://{connection[0]['master_address']}:{connection[0]['master_port']}"),
                    world_size=manifest["communicator_world_size"],
                    rank=comm_rank,
                    group_name=self._m2n_group_name,
                    pg_options=_process_group_options(),
                )
                self._m2n_comm_ptr = _warm_and_borrow_nccl_comm(self._m2n_pg, device)
            except Exception as exc:
                local_error = f"trainer rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
        if dist.get_rank() == 0:
            try:
                results = ray.get(refs)
                _check_engine_results(results, "M2N initialization")
            except Exception as exc:
                local_error = f"SGLang initialization: {type(exc).__name__}: {exc}"

        failures = _collect_errors(local_error)
        if failures:
            cleanup_refs = []
            if dist.get_rank() == 0:
                cleanup_refs = [
                    engine.destroy_weights_update_group.remote(self._m2n_group_name) for engine in rollout_engines
                ]
            self._teardown_local_m2n()
            if dist.get_rank() == 0:
                try:
                    _check_engine_results(ray.get(cleanup_refs), "failed M2N setup cleanup")
                except Exception as exc:
                    logger.warning(
                        "Failed to clean up SGLang M2N groups after setup failure: %s",
                        exc,
                    )
            self._connection_stale = True
            raise RuntimeError("NCCL M2N connection setup failed: " + " | ".join(failures))

        local_pp_rank = self._residual_pp_rank
        for pp_rank in range(self.args.pipeline_model_parallel_size):
            residual_error: str | None = None
            if self._is_source and local_pp_rank == pp_rank:
                try:
                    self._residual_group_started = True
                    self._model_update_groups = connect_rollout_engines_from_distributed(
                        self.args,
                        self._group_name,
                        rollout_engines,
                        engine_gpu_counts=engine_gpu_counts,
                    )
                except Exception as exc:
                    residual_error = (
                        f"trainer rank {dist.get_rank()} residual broadcast setup: " f"{type(exc).__name__}: {exc}"
                    )
            residual_failures = _collect_errors(residual_error)
            if residual_failures:
                self._connection_stale = True
                self._disconnect_existing()
                raise RuntimeError("NCCL M2N residual connection setup failed: " + " | ".join(residual_failures))
        self._connection_stale = False

    def _use_bucketed_update_unit(self, update_unit: list[tuple[str, torch.Tensor]]) -> bool:
        return tuple(name for name, _tensor in update_unit) not in self._m2n_routed_units

    def _update_weight_implementation(
        self,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        pbar: Any | None = None,
    ) -> None:
        if self._deferred_update_error is not None:
            converted_named_tensors.clear()
            if pbar:
                pbar.update(1)
            return

        broadcast_bytes = sum(tensor.numel() * tensor.element_size() for _name, tensor in converted_named_tensors)
        lock_acquired = False
        try:
            while not ray.get(self.rollout_engine_lock.acquire.remote()):
                time.sleep(0.1)
            lock_acquired = True
            refs = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                None,
                self.rollout_engines,
                converted_named_tensors,
            )
            ray.get(refs)
        except Exception as exc:
            self._deferred_update_error = (
                f"trainer rank {dist.get_rank()} residual update: " f"{type(exc).__name__}: {exc}"
            )
        finally:
            if lock_acquired:
                try:
                    ray.get(self.rollout_engine_lock.release.remote())
                except Exception as exc:
                    release_error = (
                        f"trainer rank {dist.get_rank()} residual lock release: " f"{type(exc).__name__}: {exc}"
                    )
                    self._deferred_update_error = (
                        release_error
                        if self._deferred_update_error is None
                        else f"{self._deferred_update_error}; {release_error}"
                    )
        converted_named_tensors.clear()
        if pbar:
            pbar.update(1)
        if dist.get_rank() == 0 and self._deferred_update_error is None:
            self.update_weight_metrics["m2n_broadcast_bytes"] += float(broadcast_bytes)

    def _raise_deferred_update_errors(self) -> None:
        failures = _collect_errors(self._deferred_update_error)
        if failures:
            self._connection_stale = True
            raise RuntimeError("NCCL M2N residual update failed: " + " | ".join(failures))

    def _source_tensor(self, entry: Mapping[str, Any]) -> torch.Tensor:
        if self._m2n_comm_rank is None:
            raise RuntimeError("Current trainer rank is not in the NCCL M2N communicator")
        source = entry["source"]
        names = source["names_by_rank"].get(str(self._m2n_comm_rank))
        if not names:
            raise RuntimeError(
                f"Manifest entry {entry['name']} has no source recipe for communicator rank " f"{self._m2n_comm_rank}"
            )
        recipe = source["recipe"]
        pair_id = entry.get("pair_id")
        tensor_role = entry.get("tensor_role")
        if pair_id is not None:
            if (
                not isinstance(pair_id, str)
                or tensor_role not in ("weight", "scale")
                or not recipe.startswith("expert_")
            ):
                raise RuntimeError(f"Invalid NCCL M2N FP8 source recipe for {entry['name']}")
            cache = self._m2n_fp8_pair_cache
            if pair_id not in cache:
                base_recipe = recipe.removesuffix("_scale") if tensor_role == "scale" else recipe
                tensors = [self._m2n_local_tensors[name].data for name in names]
                logical = self._logical_source_tensor(tensors, base_recipe)
                qweight, scale = _quantize_canonical_block_fp8(logical)
                cache[pair_id] = {"weight": qweight, "scale": scale}
            result = cache[pair_id][tensor_role]
            expected_shape = tuple(source["local_shape"])
            expected_dtype = _dtype_from_name(entry["dtype"])
            if tuple(result.shape) != expected_shape or result.dtype != expected_dtype:
                raise RuntimeError(
                    f"NCCL M2N FP8 source {entry['name']} produced "
                    f"shape={tuple(result.shape)} dtype={result.dtype}; expected "
                    f"shape={expected_shape} dtype={expected_dtype}"
                )
            return result

        tensors = [self._m2n_local_tensors[name].data for name in names]
        return self._logical_source_tensor(tensors, recipe)

    @staticmethod
    def _logical_source_tensor(
        tensors: Sequence[torch.Tensor],
        recipe: str,
    ) -> torch.Tensor:
        if recipe.startswith("dense_fc1_"):
            index = int(recipe.rsplit("_", 1)[1])
            return tensors[0].chunk(2, dim=0)[index].contiguous()
        if recipe == "dense_fc2":
            return tensors[0].contiguous()
        if recipe.startswith("expert_fc1_"):
            index = int(recipe.rsplit("_", 1)[1])
            return torch.stack([tensor.chunk(2, dim=0)[index] for tensor in tensors])
        if recipe == "expert_fc2":
            return torch.stack(tensors)
        raise RuntimeError(f"Unknown NCCL M2N source recipe {recipe!r}")

    def _run_m2n_batch(self) -> None:
        if (
            self._m2n_manifest is None
            or self._m2n_pg is None
            or self._m2n_comm_ptr is None
            or self._m2n_comm_rank is None
        ):
            return
        m2n = _nccl_rl()
        stream = torch.cuda.current_stream()

        def placements(descriptors: Sequence[Mapping[str, Any]]) -> list[Any]:
            result = []
            for descriptor in descriptors:
                if descriptor["type"] == "replicate":
                    result.append(m2n.Replicate())
                elif descriptor["type"] == "shard":
                    result.append(m2n.Shard(descriptor["dim"]))
                else:
                    raise RuntimeError(f"Unknown NCCL M2N placement {descriptor!r}")
            return result

        try:
            for entry in self._m2n_manifest["entries"]:
                source_descriptor = entry["source"]
                destination_descriptor = entry["destination"]
                source = None
                if any(self._m2n_comm_rank in row for row in source_descriptor["mesh"]):
                    source = self._source_tensor(entry)
                dtype = _dtype_from_name(entry["dtype"])
                m2n.reshard(
                    source,
                    None,
                    self._m2n_comm_ptr,
                    stream,
                    src_mesh=source_descriptor["mesh"],
                    src_placements=placements(source_descriptor["placements"]),
                    src_local_shape=source_descriptor["local_shape"],
                    src_dtype=dtype,
                    dst_mesh=destination_descriptor["mesh"],
                    dst_placements=placements(destination_descriptor["placements"]),
                    dst_local_shape=destination_descriptor["local_shape"],
                    dst_dtype=dtype,
                )
                stream.synchronize()
                tensor_role = entry.get("tensor_role")
                if tensor_role is not None:
                    pair = self._m2n_fp8_pair_cache.get(entry["pair_id"])
                    if pair is not None:
                        pair.pop(tensor_role, None)
                        if not pair:
                            self._m2n_fp8_pair_cache.pop(entry["pair_id"])
        finally:
            self._m2n_fp8_pair_cache.clear()

    def _update_bulk_weights(self) -> bool:
        if self._m2n_manifest is None or self._m2n_group_name is None:
            raise RuntimeError("NCCL M2N updater is not connected")
        refs = []
        lock_acquired = False
        startup_error: str | None = None
        if dist.get_rank() == 0:
            try:
                while not ray.get(self.rollout_engine_lock.acquire.remote()):
                    time.sleep(0.1)
                lock_acquired = True
                refs = [
                    engine.update_weights_from_distributed.remote(
                        names=[entry["name"] for entry in self._m2n_manifest["entries"]],
                        dtypes=[_dtype_from_name(entry["dtype"]) for entry in self._m2n_manifest["entries"]],
                        shapes=[entry["global_shape"] for entry in self._m2n_manifest["entries"]],
                        group_name=self._m2n_group_name,
                        load_format="nccl_m2n",
                    )
                    for engine in self.rollout_engines
                ]
            except Exception as exc:
                startup_error = f"trainer rank 0 M2N startup: {type(exc).__name__}: {exc}"

        startup_failures = _collect_errors(startup_error)
        if startup_failures:
            if dist.get_rank() == 0 and lock_acquired:
                ray.get(self.rollout_engine_lock.release.remote())
            raise RuntimeError("NCCL M2N update startup failed: " + " | ".join(startup_failures))

        local_error: str | None = None
        try:
            self._run_m2n_batch()
        except Exception as exc:
            local_error = f"trainer rank {dist.get_rank()} M2N transfer: " f"{type(exc).__name__}: {exc}"

        if dist.get_rank() == 0:
            try:
                _check_engine_results(ray.get(refs), "M2N weight update")
            except Exception as exc:
                local_error = (
                    f"SGLang M2N transfer: {type(exc).__name__}: {exc}"
                    if local_error is None
                    else f"{local_error}; SGLang M2N transfer: " f"{type(exc).__name__}: {exc}"
                )
            try:
                if lock_acquired:
                    ray.get(self.rollout_engine_lock.release.remote())
            except Exception as exc:
                local_error = (
                    f"rollout-engine lock release: {type(exc).__name__}: {exc}"
                    if local_error is None
                    else f"{local_error}; rollout-engine lock release: " f"{type(exc).__name__}: {exc}"
                )

        failures = _collect_errors(local_error)
        if failures:
            raise RuntimeError("NCCL M2N weight update failed: " + " | ".join(failures))

        if dist.get_rank() == 0:
            self.update_weight_metrics = {
                "m2n_staging_bytes": float(
                    sum(
                        _tensor_bytes(entry["global_shape"], entry["dtype"]) for entry in self._m2n_manifest["entries"]
                    )
                ),
                "m2n_broadcast_bytes": 0.0,
            }
        return True

    def _finalize_and_resume_engines(self) -> None:
        local_error: str | None = None
        if dist.get_rank() == 0:
            try:
                end_weight_update(self.rollout_engines)
                results = ray.get(
                    [
                        engine.update_weight_version.remote(weight_version=str(self.weight_version))
                        for engine in self.rollout_engines
                    ]
                )
                _check_engine_results(results, "weight-version finalization")
                ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
            except Exception as exc:
                local_error = f"rollout finalization: {type(exc).__name__}: {exc}"
        failures = _collect_errors(local_error)
        if failures:
            raise RuntimeError("NCCL M2N rollout finalization failed: " + " | ".join(failures))
