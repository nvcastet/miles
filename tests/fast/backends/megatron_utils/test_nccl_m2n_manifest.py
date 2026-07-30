from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import (
    nccl_m2n,
)
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.nccl_m2n import (
    UpdateWeightFromNcclM2N,
    _build_manifest,
    _new_m2n_group_name,
)


_LAYER_COUNTS = (4, 6, 6, 6, 6, 6, 6, 3)
_EXPERTS_PER_RANK = 2


def _dense_name(layer, projection):
    return f"module.module.decoder.layers.{layer}.mlp.linear_fc{projection}.weight"


def _expert_name(layer, projection, expert_id):
    return f"module.module.decoder.layers.{layer}.mlp.experts." f"linear_fc{projection}.weight{expert_id}"


def _unmapped_name(layer):
    return f"module.module.decoder.layers.{layer}." "self_attention.linear_proj.weight"


_DENSE_FC1 = _dense_name(0, 1)
_UNMAPPED = _unmapped_name(0)
_FP8_CONFIG = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
}


def _spec(
    name,
    *,
    layer,
    family,
    projection,
    local_shape,
    expert_id=None,
    partition_dim=-1,
    partition_stride=1,
):
    return {
        "name": name,
        "family": family,
        "layer": layer,
        "projection": projection,
        "expert_id": expert_id,
        "dtype": "bfloat16",
        "local_shape": local_shape,
        "partition_dim": partition_dim,
        "partition_stride": partition_stride,
    }


def _layers_by_stage():
    cursor = 0
    result = []
    for count in _LAYER_COUNTS:
        result.append(tuple(range(cursor, cursor + count)))
        cursor += count
    return tuple(result)


_LAYERS_BY_STAGE = _layers_by_stage()


def _payload(world_rank):
    pp_rank, stage_rank = divmod(world_rank, 4)
    tp_rank = stage_rank % 2
    cp_rank = stage_rank // 2
    ep_rank = stage_rank
    specs = []
    update_units = []
    for layer in _LAYERS_BY_STAGE[pp_rank]:
        dense_fc1 = _dense_name(layer, 1)
        dense_fc2 = _dense_name(layer, 2)
        specs.extend(
            [
                _spec(
                    dense_fc1,
                    layer=layer,
                    family="dense",
                    projection="fc1",
                    local_shape=[8, 4],
                    partition_dim=0,
                    partition_stride=2,
                ),
                _spec(
                    dense_fc2,
                    layer=layer,
                    family="dense",
                    projection="fc2",
                    local_shape=[4, 4],
                    partition_dim=1,
                ),
            ]
        )
        update_units.extend(([dense_fc1], [dense_fc2]))
        for expert_id in range(
            ep_rank * _EXPERTS_PER_RANK,
            (ep_rank + 1) * _EXPERTS_PER_RANK,
        ):
            expert_fc1 = _expert_name(layer, 1, expert_id)
            expert_fc2 = _expert_name(layer, 2, expert_id)
            specs.extend(
                [
                    _spec(
                        expert_fc1,
                        layer=layer,
                        family="routed_expert",
                        projection="fc1",
                        local_shape=[8, 4],
                        expert_id=expert_id,
                    ),
                    _spec(
                        expert_fc2,
                        layer=layer,
                        family="routed_expert",
                        projection="fc2",
                        local_shape=[4, 4],
                        expert_id=expert_id,
                    ),
                ]
            )
            update_units.extend(([expert_fc1], [expert_fc2]))
        update_units.append([_unmapped_name(layer)])
    return {
        "topology": {
            "world_rank": world_rank,
            "pp_rank": pp_rank,
            "pp_size": 8,
            "tp_rank": tp_rank,
            "tp_size": 2,
            "cp_rank": cp_rank,
            "cp_size": 2,
            "dense_dp_rank": 0,
            "dense_dp_size": 1,
            "ep_rank": ep_rank,
            "ep_size": 4,
            "etp_rank": 0,
            "etp_size": 1,
            "expert_dp_rank": 0,
            "expert_dp_size": 1,
            "independent_dp_rank": 0,
            "independent_dp_size": 1,
        },
        "specs": specs,
        "update_units": update_units,
    }


def _payloads():
    return [_payload(world_rank) for world_rank in range(32)]


def _reduced_pp2_payload(world_rank):
    pp_rank, stage_rank = divmod(world_rank, 2)
    specs = []
    update_units = []
    for layer in ((0, 1), (2,))[pp_rank]:
        for projection, local_shape, partition_dim, partition_stride in (
            (1, [4, 4], 0, 2),
            (2, [4, 4], 1, 1),
        ):
            name = _dense_name(layer, projection)
            specs.append(
                _spec(
                    name,
                    layer=layer,
                    family="dense",
                    projection=f"fc{projection}",
                    local_shape=local_shape,
                    partition_dim=partition_dim,
                    partition_stride=partition_stride,
                )
            )
            update_units.append([name])
        for projection, local_shape in ((1, [8, 4]), (2, [4, 4])):
            name = _expert_name(layer, projection, stage_rank)
            specs.append(
                _spec(
                    name,
                    layer=layer,
                    family="routed_expert",
                    projection=f"fc{projection}",
                    local_shape=local_shape,
                    expert_id=stage_rank,
                )
            )
            update_units.append([name])
    return {
        "topology": {
            "world_rank": world_rank,
            "pp_rank": pp_rank,
            "pp_size": 2,
            "tp_rank": stage_rank,
            "tp_size": 2,
            "cp_rank": 0,
            "cp_size": 1,
            "dense_dp_rank": 0,
            "dense_dp_size": 1,
            "ep_rank": stage_rank,
            "ep_size": 2,
            "etp_rank": 0,
            "etp_size": 1,
            "expert_dp_rank": 0,
            "expert_dp_size": 1,
            "independent_dp_rank": 0,
            "independent_dp_size": 1,
        },
        "specs": specs,
        "update_units": update_units,
    }


def _fp8_payloads():
    payloads = [_reduced_pp2_payload(world_rank) for world_rank in range(4)]
    for payload in payloads:
        for spec in payload["specs"]:
            if spec["family"] != "routed_expert":
                continue
            spec["local_shape"] = [512, 256] if spec["projection"] == "fc1" else [256, 256]
    return payloads


def _reduced_pp1_payload(world_rank):
    payload = _reduced_pp2_payload(world_rank)
    payload["topology"]["pp_size"] = 1
    return payload


def _dense_pp1_payload(world_rank):
    payload = _reduced_pp1_payload(world_rank)
    payload["topology"].update(
        ep_rank=0,
        ep_size=1,
        expert_dp_rank=world_rank,
        expert_dp_size=2,
    )
    payload["specs"] = [spec for spec in payload["specs"] if spec["family"] == "dense"]
    dense_names = {spec["name"] for spec in payload["specs"]}
    payload["update_units"] = [unit for unit in payload["update_units"] if set(unit) <= dense_names]
    return payload


def _entry(manifest, name):
    return next(entry for entry in manifest["entries"] if entry["name"] == name)


def _stage_for_layer(layer):
    return next(pp_rank for pp_rank, layers in enumerate(_LAYERS_BY_STAGE) if layer in layers)


def test_manifest_models_pp8_tp2_cp2_ep4_and_eight_rollout_engines():
    payloads = _payloads()
    manifest = _build_manifest(payloads, [4] * 8)
    reordered = _build_manifest(list(reversed(payloads)), [4] * 8)

    assert manifest == reordered
    assert manifest["schema_version"] == 1
    assert manifest["source_world_ranks"] == list(range(32))
    assert manifest["trainer_world_to_comm_rank"] == {str(rank): rank for rank in range(32)}
    assert manifest["communicator_world_size"] == 64
    assert "destination_mesh" not in manifest

    destination_mesh = [list(range(engine_start, engine_start + 4)) for engine_start in range(32, 64, 4)]
    dense_meshes_by_stage = {
        pp_rank: {
            tuple(tuple(row) for row in entry["source"]["mesh"])
            for entry in manifest["entries"]
            if entry["family"] == "dense" and entry["pp_rank"] == pp_rank
        }
        for pp_rank in range(8)
    }
    expert_meshes_by_stage = {
        pp_rank: {
            tuple(tuple(row) for row in entry["source"]["mesh"])
            for entry in manifest["entries"]
            if entry["family"] == "routed_expert" and entry["pp_rank"] == pp_rank
        }
        for pp_rank in range(8)
    }
    assert dense_meshes_by_stage == {pp_rank: {((4 * pp_rank, 4 * pp_rank + 1),)} for pp_rank in range(8)}
    assert expert_meshes_by_stage == {
        pp_rank: {tuple((tuple(range(4 * pp_rank, 4 * pp_rank + 4)),))} for pp_rank in range(8)
    }

    assert len(manifest["entries"]) == 43 * 6
    assert len({entry["name"] for entry in manifest["entries"]}) == 43 * 6
    assert all(
        entry["pp_rank"] == _stage_for_layer(int(entry["name"].split(".layers.", 1)[1].split(".", 1)[0]))
        for entry in manifest["entries"]
    )
    assert all(entry["destination"]["mesh"] == destination_mesh for entry in manifest["entries"])

    dense = _entry(manifest, "model.layers.12.mlp.gate_proj.weight")
    assert set(dense) == {
        "name",
        "family",
        "pp_rank",
        "dtype",
        "global_shape",
        "source",
        "destination",
    }
    assert set(dense["source"]) == {
        "mesh",
        "placements",
        "local_shape",
        "names_by_rank",
        "recipe",
    }
    assert set(dense["destination"]) == {
        "mesh",
        "placements",
        "local_shape",
        "parameter",
        "recipe",
    }
    assert dense["pp_rank"] == 2
    assert dense["source"]["mesh"] == [[8, 9]]
    assert dense["source"]["placements"] == [
        {"type": "replicate"},
        {"type": "shard", "dim": 0},
    ]
    assert dense["source"]["local_shape"] == [4, 4]
    assert dense["source"]["names_by_rank"] == {
        "8": [_dense_name(12, 1)],
        "9": [_dense_name(12, 1)],
    }
    assert dense["destination"]["mesh"] == destination_mesh
    assert dense["destination"]["placements"] == [
        {"type": "replicate"},
        {"type": "shard", "dim": 0},
    ]
    assert dense["destination"]["local_shape"] == [2, 4]

    expert = _entry(manifest, "model.layers.12.mlp.experts.gate_proj.weight")
    assert expert["family"] == "routed_expert"
    assert expert["pp_rank"] == 2
    assert expert["source"]["mesh"] == [[8, 9, 10, 11]]
    assert expert["source"]["placements"] == [
        {"type": "replicate"},
        {"type": "shard", "dim": 0},
    ]
    assert expert["source"]["local_shape"] == [2, 4, 4]
    assert expert["source"]["names_by_rank"] == {
        str(8 + ep_rank): [
            _expert_name(12, 1, expert_id)
            for expert_id in range(
                ep_rank * _EXPERTS_PER_RANK,
                (ep_rank + 1) * _EXPERTS_PER_RANK,
            )
        ]
        for ep_rank in range(4)
    }
    assert expert["destination"]["mesh"] == destination_mesh
    assert expert["destination"]["placements"] == [
        {"type": "replicate"},
        {"type": "shard", "dim": 0},
    ]
    assert expert["destination"]["local_shape"] == [2, 4, 4]
    assert expert["destination"]["parameter"] == ("model.layers.12.mlp.experts.w13_weight")

    assert "dense_source_mesh" not in manifest
    assert "expert_source_mesh" not in manifest
    assert "window_arena_bytes" not in manifest


def test_cartesian_tp_ep_coordinates_are_rejected():
    payloads = _payloads()
    for payload in payloads:
        stage_rank = payload["topology"]["world_rank"] % 4
        payload["topology"]["ep_rank"] = stage_rank // 2

    with pytest.raises(ValueError, match=r"(?i)(expert|EP)"):
        _build_manifest(payloads, [4] * 8)


def test_manifest_rejects_inconsistent_expert_coordinate_grid():
    payloads = _payloads()
    for payload in payloads:
        payload["topology"]["expert_dp_size"] = 2

    with pytest.raises(ValueError, match="expert topology describes 64 ranks"):
        _build_manifest(payloads, [4] * 8)


def test_permuted_complete_expert_id_partitions_are_rejected():
    payloads = _payloads()
    for world_rank in (0, 1):
        payload = payloads[world_rank]
        renamed = {}
        for spec in payload["specs"]:
            if spec["family"] != "routed_expert" or spec["layer"] != 0:
                continue
            old_name = spec["name"]
            new_expert_id = (spec["expert_id"] + 2) % 4
            projection = int(spec["projection"].removeprefix("fc"))
            spec["expert_id"] = new_expert_id
            spec["name"] = _expert_name(0, projection, new_expert_id)
            renamed[old_name] = spec["name"]
        payload["update_units"] = [[renamed.get(name, name) for name in unit] for unit in payload["update_units"]]

    with pytest.raises(ValueError, match="own expert IDs"):
        _build_manifest(payloads, [4] * 8)


def test_reduced_pp2_manifest_uses_each_layers_owning_stage():
    manifest = _build_manifest(
        [_reduced_pp2_payload(rank) for rank in range(4)],
        [2],
    )

    assert manifest["source_world_ranks"] == [0, 1, 2, 3]
    assert manifest["communicator_world_size"] == 6
    assert len(manifest["entries"]) == 3 * 6
    for layer, pp_rank, source_mesh in (
        (0, 0, [[0, 1]]),
        (1, 0, [[0, 1]]),
        (2, 1, [[2, 3]]),
    ):
        layer_entries = [entry for entry in manifest["entries"] if entry["name"].startswith(f"model.layers.{layer}.")]
        assert {entry["pp_rank"] for entry in layer_entries} == {pp_rank}
        assert all(entry["source"]["mesh"] == source_mesh for entry in layer_entries)
        assert all(entry["destination"]["mesh"] == [[4, 5]] for entry in layer_entries)


def test_reduced_pp1_manifest_builds_a_four_rank_2t2r_communicator():
    manifest = _build_manifest(
        [_dense_pp1_payload(rank) for rank in range(2)],
        [2],
        destination_ep_size=1,
    )

    assert manifest["source_world_ranks"] == [0, 1]
    assert manifest["trainer_world_to_comm_rank"] == {"0": 0, "1": 1}
    assert manifest["communicator_world_size"] == 4
    assert len(manifest["entries"]) == 2 * 3
    assert all(entry["source"]["mesh"] == [[0, 1]] for entry in manifest["entries"])
    assert all(entry["destination"]["mesh"] == [[2, 3]] for entry in manifest["entries"])


def test_routed_experts_support_rollout_ep1_with_moe_tensor_parallel_sharding():
    manifest = _build_manifest(
        [_reduced_pp1_payload(rank) for rank in range(2)],
        [2],
        destination_ep_size=1,
    )

    assert manifest["source_world_ranks"] == [0, 1]
    assert manifest["communicator_world_size"] == 4
    assert len(manifest["entries"]) == 2 * 6

    gate = _entry(manifest, "model.layers.0.mlp.experts.gate_proj.weight")
    assert gate["global_shape"] == [2, 4, 4]
    assert gate["destination"]["mesh"] == [[2, 3]]
    assert gate["destination"]["placements"] == [
        {"type": "replicate"},
        {"type": "shard", "dim": 1},
    ]
    assert gate["destination"]["local_shape"] == [2, 2, 4]

    down = _entry(manifest, "model.layers.0.mlp.experts.down_proj.weight")
    assert down["global_shape"] == [2, 4, 4]
    assert down["destination"]["mesh"] == [[2, 3]]
    assert down["destination"]["placements"] == [
        {"type": "replicate"},
        {"type": "shard", "dim": 2},
    ]
    assert down["destination"]["local_shape"] == [2, 4, 2]


def test_routed_experts_reject_hybrid_rollout_ep_and_moe_tp():
    with pytest.raises(ValueError, match="supports rollout EP=1 or EP=TP"):
        _build_manifest(
            _payloads(),
            [4] * 8,
            destination_ep_size=2,
        )


@pytest.mark.parametrize("scale_fmt", [None, "fp32"])
def test_fp8_manifest_v1_models_complete_expert_weight_scale_pairs(scale_fmt):
    quantization_config = deepcopy(_FP8_CONFIG)
    if scale_fmt is not None:
        quantization_config["scale_fmt"] = scale_fmt

    payloads = _fp8_payloads()
    manifest = _build_manifest(
        payloads,
        [2],
        quantization_config=quantization_config,
    )
    reordered = _build_manifest(
        list(reversed(payloads)),
        [2],
        quantization_config=quantization_config,
    )

    assert manifest == reordered
    assert manifest["schema_version"] == 1
    assert manifest["quantization"] == {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "weight_dtype": "float8_e4m3fn",
        "scale_dtype": "float32",
        "scale_format": "canonical",
    }
    assert len(manifest["entries"]) == 3 * 3 * 2
    assert {entry["family"] for entry in manifest["entries"]} == {"routed_expert"}
    assert all(all(".mlp.experts." in name for name in update_unit) for update_unit in manifest["routed_update_units"])

    pairs = {}
    for entry in manifest["entries"]:
        pairs.setdefault(entry["pair_id"], []).append(entry)
    assert len(pairs) == 3 * 3
    assert all(
        {entry["tensor_role"] for entry in pair} == {"weight", "scale"} and len(pair) == 2 for pair in pairs.values()
    )

    for layer in range(3):
        source_mesh = [[0, 1]] if layer < 2 else [[2, 3]]
        for component, source_recipe, destination_parameter in (
            ("gate", "expert_fc1_0", "w13_weight"),
            ("up", "expert_fc1_1", "w13_weight"),
            ("down", "expert_fc2", "w2_weight"),
        ):
            weight_name = f"model.layers.{layer}.mlp.experts." f"{component}_proj.weight"
            scale_name = f"{weight_name}_scale_inv"
            weight = _entry(manifest, weight_name)
            scale = _entry(manifest, scale_name)

            assert weight["pair_id"] == scale["pair_id"] == weight_name
            assert weight["tensor_role"] == "weight"
            assert scale["tensor_role"] == "scale"
            assert weight["dtype"] == "float8_e4m3fn"
            assert scale["dtype"] == "float32"
            assert weight["global_shape"] == [2, 256, 256]
            assert scale["global_shape"] == [2, 2, 2]
            assert weight["source"]["mesh"] == source_mesh
            assert scale["source"]["mesh"] == source_mesh
            assert weight["source"]["local_shape"] == [1, 256, 256]
            assert scale["source"]["local_shape"] == [1, 2, 2]
            assert scale["source"]["names_by_rank"] == (weight["source"]["names_by_rank"])
            assert weight["source"]["recipe"] == source_recipe
            assert scale["source"]["recipe"] == f"{source_recipe}_scale"
            assert weight["destination"]["local_shape"] == [1, 256, 256]
            assert scale["destination"]["local_shape"] == [1, 2, 2]
            assert weight["destination"]["recipe"] == f"expert_{component}"
            assert scale["destination"]["recipe"] == f"expert_{component}_scale"
            parameter_prefix = f"model.layers.{layer}.mlp.experts."
            assert weight["destination"]["parameter"] == (parameter_prefix + destination_parameter)
            assert scale["destination"]["parameter"] == (
                parameter_prefix + destination_parameter.replace("_weight", "_weight_scale_inv")
            )


def test_fp8_manifest_hash_covers_quantization_and_atomic_pair_metadata():
    manifest = _build_manifest(
        _fp8_payloads(),
        [2],
        quantization_config=_FP8_CONFIG,
    )

    for mutate in (
        lambda value: value["quantization"].__setitem__("scale_format", "ue8m0"),
        lambda value: value["entries"][0].__setitem__("pair_id", "injected-incomplete-pair"),
        lambda value: value["entries"][0].__setitem__("tensor_role", "scale"),
    ):
        tampered = deepcopy(manifest)
        mutate(tampered)
        assert nccl_m2n._manifest_digest(tampered) != manifest["manifest_hash"]


def test_fp8_manifest_supports_rollout_ep1_with_moe_tensor_parallel_sharding():
    manifest = _build_manifest(
        _fp8_payloads(),
        [2],
        quantization_config=_FP8_CONFIG,
        destination_ep_size=1,
    )

    gate = _entry(manifest, "model.layers.0.mlp.experts.gate_proj.weight")
    gate_scale = _entry(manifest, "model.layers.0.mlp.experts.gate_proj.weight_scale_inv")
    assert gate["destination"]["placements"][1] == {"type": "shard", "dim": 1}
    assert gate["destination"]["local_shape"] == [2, 128, 256]
    assert gate_scale["destination"]["placements"][1] == {"type": "shard", "dim": 1}
    assert gate_scale["destination"]["local_shape"] == [2, 1, 2]

    down = _entry(manifest, "model.layers.0.mlp.experts.down_proj.weight")
    down_scale = _entry(manifest, "model.layers.0.mlp.experts.down_proj.weight_scale_inv")
    assert down["destination"]["placements"][1] == {"type": "shard", "dim": 2}
    assert down["destination"]["local_shape"] == [2, 256, 128]
    assert down_scale["destination"]["placements"][1] == {"type": "shard", "dim": 2}
    assert down_scale["destination"]["local_shape"] == [2, 2, 1]


def test_fp8_pair_validation_rejects_missing_or_mismatched_scale():
    manifest = _build_manifest(
        _fp8_payloads(),
        [2],
        quantization_config=_FP8_CONFIG,
    )
    weight = next(entry for entry in manifest["entries"] if entry["tensor_role"] == "weight")
    entries_without_scale = [
        entry
        for entry in manifest["entries"]
        if not (entry["pair_id"] == weight["pair_id"] and entry["tensor_role"] == "scale")
    ]

    with pytest.raises(ValueError, match=r"(?i)(incomplete|pair|scale)"):
        nccl_m2n._validate_fp8_pairs(entries_without_scale)

    entries_with_bad_scale = deepcopy(manifest["entries"])
    scale = next(
        entry
        for entry in entries_with_bad_scale
        if entry["pair_id"] == weight["pair_id"] and entry["tensor_role"] == "scale"
    )
    scale["source"]["recipe"] = "wrong_scale_recipe"
    with pytest.raises(ValueError, match=r"(?i)(inconsistent|pair|scale)"):
        nccl_m2n._validate_fp8_pairs(entries_with_bad_scale)

    entries_without_projection = [entry for entry in manifest["entries"] if entry["pair_id"] != weight["pair_id"]]
    with pytest.raises(
        ValueError,
        match=r"(?i)(atomic|complete|gate|up|down|projection)",
    ):
        nccl_m2n._validate_fp8_pairs(entries_without_projection)


def test_fp8_partial_fc1_atomic_unit_drops_the_whole_expert_module():
    payloads = _fp8_payloads()
    blocked_name = _expert_name(0, 1, 0)
    unsupported_peer = f"{blocked_name}.unsupported_peer"
    payloads[0]["update_units"] = [
        [blocked_name, unsupported_peer] if unit == [blocked_name] else unit for unit in payloads[0]["update_units"]
    ]

    manifest = _build_manifest(
        payloads,
        [2],
        quantization_config=_FP8_CONFIG,
    )
    names = {entry["name"] for entry in manifest["entries"]}
    routed_names = {name for update_unit in manifest["routed_update_units"] for name in update_unit}

    assert not any(name.startswith("model.layers.0.mlp.experts.") for name in names)
    assert blocked_name not in routed_names
    assert {
        f"model.layers.1.mlp.experts.{component}_proj.{suffix}"
        for component in ("gate", "up", "down")
        for suffix in ("weight", "weight_scale_inv")
    }.issubset(names)


@pytest.mark.parametrize(
    ("quantization_config", "message"),
    [
        (
            {**_FP8_CONFIG, "quant_method": "mxfp8"},
            r"(?i)(quant|fp8)",
        ),
        (
            {**_FP8_CONFIG, "fmt": "e5m2"},
            r"(?i)(format|fmt|e4m3)",
        ),
        (
            {**_FP8_CONFIG, "activation_scheme": "static"},
            r"(?i)(activation|dynamic)",
        ),
        (
            {**_FP8_CONFIG, "weight_block_size": [64, 128]},
            r"(?i)(block|128)",
        ),
        (
            {**_FP8_CONFIG, "scale_fmt": "ue8m0"},
            r"(?i)(scale|canonical|fp32)",
        ),
        (
            {key: value for key, value in _FP8_CONFIG.items() if key != "weight_block_size"},
            r"(?i)(block|128)",
        ),
    ],
)
def test_fp8_manifest_rejects_unsupported_quantization_configs(
    quantization_config,
    message,
):
    with pytest.raises(ValueError, match=message):
        _build_manifest(
            _fp8_payloads(),
            [2],
            quantization_config=quantization_config,
        )


def test_canonical_fp8_quantizer_preserves_expert_axes_and_fp32_scale_grid():
    weight = (
        torch.arange(
            2 * 256 * 256,
            dtype=torch.float32,
        )
        .reshape(2, 256, 256)
        .to(torch.bfloat16)
    )
    expected_qweight = torch.zeros(
        512,
        256,
        dtype=torch.float8_e4m3fn,
    )
    expected_scale = torch.arange(
        8,
        dtype=torch.float32,
    ).reshape(4, 2)
    cast = Mock(return_value=(expected_qweight, expected_scale))

    with (
        patch.object(nccl_m2n, "per_block_cast_to_fp8", cast),
        patch.object(
            nccl_m2n,
            "blockwise_cast_to_fp8_triton",
            side_effect=AssertionError("unexpected Triton fallback"),
        ),
    ):
        qweight, scale = nccl_m2n._quantize_canonical_block_fp8(weight)

    cast.assert_called_once()
    torch.testing.assert_close(
        cast.call_args.args[0],
        weight.reshape(512, 256),
    )
    assert qweight.shape == weight.shape
    assert qweight.dtype == torch.float8_e4m3fn
    assert qweight.is_contiguous()
    assert scale.shape == (2, 2, 2)
    assert scale.dtype == torch.float32
    assert scale.is_contiguous()
    torch.testing.assert_close(scale, expected_scale.reshape(2, 2, 2))


def test_fp8_source_pair_is_quantized_once_per_batch_and_never_reused():
    manifest = _build_manifest(
        _fp8_payloads(),
        [2],
        quantization_config=_FP8_CONFIG,
    )
    pair_id = "model.layers.0.mlp.experts.gate_proj.weight"
    pair_entries = [entry for entry in manifest["entries"] if entry["pair_id"] == pair_id]
    assert [entry["tensor_role"] for entry in pair_entries] == [
        "weight",
        "scale",
    ]

    source_name = pair_entries[0]["source"]["names_by_rank"]["0"][0]
    fused_fc1 = torch.empty(512, 256, dtype=torch.bfloat16)
    fused_fc1[:256].fill_(1)
    fused_fc1[256:].fill_(9)

    updater = object.__new__(UpdateWeightFromNcclM2N)
    updater._m2n_manifest = {"entries": pair_entries}
    updater._m2n_pg = object()
    updater._m2n_comm_ptr = 123
    updater._m2n_comm_rank = 0
    updater._m2n_local_tensors = {source_name: fused_fc1}
    updater._m2n_fp8_pair_cache = {}

    quantized_inputs = []

    def quantize(logical):
        quantized_inputs.append(logical.clone())
        marker = float(logical[0, 0, 0])
        return (
            torch.full(
                logical.shape,
                marker,
                dtype=torch.float8_e4m3fn,
            ),
            torch.full((1, 2, 2), marker, dtype=torch.float32),
        )

    m2n = SimpleNamespace(
        Replicate=Mock(return_value=("replicate",)),
        Shard=Mock(side_effect=lambda dim: ("shard", dim)),
        reshard=Mock(),
    )
    stream = Mock()
    with (
        patch.object(nccl_m2n, "_nccl_rl", return_value=m2n),
        patch.object(
            nccl_m2n,
            "_quantize_canonical_block_fp8",
            side_effect=quantize,
        ),
        patch.object(
            nccl_m2n.torch.cuda,
            "current_stream",
            return_value=stream,
        ),
    ):
        updater._run_m2n_batch()
        assert updater._m2n_fp8_pair_cache == {}

        fused_fc1[:256].fill_(2)
        updater._run_m2n_batch()
        assert updater._m2n_fp8_pair_cache == {}

        m2n.reshard.side_effect = RuntimeError("injected transfer failure")
        with pytest.raises(RuntimeError, match="injected transfer failure"):
            updater._run_m2n_batch()
        assert updater._m2n_fp8_pair_cache == {}

    assert len(quantized_inputs) == 3
    assert torch.all(quantized_inputs[0] == 1)
    assert torch.all(quantized_inputs[1] == 2)
    assert torch.all(quantized_inputs[2] == 2)
    assert m2n.reshard.call_count == 5
    first_weight = m2n.reshard.call_args_list[0].args[0]
    first_scale = m2n.reshard.call_args_list[1].args[0]
    second_weight = m2n.reshard.call_args_list[2].args[0]
    second_scale = m2n.reshard.call_args_list[3].args[0]
    assert first_weight.dtype == torch.float8_e4m3fn
    assert first_scale.dtype == torch.float32
    assert torch.all(first_weight == 1)
    assert torch.all(first_scale == 1)
    assert torch.all(second_weight == 2)
    assert torch.all(second_scale == 2)
    assert stream.synchronize.call_count == 4


def test_partial_atomic_unit_stays_entirely_on_broadcast():
    payloads = deepcopy(_payloads())
    for payload in payloads:
        if payload["topology"]["pp_rank"] == 0:
            payload["update_units"] = [
                [_DENSE_FC1, _UNMAPPED],
                *[unit for unit in payload["update_units"] if unit not in ([_DENSE_FC1], [_UNMAPPED])],
            ]

    manifest = _build_manifest(payloads, [4] * 8)
    entry_names = {entry["name"] for entry in manifest["entries"]}

    assert "model.layers.0.mlp.gate_proj.weight" not in entry_names
    assert "model.layers.0.mlp.up_proj.weight" not in entry_names
    assert "model.layers.0.mlp.down_proj.weight" in entry_names
    assert [_DENSE_FC1, _UNMAPPED] not in manifest["routed_update_units"]
    assert "residual_update_units" not in manifest


def test_coalesced_expert_entry_does_not_leave_peer_units_marked_routed():
    payloads = _payloads()
    blocked_name = _expert_name(0, 1, 0)
    unsupported_scale = f"{blocked_name}.scale_inv"
    owner = payloads[0]
    owner["update_units"] = [
        [blocked_name, unsupported_scale] if unit == [blocked_name] else unit for unit in owner["update_units"]
    ]

    manifest = _build_manifest(payloads, [4] * 8)
    entry_names = {entry["name"] for entry in manifest["entries"]}
    routed_names = {name for update_unit in manifest["routed_update_units"] for name in update_unit}
    layer_fc1_names = {_expert_name(0, 1, expert_id) for expert_id in range(4 * _EXPERTS_PER_RANK)}

    assert "model.layers.0.mlp.experts.gate_proj.weight" not in entry_names
    assert "model.layers.0.mlp.experts.up_proj.weight" not in entry_names
    assert routed_names.isdisjoint(layer_fc1_names)
    assert "model.layers.0.mlp.experts.down_proj.weight" in entry_names
    assert "model.layers.1.mlp.experts.gate_proj.weight" in entry_names


def test_m2n_group_names_are_unique_per_connection():
    first = _new_m2n_group_name()
    second = _new_m2n_group_name()

    assert first.startswith("miles-m2n-")
    assert second.startswith("miles-m2n-")
    assert first != second


def test_prepare_failure_is_propagated_before_the_phase_barrier():
    updater = object.__new__(UpdateWeightFromNcclM2N)
    updater._connection_stale = False

    with (
        patch.object(
            nccl_m2n.UpdateWeightFromDistributed,
            "_pause_and_prepare_engines",
            side_effect=RuntimeError("injected prepare failure"),
        ),
        patch.object(
            nccl_m2n,
            "_collect_errors",
            return_value=["trainer rank 0 rollout prepare: " "RuntimeError: injected prepare failure"],
        ) as collect,
        patch.object(nccl_m2n.dist, "get_rank", return_value=0),
        pytest.raises(RuntimeError, match="injected prepare failure"),
    ):
        updater._pause_and_prepare_engines()

    collect.assert_called_once()
    assert updater._connection_stale is True


@pytest.mark.parametrize(
    ("rollout_gpus", "engine_gpus", "engine_count", "counts"),
    [
        (2, 2, 1, None),
        (2, 1, 2, [1, 1]),
        (32, 4, 8, [4] * 8),
    ],
)
def test_engine_gpu_counts_are_derived_from_rollout_configuration(
    rollout_gpus,
    engine_gpus,
    engine_count,
    counts,
):
    args = SimpleNamespace(
        rollout_num_gpus=rollout_gpus,
        rollout_num_gpus_per_engine=engine_gpus,
    )

    assert nccl_m2n._validated_engine_gpu_counts(args, engine_count, counts) == [engine_gpus] * engine_count


@pytest.mark.parametrize(
    ("engine_count", "counts", "message"),
    [
        (2, [2, 2], "expected 1 rollout engine handles"),
        (1, [2, 2], "one GPU count per rollout engine"),
        (1, [1], "homogeneous 2-GPU rollout engines"),
    ],
)
def test_connect_rejects_engine_layout_mismatches_before_group_setup(engine_count, counts, message):
    updater = object.__new__(UpdateWeightFromNcclM2N)
    updater.args = SimpleNamespace(
        rollout_num_gpus=2,
        rollout_num_gpus_per_engine=2,
    )

    with pytest.raises(ValueError, match=message):
        updater.connect_rollout_engines(
            [Mock()] * engine_count,
            Mock(),
            engine_gpu_counts=counts,
        )


def test_residual_failure_is_deferred_until_trainer_collectives_drain():
    acquire = Mock(return_value="acquire-ref")
    release = Mock(return_value="release-ref")
    updater = object.__new__(UpdateWeightFromNcclM2N)
    updater._deferred_update_error = None
    updater._connection_stale = False
    updater.rollout_engine_lock = SimpleNamespace(
        acquire=SimpleNamespace(remote=acquire),
        release=SimpleNamespace(remote=release),
    )
    updater._group_name = "miles-pp-0"
    updater._model_update_groups = object()
    updater.rollout_engines = []
    updater.update_weight_metrics = {"m2n_broadcast_bytes": 0.0}
    tensor = Mock()
    tensor.numel.return_value = 1
    tensor.element_size.return_value = 2
    converted = [("weight", tensor)]

    with (
        patch.object(nccl_m2n.dist, "get_rank", return_value=0),
        patch.object(nccl_m2n.ray, "get", side_effect=[True, None]),
        patch.object(
            nccl_m2n,
            "update_weights_from_distributed",
            side_effect=RuntimeError("injected residual failure"),
        ),
    ):
        updater._update_weight_implementation(converted)

    assert converted == []
    assert "injected residual failure" in updater._deferred_update_error
    release.assert_called_once()
    assert updater.update_weight_metrics["m2n_broadcast_bytes"] == 0.0

    with (
        patch.object(
            nccl_m2n,
            "_collect_errors",
            side_effect=lambda error: [error] if error else [],
        ),
        pytest.raises(RuntimeError, match="injected residual failure"),
    ):
        updater._raise_deferred_update_errors()

    assert updater._connection_stale is True


def test_teardown_launches_remote_before_destroying_local_and_waiting():
    events = []

    def launch_remote(group_name):
        events.append(f"launch:{group_name}")
        return f"ref:{group_name}"

    def wait_remote(ref):
        events.append(f"wait:{ref.removeprefix('ref:')}")
        return {"success": True, "message": "destroyed"}

    engine = SimpleNamespace(destroy_weights_update_group=SimpleNamespace(remote=launch_remote))
    process_group = object()
    updater = object.__new__(UpdateWeightFromNcclM2N)
    updater.rollout_engines = [engine]
    updater._m2n_manifest = {"entries": []}
    updater._m2n_group_name = "miles-m2n-old"
    updater._m2n_routed_units = set()
    updater._group_name = "miles-pp-0"
    updater._model_update_groups = process_group
    updater._residual_pp_rank = 0
    updater._residual_group_started = True
    updater._connection_stale = False
    updater._teardown_local_m2n = Mock(side_effect=lambda: events.append("local:miles-m2n-old"))

    with (
        patch.object(nccl_m2n.dist, "get_rank", return_value=0),
        patch.object(nccl_m2n.dist, "get_world_size", return_value=1),
        patch.object(
            nccl_m2n.dist,
            "all_gather_object",
            side_effect=lambda output, value, group: output.__setitem__(0, value),
        ),
        patch.object(nccl_m2n, "get_gloo_group", return_value=object()),
        patch.object(nccl_m2n.ray, "get", side_effect=wait_remote),
        patch.object(
            nccl_m2n.dist,
            "destroy_process_group",
            side_effect=lambda group: events.append("local:miles-pp-0"),
        ),
        patch.object(UpdateWeightFromNcclM2N, "_is_source", True),
    ):
        updater._disconnect_existing()

    assert events == [
        "launch:miles-m2n-old",
        "local:miles-m2n-old",
        "wait:miles-m2n-old",
        "launch:miles-pp-0",
        "local:miles-pp-0",
        "wait:miles-pp-0",
    ]


def test_mixed_teardown_success_is_idempotently_retryable():
    destroy_remote_a = Mock(return_value="destroy-ref-a")
    destroy_remote_b = Mock(return_value="destroy-ref-b")
    engines = [
        SimpleNamespace(destroy_weights_update_group=SimpleNamespace(remote=destroy_remote_a)),
        SimpleNamespace(destroy_weights_update_group=SimpleNamespace(remote=destroy_remote_b)),
    ]
    updater = object.__new__(UpdateWeightFromNcclM2N)
    updater.rollout_engines = engines
    updater._m2n_manifest = {"entries": []}
    updater._m2n_group_name = "miles-m2n-old"
    updater._m2n_routed_units = {("weight",)}
    updater._model_update_groups = None
    updater._connection_stale = False
    updater._teardown_local_m2n = Mock()

    with (
        patch.object(nccl_m2n.dist, "get_rank", return_value=0),
        patch.object(nccl_m2n.dist, "get_world_size", return_value=1),
        patch.object(
            nccl_m2n.dist,
            "all_gather_object",
            side_effect=lambda output, value, group: output.__setitem__(0, value),
        ),
        patch.object(nccl_m2n.dist, "barrier"),
        patch.object(nccl_m2n, "get_gloo_group", return_value=object()),
        patch.object(
            nccl_m2n.ray,
            "get",
            side_effect=[
                {"success": True, "message": "destroyed"},
                {"success": False, "message": "injected teardown failure"},
                {"success": True, "message": "already absent"},
                {"success": True, "message": "destroyed"},
            ],
        ),
        patch.object(
            UpdateWeightFromNcclM2N,
            "_is_source",
            False,
        ),
    ):
        with pytest.raises(RuntimeError, match="injected teardown failure"):
            updater._disconnect_existing()

        assert updater._m2n_group_name == "miles-m2n-old"
        assert updater._m2n_manifest == {"entries": []}
        assert updater._connection_stale is True

        updater._disconnect_existing()

    assert destroy_remote_a.call_args_list == [
        call("miles-m2n-old"),
        call("miles-m2n-old"),
    ]
    assert destroy_remote_b.call_args_list == [
        call("miles-m2n-old"),
        call("miles-m2n-old"),
    ]
    assert updater._m2n_group_name is None
    assert updater._m2n_manifest is None
    assert updater._m2n_routed_units == set()


def test_local_teardown_failure_is_propagated_collectively_and_retryable():
    updater = object.__new__(UpdateWeightFromNcclM2N)
    updater.rollout_engines = []
    updater._m2n_manifest = {"entries": []}
    updater._m2n_group_name = "miles-m2n-old"
    updater._m2n_routed_units = set()
    updater._model_update_groups = None
    updater._connection_stale = False
    updater._teardown_local_m2n = Mock(side_effect=[RuntimeError("injected local teardown failure"), None])

    with (
        patch.object(nccl_m2n.dist, "get_rank", return_value=0),
        patch.object(nccl_m2n.dist, "get_world_size", return_value=1),
        patch.object(
            nccl_m2n.dist,
            "all_gather_object",
            side_effect=lambda output, value, group: output.__setitem__(0, value),
        ) as gather,
        patch.object(nccl_m2n.dist, "barrier"),
        patch.object(nccl_m2n, "get_gloo_group", return_value=object()),
        patch.object(UpdateWeightFromNcclM2N, "_is_source", False),
    ):
        with pytest.raises(RuntimeError, match="injected local teardown failure"):
            updater._disconnect_existing()

        assert gather.call_count == 1
        assert updater._m2n_group_name == "miles-m2n-old"

        updater._disconnect_existing()

    assert updater._m2n_group_name is None
    assert updater._m2n_manifest is None


def test_unreachable_retired_engine_does_not_block_replacement():
    destroy_remote = Mock(return_value="destroy-ref")
    retired_engine = SimpleNamespace(destroy_weights_update_group=SimpleNamespace(remote=destroy_remote))
    updater = object.__new__(UpdateWeightFromNcclM2N)
    updater.rollout_engines = [retired_engine]
    updater._m2n_manifest = {"entries": []}
    updater._m2n_group_name = "miles-m2n-retired"
    updater._m2n_routed_units = set()
    updater._group_name = "miles-pp-0"
    updater._model_update_groups = object()
    updater._connection_stale = True
    updater._teardown_local_m2n = Mock()

    with (
        patch.object(nccl_m2n.dist, "get_rank", return_value=0),
        patch.object(nccl_m2n.dist, "get_world_size", return_value=1),
        patch.object(
            nccl_m2n.dist,
            "all_gather_object",
            side_effect=lambda output, value, group: output.__setitem__(0, value),
        ),
        patch.object(nccl_m2n.dist, "barrier"),
        patch.object(nccl_m2n, "get_gloo_group", return_value=object()),
        patch.object(
            nccl_m2n.ray,
            "get",
            side_effect=nccl_m2n.requests.exceptions.ConnectionError("retired engine is offline"),
        ),
        patch.object(nccl_m2n.dist, "destroy_process_group") as destroy_group,
        patch.object(UpdateWeightFromNcclM2N, "_is_source", True),
    ):
        updater._disconnect_existing(replacement_engines=[])

    assert destroy_remote.call_args_list == [
        call("miles-m2n-retired"),
        call("miles-pp-0"),
    ]
    destroy_group.assert_called_once()
    assert updater._model_update_groups is None
    assert updater._m2n_group_name is None
    assert updater._m2n_manifest is None


def test_failed_residual_teardown_retains_connection_state_for_retry():
    destroy_remote = Mock(return_value="destroy-ref")
    engine = SimpleNamespace(destroy_weights_update_group=SimpleNamespace(remote=destroy_remote))
    process_group = object()
    updater = object.__new__(UpdateWeightFromNcclM2N)
    updater.rollout_engines = [engine]
    updater._m2n_manifest = {"entries": []}
    updater._m2n_group_name = "miles-m2n-old"
    updater._m2n_routed_units = set()
    updater._group_name = "miles-pp-0"
    updater._model_update_groups = process_group
    updater._residual_group_started = True
    updater._connection_stale = False
    updater._teardown_local_m2n = Mock()

    with (
        patch.object(nccl_m2n.dist, "get_rank", return_value=0),
        patch.object(nccl_m2n.dist, "get_world_size", return_value=1),
        patch.object(
            nccl_m2n.dist,
            "all_gather_object",
            side_effect=lambda output, value, group: output.__setitem__(0, value),
        ),
        patch.object(nccl_m2n, "get_gloo_group", return_value=object()),
        patch.object(
            nccl_m2n.ray,
            "get",
            side_effect=[
                {"success": True, "message": "M2N destroyed"},
                {"success": False, "message": "residual destroy failed"},
                {"success": True, "message": "M2N already absent"},
                {"success": True, "message": "residual destroyed"},
            ],
        ),
        patch.object(nccl_m2n.dist, "destroy_process_group") as destroy_group,
        patch.object(UpdateWeightFromNcclM2N, "_is_source", True),
    ):
        with pytest.raises(RuntimeError, match="residual destroy failed"):
            updater._disconnect_existing()

        assert updater._m2n_group_name == "miles-m2n-old"
        assert updater._m2n_manifest == {"entries": []}
        assert updater._model_update_groups is None

        updater._disconnect_existing()

    assert destroy_remote.call_args_list == [
        call("miles-m2n-old"),
        call("miles-pp-0"),
        call("miles-m2n-old"),
        call("miles-pp-0"),
    ]
    destroy_group.assert_called_once_with(process_group)
    assert updater._m2n_group_name is None
    assert updater._m2n_manifest is None
    assert updater._model_update_groups is None
