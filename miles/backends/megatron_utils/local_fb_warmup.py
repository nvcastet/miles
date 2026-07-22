"""One-time, pipeline-free Megatron forward/backward warmup."""

import logging
from contextlib import nullcontext

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.distributed.finalize_model_grads import reset_model_temporary_tensors
from megatron.core.fp8_utils import get_fp8_recipe
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.tensor_parallel.random import _fork_rng
from megatron.core.transformer.moe.moe_utils import clear_aux_losses_tracker
from megatron.core.utils import get_attr_wrapped_model, get_model_config
from transformer_engine.pytorch.graph import restore_fp8_tensors, save_fp8_tensors

from miles.backends.megatron_utils.parallel import get_packed_seq_params
from miles.backends.training_utils.data import get_batch
from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)

_WARMED_ATTR = "_miles_local_fb_warmed"
_BATCH_KEYS = [
    "tokens",
    "multimodal_train_inputs",
    "total_lengths",
    "response_lengths",
    "loss_masks",
    "max_seq_lens",
    "witness_ids",
]


def _zero_grad(model, optimizer):
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    if optimizer is not None:
        optimizer.zero_grad()


def _clear_deferred_embedding_buffers(config, model_chunk):
    if not getattr(config, "defer_embedding_wgrad_compute", False):
        return

    output_model = get_attr_wrapped_model(model_chunk, "post_process", return_model_obj=True)
    if output_model.post_process:
        output_model.embedding_activation_buffer.clear()
        output_model.grad_output_buffer.clear()


def _clear_loss_trackers(args):
    clear_aux_losses_tracker()
    if args.enable_mtp_training:
        from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

        if "values" in MTPLossLoggingHelper.tracker:
            MTPLossLoggingHelper.clean_loss_in_tracker()

    try:
        from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossLoggingHelper
    except ImportError:
        pass
    else:
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()


def _tensor_leaves(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_leaves(item)


def _zero_backward_seed(value):
    tensors = [tensor for tensor in _tensor_leaves(value) if tensor.requires_grad]
    if not tensors:
        raise RuntimeError("The local F/B warmup model output has no differentiable tensors")
    loss = tensors[0].float().sum() * 0.0
    for tensor in tensors[1:]:
        loss = loss + tensor.float().sum() * 0.0
    return loss


def _get_local_input_shape(tokens, config):
    # Miles uses variable sequence lengths, so MCore's pipeline shape helper returns ().
    # The batch tokens are already CP-local; only sequence parallelism further shards them.
    seq_length = tokens.shape[-1]
    if config.sequence_parallel:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        if seq_length % tp_size != 0:
            raise RuntimeError(f"Local sequence length {seq_length} is not divisible by TP size {tp_size}")
        seq_length //= tp_size
    if getattr(config, "dsv4_mode", False):
        return (seq_length, tokens.shape[0], config.dsv4_hc_mult, config.hidden_size)
    return (seq_length, tokens.shape[0], config.hidden_size)


def run_local_fb_warmup(args, rollout_id, model, optimizer, data_iterator):
    """Run one local forward/backward on every pipeline stage."""
    if len(model) != 1 or len(data_iterator) != 1:
        raise RuntimeError("The local F/B warmup does not support virtual pipeline stages")

    model_chunk = model[0]
    if getattr(model_chunk, _WARMED_ATTR, False):
        return

    config = get_model_config(model_chunk)
    gloo_group = get_gloo_group()
    if dist.get_rank() == 0:
        logger.info("Starting PP-free local F/B warmup before rollout %s", rollout_id)
    dist.barrier(group=gloo_group)

    training = model_chunk.training
    local_input = None
    pre_hook_disabled = False
    buffer_state = [(buffer, buffer.detach().clone()) for _, buffer in model_chunk.named_buffers()]
    fp8_state = (
        save_fp8_tensors([model_chunk], get_fp8_recipe(config))
        if getattr(config, "fp8", None) is not None
        else None
    )

    _zero_grad(model, optimizer)
    try:
        if args.use_distributed_optimizer and args.overlap_param_gather:
            model_chunk.disable_forward_pre_hook(param_sync=False)
            pre_hook_disabled = True

        batch = get_batch(
            data_iterator[0],
            _BATCH_KEYS,
            args.data_pad_size_multiplier,
            args.qkv_format,
            allgather_cp=args.allgather_cp,
        )
        tokens = batch["tokens"]

        if not get_attr_wrapped_model(model_chunk, "pre_process"):
            local_input = torch.zeros(
                _get_local_input_shape(tokens, config),
                dtype=config.pipeline_dtype or config.params_dtype,
                device=torch.cuda.current_device(),
                requires_grad=True,
            )

        forward_kwargs = {
            "input_ids": tokens,
            "position_ids": None,
            "attention_mask": None,
            "labels": None,
            "packed_seq_params": get_packed_seq_params(batch, args),
            "loss_mask": batch["full_loss_masks"],
        }
        if args.enable_witness:
            forward_kwargs["witness_ids"] = batch["witness_ids"]
        if args.enable_mtp_training:
            forward_kwargs["mtp_kwargs"] = {"mtp_labels": tokens}
        if batch["multimodal_train_inputs"] is not None:
            forward_kwargs.update(batch["multimodal_train_inputs"])

        model_chunk.train()
        if hasattr(model_chunk, "set_is_first_microbatch"):
            model_chunk.set_is_first_microbatch()
        if local_input is not None:
            get_attr_wrapped_model(model_chunk, "set_input_tensor")([local_input])

        pp_group = mpu.get_pipeline_model_parallel_group()
        p2p = P2PCommunicator(pp_group=pp_group, config=config)
        is_pp_first_stage = mpu.is_pipeline_first_stage()
        is_pp_last_stage = mpu.is_pipeline_last_stage()
        p2p_buffer_shape = _get_local_input_shape(tokens, config)
        p2p_buffer = torch.empty(p2p_buffer_shape, dtype=config.pipeline_dtype, device=torch.cuda.current_device())

        autocast = (
            torch.autocast("cuda", dtype=config.autocast_dtype) if config.enable_autocast else nullcontext()
        )
        with _fork_rng(), torch.enable_grad(), model_chunk.no_sync(), autocast:
            # Warmup JIT kernels concurrently without PP serialization
            output = model_chunk(**forward_kwargs)
            _zero_backward_seed(output).backward()
        # Setup transport channels (NCCL lazily creates them)
        _, _ = p2p.send_forward_backward_recv_forward_backward(
            # Avoid PP-group wraparound at the boundaries.
            output_tensor=None if is_pp_last_stage else p2p_buffer,
            input_tensor_grad=None if is_pp_first_stage else p2p_buffer,
            recv_prev=not is_pp_first_stage,
            recv_next=not is_pp_last_stage,
            tensor_shape=p2p_buffer_shape,
        )
        torch.cuda.synchronize()
    finally:
        if fp8_state is not None:
            restore_fp8_tensors([model_chunk], fp8_state)
        for iterator in data_iterator:
            iterator.reset()
        model_chunk.train(training)
        if local_input is not None:
            get_attr_wrapped_model(model_chunk, "set_input_tensor")([None])
        _clear_deferred_embedding_buffers(config, model_chunk)
        _clear_loss_trackers(args)
        reset_model_temporary_tensors(config, model)
        _zero_grad(model, optimizer)
        for buffer, saved_buffer in buffer_state:
            buffer.detach().copy_(saved_buffer)
        if pre_hook_disabled:
            model_chunk.enable_forward_pre_hook()

    torch.cuda.synchronize()
    dist.barrier(group=gloo_group)
    setattr(model_chunk, _WARMED_ATTR, True)
    if dist.get_rank() == 0:
        logger.info("PP-free local F/B warmup complete on every stage")
