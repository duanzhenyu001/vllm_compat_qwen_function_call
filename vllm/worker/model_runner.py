import dataclasses
import gc
import time
import warnings
import weakref
from collections import defaultdict
from typing import (TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Set,
                    Tuple, Type, TypeVar, Union)

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

try:
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.decode import CUDAGraphBatchDecodeWithPagedKVCacheWrapper
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024
except ImportError:
    BatchDecodeWithPagedKVCacheWrapper = None
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper = None
    BatchPrefillWithPagedKVCacheWrapper = None
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 0

from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, MultiModalConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig)
from vllm.distributed import get_pp_group
from vllm.distributed.parallel_state import graph_capture
from vllm.inputs import INPUT_REGISTRY
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.models.interfaces import (supports_lora,
                                                   supports_vision)
from vllm.model_executor.models.utils import set_cpu_offload_max_bytes
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensors,
                             MultiModalInputs)
from vllm.prompt_adapter.layers import PromptAdapterMapping
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.prompt_adapter.worker_manager import (
    LRUCacheWorkerPromptAdapterManager)
from vllm.sampling_params import SamplingParams
from vllm.sequence import (IntermediateTensors, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.utils import (CudaMemoryProfiler, get_kv_cache_torch_dtype, is_hip,
                        is_pin_memory_available)
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase, ModelRunnerInputBuilderBase,
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)

_PAD_SLOT_ID = -1
LORA_WARMUP_RANK = 8
_BATCH_SIZE_ALIGNMENT = 8
# Capture graphs for token size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [
    _BATCH_SIZE_ALIGNMENT * i for i in range(1, 33)
]
_NUM_WARMUP_ITERS = 2

TModelInputForGPU = TypeVar('TModelInputForGPU', bound="ModelInputForGPU")


@dataclasses.dataclass(frozen=True)
class ModelInputForGPU(ModelRunnerInputBase):
    """
    This base class contains metadata needed for the base model forward pass
    but not metadata for possible additional steps, e.g., sampling. Model
    runners that run additional steps should subclass this method to add
    additional fields.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    lora_mapping: Optional["LoRAMapping"] = None
    lora_requests: Optional[Set[LoRARequest]] = None
    attn_metadata: Optional["AttentionMetadata"] = None
    prompt_adapter_mapping: Optional[PromptAdapterMapping] = None
    prompt_adapter_requests: Optional[Set[PromptAdapterRequest]] = None
    multi_modal_kwargs: Optional[Mapping[str, BatchedTensors]] = None
    request_ids_to_seq_ids: Optional[Dict[str, List[int]]] = None
    finished_requests_ids: Optional[List[str]] = None
    virtual_engine: int = 0

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "prompt_adapter_mapping": self.prompt_adapter_mapping,
            "prompt_adapter_requests": self.prompt_adapter_requests,
            "virtual_engine": self.virtual_engine,
            "request_ids_to_seq_ids": self.request_ids_to_seq_ids,
            "finished_requests_ids": self.finished_requests_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type[TModelInputForGPU],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> TModelInputForGPU:
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


@dataclasses.dataclass(frozen=True)
class ModelInputForGPUWithSamplingMetadata(ModelInputForGPU):
    """
    Used by the ModelRunner.
    """
    sampling_metadata: Optional["SamplingMetadata"] = None
    # Used for speculative decoding. We do not broadcast it because it is only
    # used by the driver worker.
    is_prompt: Optional[bool] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "prompt_adapter_mapping": self.prompt_adapter_mapping,
            "prompt_adapter_requests": self.prompt_adapter_requests,
            "virtual_engine": self.virtual_engine,
            "request_ids_to_seq_ids": self.request_ids_to_seq_ids,
            "finished_requests_ids": self.finished_requests_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForGPUWithSamplingMetadata":
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


class ModelInputForGPUBuilder(ModelRunnerInputBuilderBase[ModelInputForGPU]):
    """TBA"""

    def __init__(self,
                 runner: "GPUModelRunnerBase",
                 finished_requests_ids: Optional[List[str]] = None):
        super().__init__()
        self.runner = runner
        self.model_input_cls = self.runner._model_input_cls
        self.attn_backend = self.runner.attn_backend
        self.scheduler_config = self.runner.scheduler_config
        self.sliding_window = self.runner.sliding_window
        self.block_size = self.runner.block_size
        self.enable_lora = self.runner.lora_config is not None
        self.enable_prompt_adapter = (self.runner.prompt_adapter_config
                                      is not None)
        self.multi_modal_input_mapper = self.runner.multi_modal_input_mapper
        self.finished_requests_ids = finished_requests_ids
        self.decode_only = True

        # Common inputs.
        self.input_tokens: List[int] = []
        self.input_positions: List[int] = []
        self.seq_lens: List[int] = []
        self.query_lens: List[int] = []
        self.max_decode_seq_len: int = 0
        self.request_ids_to_seq_ids: Dict[str, List[int]] = defaultdict(list)

        # LoRA inputs.
        self.lora_index_mapping: List[int] = []
        self.lora_prompt_mapping: List[int] = []
        self.lora_requests: Set[LoRARequest] = set()

        # Prompt adapter inputs.
        self.prompt_adapter_index_mapping: List[int] = []
        self.prompt_adapter_prompt_mapping: List[int] = []
        self.prompt_adapter_requests: Set[PromptAdapterRequest] = set()

        # Multi-modal inputs.
        self.multi_modal_inputs_list: List[MultiModalInputs] = []

        # Attention metadata inputs.
        self.attn_metadata_builder = self.attn_backend.make_metadata_builder(
            self)

        # Engine/Model configurations.
        self.chunked_prefill_enabled = (
            self.scheduler_config is not None
            and self.scheduler_config.chunked_prefill_enabled)
        if self.sliding_window is not None:
            self.sliding_window_blocks = (
                self.sliding_window + self.block_size - 1) // self.block_size
            self.block_aligned_sliding_window = \
                self.sliding_window_blocks * self.block_size

    def _compute_len_for_sliding_window(self, seq_len: int):
        curr_sliding_window_blocks = 0
        sliding_seq_len = seq_len

        # TODO(sang): This is a hack to make sliding window work with
        # paged attn. We can remove it if we make paged attn kernel
        # to properly handle slinding window attn.
        if self.sliding_window is not None:
            curr_sliding_window_blocks = self.sliding_window_blocks
            if self.scheduler_config.use_v2_block_manager:
                # number of elements in last block
                suff_len = seq_len % self.block_size
                sliding_seq_len = min(
                    seq_len, self.block_aligned_sliding_window + suff_len)
                if suff_len > 0:
                    curr_sliding_window_blocks += 1
            else:
                sliding_seq_len = min(seq_len, self.sliding_window)
        return curr_sliding_window_blocks, sliding_seq_len

    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        seq_ids = list(seq_group_metadata.seq_data.keys())
        n_seqs = len(seq_ids)
        is_prompt = seq_group_metadata.is_prompt
        token_chunk_size = seq_group_metadata.token_chunk_size

        if is_prompt:
            assert n_seqs == 1
            self.decode_only = False

        # Mapping from request IDs to sequence IDs. Used for Jamba models
        # that manages the cache by itself.
        self.request_ids_to_seq_ids[seq_group_metadata.request_id] = []
        # The number of input tokens in each sequence.
        token_lens: List[int] = []
        # The number of tokens that are already computed.
        context_lens: List[int] = []
        # The current sliding window block for each sequence.
        curr_sliding_window_blocks: List[int] = []
        # The original sequence length (before applying sliding window)
        # for each sequence.
        orig_seq_lens: List[int] = []
        # The sequence length (may be capped to the sliding window).
        curr_seq_lens: List[int] = []
        for seq_id in seq_ids:
            seq_data = seq_group_metadata.seq_data[seq_id]
            self.request_ids_to_seq_ids[seq_group_metadata.request_id].append(
                seq_id)
            computed_block_nums = seq_group_metadata.computed_block_nums

            # Check if hit prefix cache (i.e., some blocks are already computed)
            # Note that prefix caching does not support sliding window.
            prefix_cache_hit = (computed_block_nums is not None
                                and len(computed_block_nums) > 0
                                and self.sliding_window is None and is_prompt)
            if self.chunked_prefill_enabled and prefix_cache_hit:
                raise RuntimeError(
                    "chunked prefill cannot be used with prefix caching now.")

            # Compute context length (the number of tokens that are
            # already computed) and sequence length (total number of tokens).
            seq_len = seq_data.get_len()
            if is_prompt:
                context_len = seq_data.get_num_computed_tokens()
            else:
                # get_num_computed_tokens is incorrect for spec decoding.
                # So, we should have a special logic here.
                # TODO(sang): Fix it.
                context_len = seq_len - 1
            seq_len = min(seq_len, context_len + token_chunk_size)

            # Compute tokens.
            if is_prompt:
                tokens = seq_data.get_token_ids()[context_len:seq_len]
            else:
                # Optimization. get_token_ids requires the entire copy of
                # tokens.
                tokens = [seq_data.get_last_token_id()]
            if prefix_cache_hit:
                assert computed_block_nums is not None
                context_len = len(computed_block_nums) * self.block_size
                tokens = tokens[context_len:]

            # These are seq_len/context_len capped to the sliding window.
            # They are passed to decode kernel.
            # We still need original seq_len/context_len to compute slot
            # mapping (and input position) below.
            if is_prompt:
                curr_sliding_window_block = 0
                sliding_seq_len = seq_len
                query_len = seq_len - context_len
            else:
                curr_sliding_window_block, sliding_seq_len = (
                    self._compute_len_for_sliding_window(seq_len))
                query_len = 1

            self.seq_lens.append(sliding_seq_len)
            if not is_prompt:
                self.max_decode_seq_len = max(self.max_decode_seq_len,
                                              sliding_seq_len)
            self.query_lens.append(query_len)
            self.input_tokens.extend(tokens)
            self.input_positions.extend(list(range(context_len, seq_len)))

            # Intermediate data of the current sequence group for
            # the attention metadata.
            token_lens.append(len(tokens))
            context_lens.append(context_len)
            curr_seq_lens.append(sliding_seq_len)
            curr_sliding_window_blocks.append(curr_sliding_window_block)
            orig_seq_lens.append(seq_len)

        # Update attention metadata. Note that input builder attributes
        # (self.xxx) include all added sequences, so we need to slice
        # the last n_seqs sequences.
        self.attn_metadata_builder.add_seq_group(
            seq_group_metadata, token_lens, orig_seq_lens, curr_seq_lens,
            self.query_lens[-n_seqs:], context_lens,
            curr_sliding_window_blocks, prefix_cache_hit,
            self.chunked_prefill_enabled)

        # LoRA data.
        if self.enable_lora:
            lora_id = seq_group_metadata.lora_int_id
            for query_len in self.query_lens[-n_seqs:]:
                if lora_id > 0:
                    self.lora_requests.add(seq_group_metadata.lora_request)
                self.lora_index_mapping += [lora_id] * query_len
                self.lora_prompt_mapping.extend(
                    [lora_id] *
                    (query_len if seq_group_metadata.sampling_params
                     and seq_group_metadata.sampling_params.prompt_logprobs
                     is not None else 1))

        # Prompt adapter data. Note that when is_prompt=True,
        # we expect only one sequence in the group.
        if self.enable_prompt_adapter:
            prompt_adapter_id = seq_group_metadata.prompt_adapter_id
            if prompt_adapter_id > 0 and is_prompt:
                query_len = self.query_lens[-1]
                self.prompt_adapter_requests.add(
                    seq_group_metadata.prompt_adapter_request)

                num_tokens = seq_group_metadata.\
                    prompt_adapter_num_virtual_tokens
                pm = [prompt_adapter_id
                      ] * num_tokens + [0] * (query_len - num_tokens)
                self.prompt_adapter_index_mapping += pm
                self.prompt_adapter_prompt_mapping.extend(
                    [prompt_adapter_id] *
                    (query_len if seq_group_metadata.sampling_params
                     and seq_group_metadata.sampling_params.prompt_logprobs
                     else 1))

        # Multi-modal data.
        mm_data = seq_group_metadata.multi_modal_data
        if mm_data:
            mm_kwargs = self.multi_modal_input_mapper(mm_data)
            self.multi_modal_inputs_list.append(mm_kwargs)

    def build(self) -> ModelInputForGPU:
        if not self.input_tokens:
            return self.model_input_cls()

        batch_size = len(self.input_tokens)
        use_captured_graph = (
            self.decode_only and not self.runner.model_config.enforce_eager
            and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
            and self.max_decode_seq_len <= self.runner.max_seq_len_to_capture)

        # If cuda graph can be used, pad tensors accordingly.
        # See `capture_model` API for more details.
        # vLLM uses cuda graph only for decoding requests.
        cuda_graph_pad_size = -1
        if use_captured_graph:
            graph_batch_size = _get_graph_batch_size(batch_size)
            assert graph_batch_size >= batch_size
            cuda_graph_pad_size = graph_batch_size - batch_size
            batch_size = graph_batch_size

        # Tokens and positions.
        self.input_tokens.extend([0] * cuda_graph_pad_size)
        self.input_positions.extend([0] * cuda_graph_pad_size)
        input_tokens_tensor = torch.tensor(self.input_tokens,
                                           dtype=torch.long,
                                           device=self.runner.device)
        input_positions_tensor = torch.tensor(self.input_positions,
                                              dtype=torch.long,
                                              device=self.runner.device)

        # Sequence and query lengths.
        self.seq_lens.extend([1] * cuda_graph_pad_size)

        # Attention metadata.
        attn_metadata = self.attn_metadata_builder.build(
            self.runner, self.seq_lens, self.query_lens, cuda_graph_pad_size,
            batch_size)

        # LoRA data.
        if self.enable_lora:
            self.lora_index_mapping.extend([0] * cuda_graph_pad_size)
            lora_mapping = LoRAMapping(
                self.lora_index_mapping,
                self.lora_prompt_mapping,
            )
        else:
            lora_mapping = None

        # Prompt adapter data.
        if self.enable_prompt_adapter:
            self.prompt_adapter_index_mapping.extend([0] * cuda_graph_pad_size)
            prompt_adapter_mapping = PromptAdapterMapping(
                self.prompt_adapter_index_mapping,
                self.prompt_adapter_prompt_mapping,
            )
        else:
            prompt_adapter_mapping = None

        # Multi-modal data.
        multi_modal_kwargs = MultiModalInputs.batch(
            self.multi_modal_inputs_list, device=self.runner.device)

        return self.model_input_cls(
            input_tokens=input_tokens_tensor,
            input_positions=input_positions_tensor,
            attn_metadata=attn_metadata,
            seq_lens=self.seq_lens,
            query_lens=self.query_lens,
            lora_mapping=lora_mapping,
            lora_requests=self.lora_requests,
            multi_modal_kwargs=multi_modal_kwargs,
            request_ids_to_seq_ids=self.request_ids_to_seq_ids,
            finished_requests_ids=self.finished_requests_ids,
            prompt_adapter_mapping=prompt_adapter_mapping,
            prompt_adapter_requests=self.prompt_adapter_requests)


class GPUModelRunnerBase(ModelRunnerBase[TModelInputForGPU]):
    """
    Helper class for shared methods between GPU model runners.
    """
    _model_input_cls: Type[TModelInputForGPU]

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        multimodal_config: Optional[MultiModalConfig] = None,
        return_hidden_states: bool = False,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        self.prompt_adapter_config = prompt_adapter_config
        self.multimodal_config = multimodal_config
        self.return_hidden_states = return_hidden_states

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_seq_len_to_capture = self.model_config.max_seq_len_to_capture

        self.graph_runners: List[Dict[int, CUDAGraphRunner]] = [
            {} for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.graph_memory_pool: Optional[Tuple[
            int, int]] = None  # Set during graph capture.

        self.has_seqlen_agnostic = model_config.contains_seqlen_agnostic_layers(
            parallel_config)

        # When using CUDA graph, the input block tables must be padded to
        # max_seq_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE), self.get_max_block_per_batch()),
            dtype=np.int32)
        num_attn_heads = self.model_config.get_num_attention_heads(
            self.parallel_config)
        self.attn_backend = get_attn_backend(
            num_attn_heads,
            self.model_config.get_head_size(),
            self.model_config.get_num_kv_heads(self.parallel_config),
            self.model_config.get_sliding_window(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
        ) if num_attn_heads else None

        # Multi-modal data support
        self.multi_modal_input_mapper = MULTIMODAL_REGISTRY \
            .create_input_mapper(self.model_config)

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        # Set after load_model.
        self.lora_manager: Optional[LRUCacheWorkerLoRAManager] = None
        self.prompt_adapter_manager: LRUCacheWorkerPromptAdapterManager = None

        self.flashinfer_decode_workspace_buffer = None
        self.flashinfer_decode_wrapper = None
        self.flashinfer_prefill_workspace_buffer = None
        self.flashinfer_prefill_wrapper = None

        set_cpu_offload_max_bytes(
            int(self.cache_config.cpu_offload_gb * 1024**3))

    def load_model(self) -> None:
        with CudaMemoryProfiler() as m:
            self.model = get_model(model_config=self.model_config,
                                   device_config=self.device_config,
                                   load_config=self.load_config,
                                   lora_config=self.lora_config,
                                   multimodal_config=self.multimodal_config,
                                   parallel_config=self.parallel_config,
                                   scheduler_config=self.scheduler_config,
                                   cache_config=self.cache_config)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

        if self.lora_config:
            assert supports_lora(self.model), "Model does not support LoRA"
            assert not supports_vision(
                self.model
            ), "To be tested: vision language model with LoRA settings."

            self.lora_manager = LRUCacheWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
                self.vocab_size,
                self.lora_config,
                self.device,
                self.model.embedding_modules,
                self.model.embedding_padding_modules,
                max_position_embeddings=self.model.config.
                max_position_embeddings,
            )
            self.model = self.lora_manager.create_lora_manager(self.model)

        if self.prompt_adapter_config:
            self.prompt_adapter_manager = LRUCacheWorkerPromptAdapterManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens, self.device,
                self.prompt_adapter_config)
            self.model = (
                self.prompt_adapter_manager.create_prompt_adapter_manager(
                    self.model))

        if self.kv_cache_dtype == "fp8" and is_hip():
            # Currently only ROCm accepts kv-cache scaling factors
            # via quantization_param_path and this will be deprecated
            # in the future.
            if self.model_config.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    warnings.warn(
                        "Loading kv cache scaling factor from JSON is "
                        "deprecated and will be removed. Please include "
                        "kv cache scaling factors in the model checkpoint.",
                        FutureWarning,
                        stacklevel=2)
                    self.model.load_kv_cache_scales(
                        self.model_config.quantization_param_path)
                    logger.info("Loaded KV cache scaling factors from %s",
                                self.model_config.quantization_param_path)
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__)
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!")

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.model_executor.model_loader.loader import ShardedStateLoader
        ShardedStateLoader.save_model(
            self.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        from vllm.model_executor.model_loader.loader import TensorizerLoader
        TensorizerLoader.save_model(
            self.model,
            tensorizer_config=tensorizer_config,
        )

    def get_max_block_per_batch(self) -> int:
        block_size = self.block_size
        return (self.max_seq_len_to_capture + block_size - 1) // block_size

    def _prepare_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        finished_requests_ids: Optional[List[str]] = None
    ) -> TModelInputForGPU:
        """Helper method to prepare the model input based on a given sequence
        group. Prepares metadata needed for the base model forward pass but not
        metadata for possible additional steps, e.g., sampling.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If cuda graph is required, this API automatically pads inputs.
        """
        builder = ModelInputForGPUBuilder(weakref.proxy(self),
                                          finished_requests_ids)
        for seq_group_metadata in seq_group_metadata_list:
            builder.add_seq_group(seq_group_metadata)
        return builder.build()  # type: ignore

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        dummy_lora_requests: List[LoRARequest] = []
        dummy_lora_requests_per_seq: List[LoRARequest] = []
        if self.lora_config:
            assert self.lora_manager is not None
            with self.lora_manager.dummy_lora_cache():
                for idx in range(self.lora_config.max_loras):
                    lora_id = idx + 1
                    dummy_lora_request = LoRARequest(
                        lora_name=f"warmup_{lora_id}",
                        lora_int_id=lora_id,
                        lora_local_path="/not/a/real/path",
                    )
                    self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                     rank=LORA_WARMUP_RANK)
                    dummy_lora_requests.append(dummy_lora_request)
                dummy_lora_requests_per_seq = [
                    dummy_lora_requests[idx % len(dummy_lora_requests)]
                    for idx in range(max_num_seqs)
                ]

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        # Additional GPU memory may be needed for vision encoding, which needs
        # to be accounted for when calculating the GPU blocks for
        # vLLM blocker manager.
        # To exercise the worst scenario for GPU memory consumption,
        # the number of seqs (batch_size) is chosen to maximize the number
        # of images processed.
        model_config = self.model_config

        if supports_vision(self.model):
            max_mm_tokens = MULTIMODAL_REGISTRY \
                .get_max_multimodal_tokens(model_config)
            max_num_seqs_orig = max_num_seqs
            max_num_seqs = min(max_num_seqs,
                               max_num_batched_tokens // max_mm_tokens)
            if max_num_seqs < 1:
                expr = (f"min({max_num_seqs_orig}, "
                        f"{max_num_batched_tokens} // {max_mm_tokens})")
                logger.warning(
                    "Computed max_num_seqs (%s) to be less than 1. "
                    "Setting it to the minimum value of 1.", expr)
                max_num_seqs = 1

        batch_size = 0
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            batch_size += seq_len

            seq_data, dummy_multi_modal_data = INPUT_REGISTRY \
                .dummy_data_for_profiling(model_config, seq_len)

            # Having more tokens is over-conservative but otherwise fine
            assert len(seq_data.prompt_token_ids) >= seq_len, (
                f"Expected at least {seq_len} dummy tokens for profiling, "
                f"but got: {len(seq_data.prompt_token_ids)}")

            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                multi_modal_data=dummy_multi_modal_data,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        finished_requests_ids = [seq.request_id for seq in seqs]
        model_input = self.prepare_model_input(
            seqs, finished_requests_ids=finished_requests_ids)
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=batch_size,
                dtype=self.model_config.dtype,
                device=self.device)
        self.execute_model(model_input, kv_caches, intermediate_tensors)
        torch.cuda.synchronize()
        return

    def remove_all_loras(self):
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.remove_all_adapters()

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.set_active_adapters(lora_requests, lora_mapping)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.add_adapter(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.remove_adapter(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.pin_adapter(lora_id)

    def list_loras(self) -> Set[int]:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.list_adapters()

    def remove_all_prompt_adapters(self):
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        self.prompt_adapter_manager.remove_all_adapters()

    def set_active_prompt_adapters(
            self, prompt_adapter_requests: Set[PromptAdapterRequest],
            prompt_adapter_mapping: PromptAdapterMapping) -> None:
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        self.prompt_adapter_manager.set_active_adapters(
            prompt_adapter_requests, prompt_adapter_mapping)

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        return self.prompt_adapter_manager.add_adapter(prompt_adapter_request)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        return self.prompt_adapter_manager.remove_adapter(prompt_adapter_id)

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        return self.prompt_adapter_manager.pin_adapter(prompt_adapter_id)

    def list_prompt_adapters(self) -> Set[int]:
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        return self.prompt_adapter_manager.list_adapters()

    @torch.inference_mode()
    def capture_model(self, kv_caches: List[List[torch.Tensor]]) -> None:
        """Cuda graph capture a model.

        Note that CUDA graph's performance gain is negligible if number
        of batched tokens are larger than 200. And since CUDA graph
        requires fixed sized tensors, supporting large/variable batch
        size requires high GPU memory overhead. Thus, vLLM only captures
        decoding requests. Mixed batch (chunked prefill + decoding) or
        prefill requests are not captured.

        Since it is used for decoding-only, it assumes there's only 1 token
        per sequence in the batch.
        """
        assert not self.model_config.enforce_eager
        logger.info("Capturing the model for CUDA graphs. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI.")
        logger.info("CUDA graphs can take additional 1~3 GiB memory per GPU. "
                    "If you are running out of memory, consider decreasing "
                    "`gpu_memory_utilization` or enforcing eager mode. "
                    "You can also reduce the `max_num_seqs` as needed "
                    "to decrease memory usage.")
        start_time = time.perf_counter()

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        input_tokens = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        input_positions = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        slot_mapping = torch.empty(max_batch_size, dtype=torch.long).cuda()
        slot_mapping.fill_(_PAD_SLOT_ID)
        seq_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
        block_tables = torch.from_numpy(self.graph_block_tables).cuda()
        intermediate_inputs = None
        if not get_pp_group().is_first_rank:
            intermediate_inputs = self.model.make_empty_intermediate_tensors(
                batch_size=max_batch_size,
                dtype=self.model_config.dtype,
                device=self.device)

        # Prepare buffer for outputs. These will be reused for all batch sizes.
        # It will be filled after the first graph capture.
        hidden_or_intermediate_states: List[Optional[torch.Tensor]] = [
            None
        ] * self.parallel_config.pipeline_parallel_size

        graph_batch_size = _get_graph_batch_size(
            self.scheduler_config.max_num_seqs)
        batch_size_capture_list = [
            bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
        ]

        if self.attn_backend.get_name() == "flashinfer":
            # For flashinfer, different batch sizes will share the
            # same workspace buffer.
            decode_workspace_buffer = \
            torch.empty(FLASHINFER_WORKSPACE_BUFFER_SIZE,
                                                dtype=torch.uint8,
                                              device=self.device)
            indices_buffer = torch.empty(max_batch_size *
                                         self.cache_config.num_gpu_blocks,
                                         dtype=torch.int32,
                                         device=self.device)
            indptr_buffer = torch.empty(max_batch_size + 1,
                                        dtype=torch.int32,
                                        device=self.device)
            last_page_len_buffer = torch.empty(max_batch_size,
                                               dtype=torch.int32,
                                               device=self.device)

        with graph_capture() as graph_capture_context:
            # NOTE: Capturing the largest batch size first may help reduce the
            # memory usage of CUDA graph.
            for virtual_engine in range(
                    self.parallel_config.pipeline_parallel_size):
                for batch_size in reversed(batch_size_capture_list):
                    if self.attn_backend.get_name() == "flashinfer":
                        indptr_buffer = indptr_buffer[:batch_size + 1]
                        last_page_len_buffer = last_page_len_buffer[:
                                                                    batch_size]

                        num_qo_heads = (
                            self.model_config.get_num_attention_heads(
                                self.parallel_config))
                        num_kv_heads = self.model_config.get_num_kv_heads(
                            self.parallel_config)
                        if num_qo_heads // num_kv_heads >= 4:
                            use_tensor_cores = True
                        else:
                            use_tensor_cores = False
                        decode_wrapper = \
                            CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
                            decode_workspace_buffer, indptr_buffer,
                            indices_buffer, last_page_len_buffer, "NHD",
                            use_tensor_cores)
                        kv_cache_dtype = get_kv_cache_torch_dtype(
                            self.kv_cache_dtype, self.model_config.dtype)

                        paged_kv_indptr_tensor_host = torch.arange(
                            0, batch_size + 1, dtype=torch.int32)
                        paged_kv_indices_tensor_host = torch.arange(
                            0, batch_size, dtype=torch.int32)
                        paged_kv_last_page_len_tensor_host = torch.full(
                            (batch_size, ), self.block_size, dtype=torch.int32)
                        query_start_loc_host = torch.arange(0,
                                                            batch_size + 1,
                                                            dtype=torch.int32)

                        attn_metadata = self.attn_backend.make_metadata(
                            num_prefills=0,
                            slot_mapping=slot_mapping[:batch_size],
                            num_prefill_tokens=0,
                            num_decode_tokens=batch_size,
                            max_prefill_seq_len=0,
                            block_tables=block_tables,
                            paged_kv_indptr=paged_kv_indptr_tensor_host,
                            paged_kv_indices=paged_kv_indices_tensor_host,
                            paged_kv_last_page_len=
                            paged_kv_last_page_len_tensor_host,
                            num_qo_heads=num_qo_heads,
                            num_kv_heads=num_kv_heads,
                            head_dim=self.model_config.get_head_size(),
                            page_size=self.block_size,
                            seq_start_loc=None,
                            query_start_loc=query_start_loc_host,
                            device=self.device,
                            data_type=kv_cache_dtype,
                            use_cuda_graph=True,
                            decode_wrapper=decode_wrapper,
                            prefill_wrapper=None)
                        attn_metadata.begin_forward()
                    else:
                        attn_metadata = self.attn_backend.make_metadata(
                            num_prefills=0,
                            num_prefill_tokens=0,
                            num_decode_tokens=batch_size,
                            slot_mapping=slot_mapping[:batch_size],
                            seq_lens=None,
                            seq_lens_tensor=seq_lens[:batch_size],
                            max_query_len=None,
                            max_prefill_seq_len=0,
                            max_decode_seq_len=self.max_seq_len_to_capture,
                            query_start_loc=None,
                            seq_start_loc=None,
                            context_lens_tensor=None,
                            block_tables=block_tables[:batch_size],
                            use_cuda_graph=True,
                        )

                    if self.lora_config:
                        lora_mapping = LoRAMapping(
                            [0] * batch_size,
                            [0] * batch_size,
                        )
                        self.set_active_loras(set(), lora_mapping)

                    if self.prompt_adapter_config:
                        prompt_adapter_mapping = PromptAdapterMapping(
                            [-1] * batch_size,
                            [-1] * batch_size,
                        )
                        self.set_active_prompt_adapters(
                            set(), prompt_adapter_mapping)

                    graph_runner = CUDAGraphRunner(
                        self.model, self.attn_backend.get_name())

                    if self.attn_backend.get_name() == "flashinfer":
                        graph_runner.flashinfer_indptr_buffer = indptr_buffer
                        graph_runner.flashinfer_indices_buffer = indices_buffer
                        graph_runner.flashinfer_last_page_len_buffer = \
                            last_page_len_buffer
                        graph_runner.flashinfer_decode_workspace_buffer = \
                                decode_workspace_buffer
                        graph_runner.flashinfer_decode_wrapper = \
                            decode_wrapper

                    capture_inputs = {
                        "input_ids":
                        input_tokens[:batch_size],
                        "positions":
                        input_positions[:batch_size],
                        "hidden_or_intermediate_states":
                        hidden_or_intermediate_states[
                            virtual_engine]  # type: ignore
                        [:batch_size]
                        if hidden_or_intermediate_states[virtual_engine]
                        is not None else None,
                        "intermediate_inputs":
                        intermediate_inputs[:batch_size]
                        if intermediate_inputs is not None else None,
                        "kv_caches":
                        kv_caches[virtual_engine],
                        "attn_metadata":
                        attn_metadata,
                        "memory_pool":
                        self.graph_memory_pool,
                        "stream":
                        graph_capture_context.stream
                    }
                    if self.has_seqlen_agnostic:
                        # Only used by Mamba-based models CUDA graph atm (Jamba)
                        capture_inputs.update({
                            "seqlen_agnostic_capture_inputs":
                            self.model.get_seqlen_agnostic_capture_inputs(
                                batch_size)
                        })
                    graph_runner.capture(**capture_inputs)
                    self.graph_memory_pool = graph_runner.graph.pool()
                    self.graph_runners[virtual_engine][batch_size] = (
                        graph_runner)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # This usually takes < 10 seconds.
        logger.info("Graph capturing finished in %.0f secs.", elapsed_time)

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()


class ModelRunner(GPUModelRunnerBase[ModelInputForGPUWithSamplingMetadata]):
    """
    GPU model runner with sampling step.
    """
    _model_input_cls: Type[ModelInputForGPUWithSamplingMetadata] = (
        ModelInputForGPUWithSamplingMetadata)

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> ModelInputForGPUWithSamplingMetadata:
        model_input = \
            ModelInputForGPUWithSamplingMetadata.from_broadcasted_tensor_dict(
                tensor_dict,
                attn_backend=self.attn_backend,
            )
        return model_input

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForGPUWithSamplingMetadata:
        """Prepare the model input based on a given sequence group, including
        metadata for the sampling step.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If cuda graph is required, this API automatically pads inputs.
        """
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        sampling_metadata = SamplingMetadata.prepare(seq_group_metadata_list,
                                                     model_input.seq_lens,
                                                     model_input.query_lens,
                                                     self.device,
                                                     self.pin_memory)
        is_prompt = (seq_group_metadata_list[0].is_prompt
                     if seq_group_metadata_list else None)
        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   is_prompt=is_prompt,
                                   virtual_engine=virtual_engine)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in ModelRunner")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        if self.prompt_adapter_config:
            assert model_input.prompt_adapter_requests is not None
            assert model_input.prompt_adapter_mapping is not None
            self.set_active_prompt_adapters(
                model_input.prompt_adapter_requests,
                model_input.prompt_adapter_mapping)

        if self.attn_backend.get_name() == "flashinfer":
            assert model_input.attn_metadata is not None
            assert model_input.input_tokens is not None
            if self.flashinfer_decode_workspace_buffer is None:
                self.flashinfer_decode_workspace_buffer = torch.empty(
                    FLASHINFER_WORKSPACE_BUFFER_SIZE,
                    dtype=torch.uint8,
                    device=self.device)
                self.flashinfer_decode_wrapper = \
                    BatchDecodeWithPagedKVCacheWrapper(
                    self.flashinfer_decode_workspace_buffer, "NHD")
                self.flashinfer_prefill_workspace_buffer = torch.empty(
                    FLASHINFER_WORKSPACE_BUFFER_SIZE,
                    dtype=torch.uint8,
                    device=self.device)
                self.flashinfer_prefill_wrapper = \
                    BatchPrefillWithPagedKVCacheWrapper(
                    self.flashinfer_prefill_workspace_buffer, "NHD")

            model_input.attn_metadata.prefill_wrapper = \
                self.flashinfer_prefill_wrapper
            if model_input.attn_metadata.use_cuda_graph:
                batch_size = model_input.input_tokens.shape[0]
                model_input.attn_metadata.decode_wrapper = self.graph_runners[
                    model_input.
                    virtual_engine][batch_size].flashinfer_decode_wrapper
            else:
                model_input.attn_metadata.decode_wrapper = \
                    self.flashinfer_decode_wrapper
            model_input.attn_metadata.begin_forward()

        # Currently cuda graph is only supported by the decode phase.
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        virtual_engine = model_input.virtual_engine
        if prefill_meta is None and decode_meta.use_cuda_graph:
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = self.graph_runners[virtual_engine][
                graph_batch_size]
        else:
            model_executable = self.model

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_seqlen_agnostic else {}
        hidden_or_intermediate_states = model_executable(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            kv_caches=kv_caches,
            attn_metadata=model_input.attn_metadata,
            intermediate_tensors=intermediate_tensors,
            **multi_modal_kwargs,
            **seqlen_agnostic_kwargs)

        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            return hidden_or_intermediate_states

        logits = self.model.compute_logits(hidden_or_intermediate_states,
                                           model_input.sampling_metadata)

        if not self.is_driver_worker:
            return []

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )

        if self.return_hidden_states:
            # we only need to pass hidden states of most recent token
            assert model_input.sampling_metadata is not None
            indices = model_input.sampling_metadata.selected_token_indices
            if model_input.is_prompt:
                hidden_states = hidden_or_intermediate_states.index_select(
                    0, indices)
            elif decode_meta.use_cuda_graph:
                hidden_states = hidden_or_intermediate_states[:len(indices)]
            else:
                hidden_states = hidden_or_intermediate_states

            output.hidden_states = hidden_states

        return [output]


class CUDAGraphRunner:

    def __init__(self, model: nn.Module, backend_name: str):
        self.model = model
        self.backend_name = backend_name

        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

        self._graph: Optional[torch.cuda.CUDAGraph] = None

        self.flashinfer_decode_workspace_buffer: Optional[torch.Tensor] = None
        self.flashinfer_indptr_buffer: Optional[torch.Tensor] = None
        self.flashinfer_indices_buffer: Optional[torch.Tensor] = None
        self.flashinfer_last_page_len_buffer: Optional[torch.Tensor] = None
        self.flashinfer_decode_wrapper: Optional[
            CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = None

    @property
    def graph(self):
        assert self._graph is not None
        return self._graph

    def capture(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_or_intermediate_states: Optional[Union[IntermediateTensors,
                                                      torch.Tensor]],
        intermediate_inputs: Optional[IntermediateTensors],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        memory_pool: Optional[Tuple[int, int]],
        stream: torch.cuda.Stream,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        assert self._graph is None
        # Run the model a few times without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        # Note one iteration is not enough for torch.jit.script
        for _ in range(_NUM_WARMUP_ITERS):
            self.model(
                input_ids,
                positions,
                kv_caches,
                attn_metadata,
                intermediate_inputs,
                **kwargs,
            )
        torch.cuda.synchronize()

        # Capture the graph.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=memory_pool, stream=stream):
            output_hidden_or_intermediate_states = self.model(
                input_ids,
                positions,
                kv_caches,
                attn_metadata,
                intermediate_inputs,
                **kwargs,
            )
            if hidden_or_intermediate_states is not None:
                if get_pp_group().is_last_rank:
                    hidden_or_intermediate_states.copy_(
                        output_hidden_or_intermediate_states)
                else:
                    for key in hidden_or_intermediate_states.tensors:
                        hidden_or_intermediate_states[key].copy_(
                            output_hidden_or_intermediate_states[key])
            else:
                hidden_or_intermediate_states = (
                    output_hidden_or_intermediate_states)

            del output_hidden_or_intermediate_states
            # make sure `output_hidden_states` is deleted
            # in the graph's memory pool
            gc.collect()
        torch.cuda.synchronize()

        # Save the input and output buffers.
        if self.backend_name == "flashinfer":
            self.input_buffers = {
                "input_ids": input_ids,
                "positions": positions,
                "kv_caches": kv_caches,
                "slot_mapping": attn_metadata.slot_mapping,
                **kwargs,
            }
        else:
            self.input_buffers = {
                "input_ids": input_ids,
                "positions": positions,
                "kv_caches": kv_caches,
                "slot_mapping": attn_metadata.slot_mapping,
                "seq_lens_tensor":
                attn_metadata.decode_metadata.seq_lens_tensor,
                "block_tables": attn_metadata.decode_metadata.block_tables,
                **kwargs,
            }
        if intermediate_inputs is not None:
            self.input_buffers.update(intermediate_inputs.tensors)
        if get_pp_group().is_last_rank:
            self.output_buffers = {
                "hidden_states": hidden_or_intermediate_states
            }
        else:
            self.output_buffers = hidden_or_intermediate_states
        return hidden_or_intermediate_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        **kwargs,
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del kv_caches

        # Copy the input tensors to the input buffers.
        self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)
        self.input_buffers["slot_mapping"].copy_(attn_metadata.slot_mapping,
                                                 non_blocking=True)
        if self.backend_name != "flashinfer":
            self.input_buffers["seq_lens_tensor"].copy_(
                attn_metadata.decode_metadata.seq_lens_tensor,
                non_blocking=True)
            self.input_buffers["block_tables"].copy_(
                attn_metadata.decode_metadata.block_tables, non_blocking=True)
        if "seqlen_agnostic_capture_inputs" in self.input_buffers:
            self.model.copy_inputs_before_cuda_graphs(self.input_buffers,
                                                      **kwargs)
        if intermediate_tensors is not None:
            for key in intermediate_tensors.tensors:
                self.input_buffers[key].copy_(intermediate_tensors[key],
                                              non_blocking=True)
        # Run the graph.
        self.graph.replay()
        if "seqlen_agnostic_capture_inputs" in self.input_buffers:
            self.model.copy_outputs_after_cuda_graphs(self.input_buffers,
                                                      **kwargs)
        # Return the output tensor.
        if get_pp_group().is_last_rank:
            return self.output_buffers["hidden_states"]

        return self.output_buffers

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _get_graph_batch_size(batch_size: int) -> int:
    """Returns the padded batch size given actual batch size.

    Batch sizes are 1, 2, 4, _BATCH_SIZE_ALIGNMENT,
    2*_BATCH_SIZE_ALIGNMENT, 3*_BATCH_SIZE_ALIGNMENT...
    """
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return ((batch_size + _BATCH_SIZE_ALIGNMENT - 1) //
                _BATCH_SIZE_ALIGNMENT * _BATCH_SIZE_ALIGNMENT)


def _is_block_tables_empty(block_tables: Union[None, Dict]):
    """
    Check if block_tables is None or a dictionary with all None values.
    """
    if block_tables is None:
        return True
    if isinstance(block_tables, dict) and all(
            value is None for value in block_tables.values()):
        return True
    return False