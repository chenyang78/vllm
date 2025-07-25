# Conserving Memory

Large models might cause your machine to run out of memory (OOM). Here are some options that help alleviate this problem.

## Tensor Parallelism (TP)

Tensor parallelism (`tensor_parallel_size` option) can be used to split the model across multiple GPUs.

The following code splits the model across 2 GPUs.

```python
from vllm import LLM

llm = LLM(model="ibm-granite/granite-3.1-8b-instruct",
          tensor_parallel_size=2)
```

!!! warning
    To ensure that vLLM initializes CUDA correctly, you should avoid calling related functions (e.g. [torch.cuda.set_device][])
    before initializing vLLM. Otherwise, you may run into an error like `RuntimeError: Cannot re-initialize CUDA in forked subprocess`.

    To control which devices are used, please instead set the `CUDA_VISIBLE_DEVICES` environment variable.

!!! note
    With tensor parallelism enabled, each process will read the whole model and split it into chunks, which makes the disk reading time even longer (proportional to the size of tensor parallelism).

    You can convert the model checkpoint to a sharded checkpoint using <gh-file:examples/offline_inference/save_sharded_state.py>. The conversion process might take some time, but later you can load the sharded checkpoint much faster. The model loading time should remain constant regardless of the size of tensor parallelism.

## Quantization

Quantized models take less memory at the cost of lower precision.

Statically quantized models can be downloaded from HF Hub (some popular ones are available at [Red Hat AI](https://huggingface.co/RedHatAI))
and used directly without extra configuration.

Dynamic quantization is also supported via the `quantization` option -- see [here](../features/quantization/README.md) for more details.

## Context length and batch size

You can further reduce memory usage by limiting the context length of the model (`max_model_len` option)
and the maximum batch size (`max_num_seqs` option).

```python
from vllm import LLM

llm = LLM(model="adept/fuyu-8b",
          max_model_len=2048,
          max_num_seqs=2)
```

## Reduce CUDA Graphs

By default, we optimize model inference using CUDA graphs which take up extra memory in the GPU.

!!! warning
    CUDA graph capture takes up more memory in V1 than in V0.

You can adjust `compilation_config` to achieve a better balance between inference speed and memory usage:

??? code

    ```python
    from vllm import LLM
    from vllm.config import CompilationConfig, CompilationLevel

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        compilation_config=CompilationConfig(
            level=CompilationLevel.PIECEWISE,
            # By default, it goes up to max_num_seqs
            cudagraph_capture_sizes=[1, 2, 4, 8, 16],
        ),
    )
    ```

You can disable graph capturing completely via the `enforce_eager` flag:

```python
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
          enforce_eager=True)
```

## Adjust cache size

If you run out of CPU RAM, try the following options:

- (Multi-modal models only) you can set the size of multi-modal input cache using `VLLM_MM_INPUT_CACHE_GIB` environment variable (default 4 GiB).
- (CPU backend only) you can set the size of KV cache using `VLLM_CPU_KVCACHE_SPACE` environment variable (default 4 GiB).

## Multi-modal input limits

You can allow a smaller number of multi-modal items per prompt to reduce the memory footprint of the model:

```python
from vllm import LLM

# Accept up to 3 images and 1 video per prompt
llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct",
          limit_mm_per_prompt={"image": 3, "video": 1})
```

You can go a step further and disable unused modalities completely by setting its limit to zero.
For example, if your application only accepts image input, there is no need to allocate any memory for videos.

```python
from vllm import LLM

# Accept any number of images but no videos
llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct",
          limit_mm_per_prompt={"video": 0})
```

You can even run a multi-modal model for text-only inference:

```python
from vllm import LLM

# Don't accept images. Just text.
llm = LLM(model="google/gemma-3-27b-it",
          limit_mm_per_prompt={"image": 0})
```

## Multi-modal processor arguments

For certain models, you can adjust the multi-modal processor arguments to
reduce the size of the processed multi-modal inputs, which in turn saves memory.

Here are some examples:

??? code

    ```python
    from vllm import LLM

    # Available for Qwen2-VL series models
    llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct",
            mm_processor_kwargs={
                "max_pixels": 768 * 768,  # Default is 1280 * 28 * 28
            })

    # Available for InternVL series models
    llm = LLM(model="OpenGVLab/InternVL2-2B",
            mm_processor_kwargs={
                "max_dynamic_patch": 4,  # Default is 12
            })
    ```
