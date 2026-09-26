---
title: "vLLM from the ground up: My PyTorch meetup walkthrough"
date: 2025-12-20
description: "The PyTorch meetup version of my vLLM talk, covering transformers, paged KV memory, scheduling, and serving features."
---

<!-- Source presentation: vLLM presentation - Pytorch meetup.pdf. Date is the blog publication date. -->

For this PyTorch meetup talk, I started with token generation and worked toward the serving engine around it. The question throughout was simple: what has to happen between receiving a prompt and keeping its response streaming efficiently?

This post follows the meetup deck in its original order, including its transformer refresher. Click any slide to open the diagram at full resolution.

## 1. Understanding the serving system

<a href="/assets/slides/vllm-pytorch-meetup/slide-01.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-01.png" alt="Slide 1: Understanding the serving system" width="2400" height="1350" loading="eager" decoding="async">
</a>

vLLM combines model execution with the machinery needed to serve many requests. To understand why that machinery matters, we first need to look at the work a single request creates. We'll move from tokens and attention to KV-cache allocation, then follow those requests through the engine.

## 2. The GPU has a memory budget

<a href="/assets/slides/vllm-pytorch-meetup/slide-02.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-02.png" alt="Slide 2: The GPU has a memory budget" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Loading the model doesn't leave the remaining memory empty. Every active request brings attention state, and execution needs temporary buffers too. Longer sequences can reduce how many requests fit at once. This is why a serving system's memory policy affects throughput even when its individual model kernels are already fast.

## 3. How token generation works

<a href="/assets/slides/vllm-pytorch-meetup/slide-03.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-03.png" alt="Slide 3: How token generation works" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The prompt supplies the initial context. The model predicts a next-token distribution, a token is selected, and generation continues with that token added to the sequence. Requests finish at different points, so a useful scheduler needs to admit new work as old work completes instead of waiting for a whole fixed batch to finish.

## 4. A quick transformer refresher

<a href="/assets/slides/vllm-pytorch-meetup/slide-04.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-04.png" alt="Slide 4: A quick transformer refresher" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The model repeatedly applies attention and feed-forward transformations to token representations, with normalization and residual connections around those computations. The picture helps locate the attention state that we'll cache. We don't need every detail of the model architecture to see why retaining that state across decoding steps saves work.

## 5. Why keys and values matter

<a href="/assets/slides/vllm-pytorch-meetup/slide-05.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-05.png" alt="Slide 5: Why keys and values matter" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Attention lets a token combine information from its available context. During causal decoding, the new token needs access to earlier keys and values at each relevant layer. Those earlier vectors are reusable. That is the starting point for KV caching, and also the reason longer conversations carry a growing memory cost.

## 6. The problem with large reservations

<a href="/assets/slides/vllm-pytorch-meetup/slide-06.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-06.png" alt="Slide 6: The problem with large reservations" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Output length is unknown when a request arrives. Reserving a contiguous region for its maximum possible length wastes space, while allocating different-sized regions can leave unusable gaps. The utilization range on this slide comes from the historical setting of the [PagedAttention work](https://arxiv.org/abs/2309.06180). The allocation problem is the point to carry forward, rather than treating that range as a current benchmark.

## 7. Several optimizations work together

<a href="/assets/slides/vllm-pytorch-meetup/slide-07.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-07.png" alt="Slide 7: Several optimizations work together" width="2400" height="1350" loading="lazy" decoding="async">
</a>

vLLM addresses more than one source of inefficiency: KV allocation, batching, kernel execution, prompt processing, and token generation. PagedAttention helps organize memory; features such as chunked prefill change how work is scheduled. Their benefits depend on what is actually limiting the service, so it's useful to understand them individually before combining them.

## 8. Logical order, flexible storage

<a href="/assets/slides/vllm-pytorch-meetup/slide-08.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-08.png" alt="Slide 8: Logical order, flexible storage" width="2400" height="1350" loading="lazy" decoding="async">
</a>

PagedAttention lets the KV state for one sequence live in separate physical blocks while preserving its logical order. The attention computation follows the block mapping to access the right state. This means the allocator can grow a request block by block instead of needing a contiguous region large enough for its eventual output.

## 9. Following the block table

<a href="/assets/slides/vllm-pytorch-meetup/slide-09.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-09.png" alt="Slide 9: Following the block table" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Request A and request B each have their own logical sequence of blocks. Their tables point into a shared pool of physical storage, where blocks can appear in a different order. When a request finishes, its unneeded blocks become available again. The mapping keeps token order separate from allocation decisions.

## 10. What KV caching actually avoids

<a href="/assets/slides/vllm-pytorch-meetup/slide-10.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-10.png" alt="Slide 10: What KV caching actually avoids" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The cache preserves earlier key and value projections so decoding doesn't repeatedly recreate them. New tokens still need their own computation, and attention still reads relevant historical state. That distinction matters: a cache removes redundant work, but a long context can remain expensive to access even when every cached entry is available.

## 11. Naming the moving parts

<a href="/assets/slides/vllm-pytorch-meetup/slide-11.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-11.png" alt="Slide 11: Naming the moving parts" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The tokenizer handles text representation, the engine coordinates requests, and the scheduler chooses the next work to run. The cache manager supplies storage, while executors and workers carry out model execution. These names become more useful once we connect them to the lifecycle of a request rather than reading them as a flat list.

## 12. The architecture diagram

<a href="/assets/slides/vllm-pytorch-meetup/slide-12.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-12.png" alt="Slide 12: The architecture diagram" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Input processing feeds the engine core, which combines scheduling and cache management with model execution. Outputs travel back through processing before reaching the caller. The diagram includes several possible devices and memory arrangements; a particular deployment uses the paths its configuration supports. The main relationship is that scheduling decisions determine the work sent to workers.

## 13. What initialization prepares

<a href="/assets/slides/vllm-pytorch-meetup/slide-13.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-13.png" alt="Slide 13: What initialization prepares" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The call graph shows how the engine builds the parts it will need repeatedly: workers, model resources, cache state, and scheduling queues. Initialization can also include profiling and warmup before requests arrive. It's an approximate map of responsibilities, so I would use it to navigate the source rather than expect every version to preserve the exact constructor sequence.

## 14. Four features worth understanding

<a href="/assets/slides/vllm-pytorch-meetup/slide-14.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-14.png" alt="Slide 14: Four features worth understanding" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The next slides each change a different part of serving. Chunked prefill shares time between prompt work and decoding. Prefix caching reuses previous prompt computation. Guided decoding constrains what can be sampled. Disaggregated prefill and decode place the two phases on separate serving resources.

## 15. Processing a long prompt in chunks

<a href="/assets/slides/vllm-pytorch-meetup/slide-15.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-15.png" alt="Slide 15: Processing a long prompt in chunks" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Chunking lets a long prefill make progress over multiple scheduling steps. That creates room for active decode requests to keep running instead of waiting behind one large prompt. Choosing the amount of prompt work per step is a latency-throughput tradeoff, which is why the right budget depends on the request mix.

## 16. Reusing a shared prefix

<a href="/assets/slides/vllm-pytorch-meetup/slide-16.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-16.png" alt="Slide 16: Reusing a shared prefix" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Many requests may share a system prompt or the start of a document. When compatible cached blocks exist, prefix caching can reuse their KV state and process only the remaining prompt work. The diagram shows why block identity and reference tracking matter: the cache has to recognize reusable computation and keep it alive while requests depend on it.

## 17. Restricting the next token

<a href="/assets/slides/vllm-pytorch-meetup/slide-17.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-17.png" alt="Slide 17: Restricting the next token" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Structured output generation applies constraints before sampling. Tokens that would break the current grammar state are masked out, leaving the model to choose among valid continuations. The slide illustrates this with a compact bitmask. This controls output structure, such as valid JSON, while the model still determines the content within those constraints.

## 18. Moving prefill away from decode

<a href="/assets/slides/vllm-pytorch-meetup/slide-18.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-18.png" alt="Slide 18: Moving prefill away from decode" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Separating the phases allows a deployment to tune prompt processing and ongoing generation independently. After prefill, the decode instance receives the KV state and continues the sequence. The extra transfer needs to pay for itself, so this is a system-level tradeoff involving network bandwidth, scheduling, and the latency targets of the service.

## 19. References from the talk

<a href="/assets/slides/vllm-pytorch-meetup/slide-19.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-19.png" alt="Slide 19: References from the talk" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The [PagedAttention paper](https://arxiv.org/abs/2309.06180) and [vLLM source](https://github.com/vllm-project/vllm/tree/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb) are good places to connect the memory ideas to implementation. The slide also includes the [Anatomy of vLLM article](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html), the [original technical talk](https://www.youtube.com/watch?v=5ZlavKF_98U), and the [3Blue1Brown neural-network series](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) used for the visual refresher.

## 20. Where to go next

<a href="/assets/slides/vllm-pytorch-meetup/slide-20.png">
  <img src="/assets/slides/vllm-pytorch-meetup/slide-20.png" alt="Slide 20: Where to go next" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Thanks for following the meetup walkthrough. Once the scheduler and cache manager make sense, the next question is how the model's repeated computation gets optimized. I follow that path in the [vLLM and torch.compile post](/blog/vllm-and-torch-compile/), from the model runner down to compiled graph pieces and CUDA graph replay.
