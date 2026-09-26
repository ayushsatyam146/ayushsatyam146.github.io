---
title: "Where torch.compile meets vLLM"
date: 2026-01-16
description: "Following vLLM's compilation path through model wrappers, graph splitting, shape dispatch, fusion passes, and CUDA graphs."
---

<!-- Source presentation: vLLM & torch.compile.pdf. Date is the blog publication date. -->

After looking at vLLM's scheduler and KV cache, I wanted to follow what happens inside model execution. Where does `torch.compile` enter the picture, and how much work does vLLM add around it?

This talk follows that integration one layer at a time. The original slides are below, with short explanations and full-size images available on click. Some class names and defaults change between releases, so the diagrams are most useful as a map of the design.

## 1. A closer look at vLLM compilation

<a href="/assets/slides/vllm-and-torch-compile/slide-01.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-01.png" alt="Slide 1: A closer look at vLLM compilation" width="2400" height="1350" loading="eager" decoding="async">
</a>

vLLM uses PyTorch's compiler machinery, but it also manages graph boundaries, cache keys, supported shapes, and CUDA graph execution around it. The interesting part is how those pieces fit a serving workload. A model that runs repeatedly with changing batches needs more than a one-time call to a compiler.

## 2. The joke behind the title

<a href="/assets/slides/vllm-and-torch-compile/slide-02.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-02.png" alt="Slide 2: The joke behind the title" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The meme captures the starting observation: follow vLLM's compilation path far enough and you'll meet `torch.compile` and its internals. There is still substantial vLLM-specific machinery along the way. That machinery adapts graph capture and execution to the constraints of an inference engine.

## 3. What is vLLM?

<a href="/assets/slides/vllm-and-torch-compile/slide-03.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-03.png" alt="Slide 3: What is vLLM?" width="2400" height="1350" loading="lazy" decoding="async">
</a>

vLLM is a library and serving system for LLM inference. It coordinates incoming requests, schedules tokens, manages KV state, and executes models on supported devices. Compilation lives inside that larger system, so understanding its location helps keep model optimization separate from request orchestration.

## 4. Compilation is one of several optimizations

<a href="/assets/slides/vllm-and-torch-compile/slide-04.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-04.png" alt="Slide 4: Compilation is one of several optimizations" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Paged KV storage, efficient kernels, chunked prefill, and speculative decoding all address different costs. Compiler-generated work fits alongside them. If a workload is limited by KV capacity or communication, making one pointwise operation faster may have little effect on the whole service. The surrounding system still determines how much useful work reaches the GPU.

## 5. Start with the architecture

<a href="/assets/slides/vllm-and-torch-compile/slide-05.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-05.png" alt="Slide 5: Start with the architecture" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Before following the compiler, locate the engine, scheduler, executor, worker, and model runner. The compiler works much closer to the model's tensor computation than to the API server. Keeping that boundary in mind prevents a common confusion: compiling a model doesn't mean compiling the entire request-handling loop.

## 6. Requests flow into model execution

<a href="/assets/slides/vllm-and-torch-compile/slide-06.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-06.png" alt="Slide 6: Requests flow into model execution" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Input processing feeds the engine core. Scheduling and KV allocation determine the work that the executor sends to workers, and results return through output processing. The diagram shows where these responsibilities connect. Compiler optimizations help execute the chosen model work; they don't replace the scheduler's decisions about which requests should run.

## 7. From the engine to the worker

<a href="/assets/slides/vllm-and-torch-compile/slide-07.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-07.png" alt="Slide 7: From the engine to the worker" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The call graph breaks the system into initialization and execution responsibilities. Workers prepare devices and model resources, while the engine core owns the repeated scheduling cycle. The model runner prepares inputs and attention metadata before invoking the model. That is the point where the serving system approaches the compiled computation.

## 8. What torch.compile contributes

<a href="/assets/slides/vllm-and-torch-compile/slide-08.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-08.png" alt="Slide 8: What torch.compile contributes" width="2400" height="1350" loading="lazy" decoding="async">
</a>

`torch.compile` can capture supported regions of PyTorch code and hand them to an optimizing backend. Dynamo performs capture, while a backend such as Inductor produces executable work. vLLM uses these capabilities as building blocks, with extra control over the shapes, transformations, and runtime paths relevant to serving.

## 9. The compiler pipeline refresher

<a href="/assets/slides/vllm-and-torch-compile/slide-09.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-09.png" alt="Slide 9: The compiler pipeline refresher" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The diagram follows Dynamo, AOTAutograd, and Inductor. For inference, AOTAutograd's machinery still helps prepare and transform the graph even though a backward graph isn't needed. Inductor then lowers supported operations and generates code or external calls. This is the compiler path that vLLM customizes around model execution.

## 10. The exact boundary in the serving stack

<a href="/assets/slides/vllm-and-torch-compile/slide-10.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-10.png" alt="Slide 10: The exact boundary in the serving stack" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Follow the stack down from the request to `GPUModelRunner` and the model's forward call. Tokenization, queuing, and much of input preparation happen outside the compiled region. The repeated tensor computation is where graph compilation and CUDA graph replay can reduce overhead. This boundary is one of the main design decisions in the integration.

## 11. The layers of the compilation pipeline

<a href="/assets/slides/vllm-and-torch-compile/slide-11.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-11.png" alt="Slide 11: The layers of the compilation pipeline" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The wrapper prepares a model for capture, `VllmBackend` manages the graph, piecewise machinery handles supported regions and shapes, and CUDA graph wrappers manage capture and replay. These are separate jobs. In particular, compiling a region and recording its device execution in a CUDA graph are different transformations with different reuse conditions.

## 12. The decorator and wrapper

<a href="/assets/slides/vllm-and-torch-compile/slide-12.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-12.png" alt="Slide 12: The decorator and wrapper" width="2400" height="1350" loading="lazy" decoding="async">
</a>

`support_torch_compile` connects eligible model classes to vLLM's compilation setup, including dynamic-dimension information and a lazy first-call path. Later calls can use prepared execution paths. The slide's guard-dropping shorthand relies on assumptions enforced by the serving system; it isn't a general suggestion to remove correctness checks from arbitrary PyTorch programs.

## 13. What the vLLM backend does

<a href="/assets/slides/vllm-and-torch-compile/slide-13.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-13.png" alt="Slide 13: What the vLLM backend does" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The backend receives an FX graph, manages compilation caching, and can split around configured operations before compiling supported pieces. Attention is an important example in the piecewise design shown here. The selected attention backend still executes that work. The [`VllmBackend` source](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/compilation/backends.py) also shows why full-graph and partitioned configurations shouldn't be collapsed into one universal path.

## 14. Dispatching by shape

<a href="/assets/slides/vllm-and-torch-compile/slide-14.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-14.png" alt="Slide 14: Dispatching by shape" width="2400" height="1350" loading="lazy" decoding="async">
</a>

One compiled graph can cover a range of shapes, while selected sizes can have more specialized implementations. At runtime, the piecewise backend chooses the appropriate callable. The diagram labels the dimension as batch size, but token count is also central in serving, especially when prefill contributes several tokens per request. The concrete shape contract matters more than the shorthand label.

## 15. The compiler interface

<a href="/assets/slides/vllm-and-torch-compile/slide-15.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-15.png" alt="Slide 15: The compiler interface" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The compiler interface separates vLLM's orchestration from backend-specific compile and load behavior. The slide includes multiple Inductor adapters and an eager path for comparison or debugging. This means the integration isn't tied to one identical backend path in every configuration. The checked implementation is in [`compiler_interface.py`](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/compilation/compiler_interface.py).

## 16. Passes that understand the workload

<a href="/assets/slides/vllm-and-torch-compile/slide-16.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-16.png" alt="Slide 16: Passes that understand the workload" width="2400" height="1350" loading="lazy" decoding="async">
</a>

vLLM can apply transformations around common inference patterns, such as normalization followed by quantization. Combining compatible work can reduce intermediate memory traffic and launch overhead. The exact passes and their order depend on configuration and implementation version. A fusion still needs to preserve the operation's numerical and mutation behavior; matching a familiar name is not enough.

## 17. CUDA graph capture and replay

<a href="/assets/slides/vllm-and-torch-compile/slide-17.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-17.png" alt="Slide 17: CUDA graph capture and replay" width="2400" height="1350" loading="lazy" decoding="async">
</a>

After code has been compiled or otherwise prepared, a CUDA graph can record a compatible execution sequence. Replaying it reduces repeated CPU launch work. Input addresses, shapes, and other capture assumptions still matter, and replay is not literally zero overhead. vLLM's [compilation debugging guide](https://docs.vllm.ai/en/latest/design/debug_vllm_compile/) usefully separates disabling compilation from disabling CUDA graphs.

## 18. References and source navigation

<a href="/assets/slides/vllm-and-torch-compile/slide-18.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-18.png" alt="Slide 18: References and source navigation" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The [vLLM compilation source](https://github.com/vllm-project/vllm/tree/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/compilation) is the best companion to these diagrams. The original slide also links to the [Anatomy of vLLM article](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) and my [earlier architecture talk](https://www.youtube.com/watch?v=20ZmaYkZ1jI). For the foundations behind the KV layout, see the [PagedAttention paper](https://arxiv.org/abs/2309.06180).

## 19. The next question: what should the graph represent?

<a href="/assets/slides/vllm-and-torch-compile/slide-19.png">
  <img src="/assets/slides/vllm-and-torch-compile/slide-19.png" alt="Slide 19: The next question: what should the graph represent?" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Once the integration makes sense, another problem appears: the same high-level operation can show up as different graph patterns for different kernels. That makes compiler passes harder to maintain. My [vLLM IR talk](/blog/vllm-torch-compile-and-ir/) picks up that question and looks at separating an operation's meaning from its implementation.
