---
title: "torch.compile in the wild: Where real projects draw the boundary"
date: 2026-08-12
description: "How inference, distributed training, reinforcement learning, and domain libraries adapt their code around torch.compile."
---

<!-- Source presentation: torch.compile in the wild.pdf. Date is the blog publication date. -->

The interesting question about `torch.compile` is often where to put it. A serving engine, a distributed training framework, and an RL library can all use the same compiler while making very different decisions about what gets captured.

In this talk, I looked across the PyTorch ecosystem to understand those decisions. These are the original slides, each followed by a short explanation. Click an image to read it at full size. Project counts, defaults, and code snippets in the slides are snapshots from the talk; the notes clarify the places where the wording is broader than the behavior.

## 1. Looking beyond the one-line API

<a href="/assets/slides/torch-compile-in-the-wild/slide-01.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-01.png" alt="Slide 1: Looking beyond the one-line API" width="2400" height="1350" loading="eager" decoding="async">
</a>

`model = torch.compile(model)` is a useful starting point, but a library has to make that call fit its own execution model. What happens before compilation? Which shapes will recur? What Python work should stay outside? Those questions explain many of the changes we see in real integrations.

## 2. A tour of the ecosystem

<a href="/assets/slides/torch-compile-in-the-wild/slide-02.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-02.png" alt="Slide 2: A tour of the ecosystem" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The deck brings together model libraries, serving engines, training frameworks, RL tools, and domain-specific packages. I wasn't trying to rank them. I wanted to see which problems kept appearing across otherwise different codebases, and which solutions only made sense for one kind of workload.

## 3. Different workloads expose different problems

<a href="/assets/slides/torch-compile-in-the-wild/slide-03.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-03.png" alt="Slide 3: Different workloads expose different problems" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Inference can be sensitive to startup time. Distributed training has to preserve communication and parameter-management behavior. RL often mixes tensor updates with an external environment. Graph and imaging libraries bring custom operations and data types. Each workload gives the compiler a different set of constraints before optimization even starts.

## 4. The compiler underneath

<a href="/assets/slides/torch-compile-in-the-wild/slide-04.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-04.png" alt="Slide 4: The compiler underneath" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Dynamo captures supported Python execution into FX graphs, and a backend turns those graphs into executable work. The usual Inductor path includes further graph transformations and code generation. A custom backend can add its own transformations or partitioning. That extension point is part of why these projects can use the same capture machinery in different ways.

## 5. Grouping projects by their constraints

<a href="/assets/slides/torch-compile-in-the-wild/slide-05.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-05.png" alt="Slide 5: Grouping projects by their constraints" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The categories in this slide help organize the discussion, but they overlap. Lightning participates in distributed training, while a serving engine may also run inside an RL system. The useful comparison is the repeated computation each project wants to optimize and the surrounding work it needs to preserve.

## 6. Inference: startup and repetition

<a href="/assets/slides/torch-compile-in-the-wild/slide-06.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-06.png" alt="Slide 6: Inference: startup and repetition" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Let's start with inference. A compiled region needs enough repeated execution to pay back the work spent preparing it. That makes both the steady-state runtime and the time before the service becomes ready relevant. A smaller compilation boundary can sometimes be the better operational choice.

## 7. The inference constraint

<a href="/assets/slides/torch-compile-in-the-wild/slide-07.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-07.png" alt="Slide 7: The inference constraint" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Stable input metadata makes reuse easier, but real services still encounter variable prompts, changing batches, and adapter updates. Libraries can pad, bucket, or otherwise stabilize selected dimensions. The question is how much specialization improves execution before extra compiled variants and startup work become too expensive.

## 8. Transformers and automatic compilation

<a href="/assets/slides/torch-compile-in-the-wild/slide-08.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-08.png" alt="Slide 8: Transformers and automatic compilation" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Transformers has generation paths that can select a compiled model call when the model, cache, device, and configuration support it. A static KV cache is useful because it avoids changing the cache allocation shape at every decode step. The eligibility checks are version-dependent; this isn't a promise that every `generate()` call automatically compiles. See the [inference optimization guide](https://huggingface.co/docs/transformers/main/en/llm_optims).

## 9. Why focus on decode?

<a href="/assets/slides/torch-compile-in-the-wild/slide-09.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-09.png" alt="Slide 9: Why focus on decode?" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Ordinary decode repeats a similar model call many times, so compilation cost can be amortized across generated tokens. Prefill has a different shape and repetition pattern. The slide describes a decode-focused path, but calling prefill compilation wasted work is too broad: repeated prompt shapes, long prompts, or a different serving strategy can make it worthwhile.

## 10. Diffusers and regional compilation

<a href="/assets/slides/torch-compile-in-the-wild/slide-10.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-10.png" alt="Slide 10: Diffusers and regional compilation" width="2400" height="1350" loading="lazy" decoding="async">
</a>

A diffusion model often repeats similar transformer blocks many times. Diffusers supports compiling those regions through `compile_repeated_blocks`, reducing the amount of code that needs fresh compilation. It also supports broader model compilation. Regional compilation is an option for balancing startup and runtime, rather than a restriction that Diffusers can never compile a whole model. See the [official guide](https://huggingface.co/docs/diffusers/optimization/fp16).

## 11. Reusing work across repeated blocks

<a href="/assets/slides/torch-compile-in-the-wild/slide-11.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-11.png" alt="Slide 11: Reusing work across repeated blocks" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The repeated-block list identifies classes that are good candidates for the regional path. Compatible instances can benefit from compiler cache reuse while keeping their own parameters. The exact model count in the slide is a snapshot, and reuse still depends on guards and configuration. Repetition creates an opportunity; it doesn't guarantee a free speedup for every architecture.

## 12. LoRA hotswapping

<a href="/assets/slides/torch-compile-in-the-wild/slide-12.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-12.png" alt="Slide 12: LoRA hotswapping" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Changing adapter values is easier for reuse when tensor shapes and the module structure stay stable. Diffusers can prepare LoRA storage for a chosen maximum rank and swap compatible adapters into it. The preparation happens before compilation. Adding newly targeted layers can still trigger recompilation, so the [hotswap requirements](https://huggingface.co/docs/diffusers/main/using-diffusers/loading_adapters) matter as much as the flag itself.

## 13. vLLM puts compilation near the model

<a href="/assets/slides/torch-compile-in-the-wild/slide-13.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-13.png" alt="Slide 13: vLLM puts compilation near the model" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The request passes through tokenization, scheduling, and worker preparation before reaching the compiled model computation. Most of that orchestration remains outside the graph. This boundary gives the serving engine control over requests while letting the compiler optimize the repeated tensor work. The [vLLM compilation walkthrough](/blog/vllm-and-torch-compile/) follows those layers in more detail.

## 14. A custom backend adds serving knowledge

<a href="/assets/slides/torch-compile-in-the-wild/slide-14.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-14.png" alt="Slide 14: A custom backend adds serving knowledge" width="2400" height="1350" loading="lazy" decoding="async">
</a>

vLLM's backend can split graphs, manage caches, apply inference-specific passes, and prepare callables for relevant shapes. CUDA graph wrappers then address repeated launch overhead. These are complementary responsibilities. A custom compiler integration can therefore do considerably more than selecting a different kernel generator.

## 15. SGLang's piecewise approach

<a href="/assets/slides/torch-compile-in-the-wild/slide-15.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-15.png" alt="Slide 15: SGLang&#x27;s piecewise approach" width="2400" height="1350" loading="lazy" decoding="async">
</a>

SGLang also uses graph partitioning around selected operations and CUDA graph execution for compatible pieces. The batch-size-versus-token-count contrast in the slide is simplified: token shapes matter to vLLM too. Backend defaults are configuration-dependent. The useful comparison is how each engine chooses split points, prepares shapes, and decides whether a region needs Inductor compilation.

## 16. Following SGLang's graph pieces

<a href="/assets/slides/torch-compile-in-the-wild/slide-16.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-16.png" alt="Slide 16: Following SGLang&#x27;s graph pieces" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The diagram shows a wrapped model call entering a custom backend, then separating supported regions from split operations. Compatible pieces can be captured and replayed while the split operations follow their own execution path. SGLang's [piecewise CUDA graph documentation](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/piecewise_cuda_graph.mdx) describes this architecture. Partial graph execution can be an intentional design, not just a failed attempt at full capture.

## 17. Distributed training changes the problem

<a href="/assets/slides/torch-compile-in-the-wild/slide-17.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-17.png" alt="Slide 17: Distributed training changes the problem" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Training adds gradients, optimizer state, and communication between devices. With sharded parameters, even making a layer's weights available can involve work around its forward call. A compiler integration has to preserve that lifecycle while looking for useful regions to optimize.

## 18. Hooks and wrapper order

<a href="/assets/slides/torch-compile-in-the-wild/slide-18.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-18.png" alt="Slide 18: Hooks and wrapper order" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Distributed wrappers can install hooks for gathering parameters, resharding, and synchronizing gradients. Whether those hooks run in the expected place depends on the call path and wrapper order. Dynamo can handle supported hooks, so the slide's blanket statement about hooks being untraceable is too strong. The actual difficulty is the behavior of the particular distributed integration.

## 19. Accelerate and the module call path

<a href="/assets/slides/torch-compile-in-the-wild/slide-19.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-19.png" alt="Slide 19: Accelerate and the module call path" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Wrapping a module with `torch.compile(module)` and compiling it in place with `module.compile()` are different integration choices. Accelerate's FSDP2 regional path uses the in-place form to preserve the call path needed by hooks installed later. Its [implementation explains the specific hook-ordering issue](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/other.py); it isn't a claim that compiled modules universally skip all hooks.

## 20. Different distributed backends need different preparation

<a href="/assets/slides/torch-compile-in-the-wild/slide-20.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-20.png" alt="Slide 20: Different distributed backends need different preparation" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The slide compares ordinary regional compilation with FSDP2 and DeepSpeed preparation. The latter paths need to account for sharding hooks, engine wrapping, and parameter handling, so one wrapper recipe isn't enough. This is a good example of a framework hiding necessary backend-specific work behind a simpler user-facing preparation API.

## 21. Choosing the right order

<a href="/assets/slides/torch-compile-in-the-wild/slide-21.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-21.png" alt="Slide 21: Choosing the right order" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Compilation and distributed wrapping both change how a model is called. If their order changes what the compiler can see or which callbacks execute, it can change correctness as well as performance. The broad approaches in the slide are to arrange the wrappers carefully or represent more distributed work explicitly inside the compiled graph.

## 22. Lightning remembers how to compile

<a href="/assets/slides/torch-compile-in-the-wild/slide-22.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-22.png" alt="Slide 22: Lightning remembers how to compile" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The Fabric path shown here records compilation options, unwraps the model while applying distributed setup, and reapplies compilation around the prepared model. That lets users request compilation before setup without manually reconstructing the final wrapper stack. The [Fabric documentation](https://lightning.ai/docs/fabric/2.2.0/advanced/compile.html) explains the reapplication behavior for DDP and FSDP.

## 23. DeepSpeed's deeper integration

<a href="/assets/slides/torch-compile-in-the-wild/slide-23.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-23.png" alt="Slide 23: DeepSpeed&#x27;s deeper integration" width="2400" height="1350" loading="lazy" decoding="async">
</a>

DeepCompile goes further by bringing parameter-management and communication work into a compiler-aware execution plan. The slides show an initial compilation followed by profiling and a later recompilation with more information. The illustrated step numbers and file count belong to that implementation snapshot. The broader tradeoff is spending more preparation effort during a long training run to improve repeated execution.

## 24. Why the metaclass appears

<a href="/assets/slides/torch-compile-in-the-wild/slide-24.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-24.png" alt="Slide 24: Why the metaclass appears" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The metaclass hook in the slide intercepts creation of AOTAutograd's internal compiled function so DeepSpeed can capture backward inputs needed by its integration. This is a very specific dependency on compiler internals. The [source implementation](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/compile/patch_compiled_func.py) includes version-specific handling and restoration logic, which helps explain the maintenance cost of reaching this deep into another framework.

## 25. Reinforcement learning has an outer world

<a href="/assets/slides/torch-compile-in-the-wild/slide-25.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-25.png" alt="Slide 25: Reinforcement learning has an outer world" width="2400" height="1350" loading="lazy" decoding="async">
</a>

An RL loop interacts with an environment, collects experience, and performs learning updates. Those activities don't all have the same execution properties. A useful compilation boundary can sit around the repeated tensor update even when environment interaction and episode bookkeeping remain in Python.

## 26. What can we compile in an RL loop?

<a href="/assets/slides/torch-compile-in-the-wild/slide-26.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-26.png" alt="Slide 26: What can we compile in an RL loop?" width="2400" height="1350" loading="lazy" decoding="async">
</a>

External simulators, side effects, resets, and changing trajectory lengths can make whole-loop capture difficult. That doesn't make every RL environment untraceable: tensor-based environments and supported paths can compile. For the pattern in this talk, the practical starting point is the update function, whose tensor computation repeats over sampled experience.

## 27. Compiling the learning update

<a href="/assets/slides/torch-compile-in-the-wild/slide-27.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-27.png" alt="Slide 27: Compiling the learning update" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The illustrated update groups loss computation, backward, and optimizer-related work behind one callable. Environment stepping and data collection stay outside. Depending on the operations and settings, that callable may still contain multiple captured regions. The useful boundary is the repeated learning work, not an assumption that the whole update must become one kernel or one graph.

## 28. Let initialization settle first

<a href="/assets/slides/torch-compile-in-the-wild/slide-28.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-28.png" alt="Slide 28: Let initialization settle first" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Early calls may initialize state or use shapes that don't represent the steady-state workload. TorchRL's [`compile_with_warmup`](https://docs.pytorch.org/rl/main/_modules/torchrl/_utils.html) runs a chosen number of eager calls before wrapping the function with `torch.compile`. This delays capture; it doesn't remove compilation cost or guarantee that later data will never require another specialization.

## 29. Data-dependent output shapes

<a href="/assets/slides/torch-compile-in-the-wild/slide-29.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-29.png" alt="Slide 29: Data-dependent output shapes" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Finding indices with the one-argument form of `torch.where(condition)` produces a result whose length depends on tensor values. That's different from knowing an input's shape before execution. Symbolic shape support can represent some such results, but using their sizes in Python control flow can still be difficult. The three-argument `torch.where(condition, x, y)` has different shape behavior.

## 30. Making TensorDict work with the compiler

<a href="/assets/slides/torch-compile-in-the-wild/slide-30.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-30.png" alt="Slide 30: Making TensorDict work with the compiler" width="2400" height="1350" loading="lazy" decoding="async">
</a>

A tensor container needs predictable structure, supported operations, and a way for compiler tools to understand its contents. The slide collects techniques used around TensorDict, including pytree integration and compile-aware execution paths. Its [module documentation](https://docs.pytorch.org/tensordict/stable/reference/nn.html) describes compilation support. The lesson is that container design can affect graph capture even when the tensors inside it are ordinary PyTorch tensors.

## 31. Domain libraries bring their own abstractions

<a href="/assets/slides/torch-compile-in-the-wild/slide-31.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-31.png" alt="Slide 31: Domain libraries bring their own abstractions" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Graph learning, computer vision, and medical imaging have operations and metadata that don't fit neatly into a plain tensor-only model. Those abstractions are useful to users. The integration challenge is to preserve their meaning while giving the compiler a tractable computation to capture.

## 32. Custom operations need a contract

<a href="/assets/slides/torch-compile-in-the-wild/slide-32.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-32.png" alt="Slide 32: Custom operations need a contract" width="2400" height="1350" loading="lazy" decoding="async">
</a>

A registered custom operator can remain opaque while exposing enough information for graph capture: schema, mutation behavior, and fake/meta behavior for output metadata. Training may also require autograd support. Tensor subclasses aren't universally unsupported either. The restrictions depend on the subclass behavior and compiler path, as PyTorch's [`__torch_function__` handling](https://github.com/pytorch/pytorch/blob/335414f462e6e79e335d4b43fab2c0589dd2cd32/torch/_dynamo/variables/torch_function.py) illustrates.

## 33. PyG generates more explicit Python

<a href="/assets/slides/torch-compile-in-the-wild/slide-33.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-33.png" alt="Slide 33: PyG generates more explicit Python" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The slide shows PyG's message-passing machinery using templates to produce concrete methods for a particular layer. That can move dynamic argument-collection work out of the hot path. Although the slide says import time, the drawn path goes through `MessagePassing.__init__`; the useful idea is generating the methods during setup before repeated execution.

## 34. Stable fields make capture easier

<a href="/assets/slides/torch-compile-in-the-wild/slide-34.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-34.png" alt="Slide 34: Stable fields make capture easier" width="2400" height="1350" loading="lazy" decoding="async">
</a>

A fixed-field structure makes it easier to reason about which values are present and how to retrieve them. Dynamically changing keys can introduce extra specialization or unsupported behavior. Python dictionaries themselves are supported in many compiled programs, so switching to a `NamedTuple` is a way to stabilize a particular interface, not a universal requirement.

## 35. Torchvision and MONAI take different routes

<a href="/assets/slides/torch-compile-in-the-wild/slide-35.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-35.png" alt="Slide 35: Torchvision and MONAI take different routes" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Torchvision's [Python ROI Align implementation](https://docs.pytorch.org/vision/main/_modules/torchvision/ops/roi_align.html) uses lazy compilation and provides a differentiable path with deterministic backward behavior. The MONAI example in the talk takes another route: move metadata handling outside the compiled tensor computation. These examples show two places to make a change—inside the operation or at the boundary around domain-specific values.

## 36. The tensor-subclass spectrum

<a href="/assets/slides/torch-compile-in-the-wild/slide-36.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-36.png" alt="Slide 36: The tensor-subclass spectrum" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The comparison records working paths, workarounds, and skipped cases observed in the talk's code review. It should not be treated as a permanent support table for current releases. TensorDict is also a container rather than simply another tensor subclass. What matters across these examples is whether the compiler can preserve each abstraction's required behavior.

## 37. Several levels of adaptation

<a href="/assets/slides/torch-compile-in-the-wild/slide-37.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-37.png" alt="Slide 37: Several levels of adaptation" width="2400" height="1350" loading="lazy" decoding="async">
</a>

A project might exclude one function from capture, add a compile-specific path, register an operator, or change its internal structure. Those choices have different costs. `allow_in_graph` also differs from an opaque custom operator: it bypasses Dynamo's inspection of a function, but downstream tracing can still enter it. Picking the escape hatch requires understanding which stage needs help.

## 38. Where should the boundary go?

<a href="/assets/slides/torch-compile-in-the-wild/slide-38.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-38.png" alt="Slide 38: Where should the boundary go?" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Across the examples, the useful region ranges from a repeated block to a model forward or a learning update. I read the diagram as a collection of workload-specific decisions. The right region has enough repeated computation to optimize while keeping difficult orchestration and state transitions manageable.

## 39. What I would carry into a new project

<a href="/assets/slides/torch-compile-in-the-wild/slide-39.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-39.png" alt="Slide 39: What I would carry into a new project" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Start by identifying repeated tensor work and the shapes it actually sees. Let lazy state initialize, inspect graph breaks and recompiles, then decide whether a smaller region or a better operator contract helps. Custom backends and compiler patches become easier to justify once a concrete limitation is visible. They don't have to be the first integration step.

## 40. Compilation becomes part of the design

<a href="/assets/slides/torch-compile-in-the-wild/slide-40.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-40.png" alt="Slide 40: Compilation becomes part of the design" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The pattern across these projects is that compiler integration influences interfaces, state management, and execution boundaries. Sometimes a small change is enough; sometimes a framework needs substantial machinery. Understanding what repeats and what must remain dynamic is more useful than assuming the same compile call belongs in the same place everywhere.

## 41. What have you had to change?

<a href="/assets/slides/torch-compile-in-the-wild/slide-41.png">
  <img src="/assets/slides/torch-compile-in-the-wild/slide-41.png" alt="Slide 41: What have you had to change?" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Thanks for following the ecosystem tour. If you've integrated `torch.compile` into a library, I'd be curious to hear where you drew the boundary and what made you move it. Those concrete decisions often explain more than a standalone speedup number.
