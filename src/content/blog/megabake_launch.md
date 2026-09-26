---
title: "Introducing MegaBake: a megakernel compiler for PyTorch"
date: 2026-09-26
description: "Why I started it, what it does today, and where I’d love some help."
---

*Why I started it, what it does today, and where I’d love some help. · 5-minute read*

<figure style="margin: 1.5rem 0; display: flex; flex-direction: column; align-items: center; text-align: center;">
  <a href="https://github.com/megabake/megabake" target="_blank" rel="noopener noreferrer">
    <img src="/assets/megabake.png" alt="MegaBake logo: a layered cake topped with a flame" width="220" height="220" style="display: block; width: 220px; height: auto; margin: 0 auto;" />
  </a>
  <figcaption><a href="https://github.com/megabake/megabake" target="_blank" rel="noopener noreferrer">Explore MegaBake on GitHub ↗</a></figcaption>
</figure>

Some of the most exciting GPU inference work uses megakernels: instead of handing every operation off to a new kernel, run much of the model in one cooperative kernel. For the right workload, that can remove launch gaps and let the GPU overlap data movement with compute. The result can be a lot faster than running each operation separately.

The catch is that building one by hand takes a lot of CUDA work. You have to write and tune the operations, work out how they share memory, and make sure the GPU blocks synchronize safely. It’s especially hard to take a regular PyTorch model and turn it into an efficient megakernel without rewriting parts of the model for a specific GPU. There are research projects working on this, but it’s not yet a routine, general-purpose step in the usual PyTorch workflow.

That gap is why I started MegaBake: I wanted to see if a compiler could take the PyTorch graph people already have and do more of this work for them. MegaBake is open source now, but it’s still a work in progress. Here’s how it works and what I’ve learned so far.

## 1. What is a megakernel?

When a PyTorch model runs on a GPU, its operations are usually split across many kernels: one for a linear layer, another for an activation, and more for normalization or attention. A megakernel groups many of these operations under one cooperative launch.

MegaBake captures a PyTorch model with `torch.export`, turns its graph into a schedule, compiles CUDA code, and launches the schedule as one cooperative kernel.

“One kernel” doesn’t mean one GPU block does all the work. Many blocks cooperate across the GPU, working through the scheduled operations and synchronizing along the way. In the current version, operations mostly run one after another. Overlapping work across operations is something I want to improve.

The Python side looks like this:

```python
compiled = megabake.compile(model, example_input)
output = megabake.run(compiled, model, example_input)
```

MegaBake currently supports common operations such as matrix multiplies, activations, normalization, attention, embeddings, and shape changes. If the model uses an operation MegaBake doesn’t support, it falls back to running the whole model eagerly in PyTorch. It can’t yet speed up the supported parts and leave the rest alone.

This doesn’t make the math itself cheaper. The kernel still has to launch, and blocks still have to synchronize. If a large matrix multiply is already the bottleneck, merging launches won’t fix it by itself.

I was already working on MegaBake when [Inferact published its TPU megakernel results](https://inferact.ai/blog/tpu-megakernels) on September 23 2026. Their Kimi K3 result is another strong demonstration of the idea: they report over 700 tokens per second with speculative decoding on a purpose-built megakernel across 16 TPU v7 chips, compared with 452 tokens per second on 16 GB200 GPUs. Those are Inferact’s numbers, not MegaBake’s. It’s a different model and hardware, but it shows again how much careful scheduling and data movement can matter. Their system is tuned for a specific setup; MegaBake is exploring whether a compiler can make this kind of work easier to approach from a standard PyTorch model.

## 2. Why start from PyTorch?

I want to make it easier to try megakernel execution on models people already have in PyTorch. MegaBake takes a regular `nn.Module` and its exported graph as input, rather than asking you to describe the model again in a custom CUDA instruction set.

That PyTorch connection is where I think MegaBake could be more adaptable. In the best case, you could change or try a model in PyTorch and let the compiler handle the supported operations. In practice, the supported set is still small, and new models will hit gaps. So this is the direction I’m aiming for, not something the current prototype has proved yet.

Stanford’s Hazy Research group has also shown what megakernels can do with custom Llama systems ([Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), [Llama-70B](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)); MegaBake’s angle is a PyTorch-native path from `nn.Module` and `torch.export` to a megakernel, with easier adaptation across models as the goal.

## 3. What still needs work?

Honestly, MegaBake is early and brittle. Some models compile; others hit graph or weight-handling bugs. When it does run, it can be slower than `torch.compile`. The README has a couple of wins on small synthetic models, near parity on one real model, and a substantial slowdown on another. Those are early snapshots, not evidence that MegaBake is generally faster.

There’s also newer compiler design work in the repository that is still just a plan; it hasn’t produced new GPU results. Please treat MegaBake as an experiment, not a production-ready replacement for PyTorch’s compiler. If it fails on your model, that’s useful to know.

## 4. How you can help

I’d especially appreciate people trying different models and reporting what works or breaks. A bug report with the model, input shape, GPU, and error is already helpful. Contributions to operator support, correctness, and reproducible benchmarks would be great too.

You can find the [code and issue tracker on GitHub](https://github.com/megabake/megabake). I hope this gives you a clear idea of what MegaBake is trying to do. If you work with PyTorch, CUDA, or compilers, I’d be glad to hear what you think and have your help making it better.
