---
title: "Why vLLM needed an IR: What it is and how it works"
date: 2026-04-16
description: "Understanding vLLM's semantic operations, implementation registry, dispatch policy, and late lowering into executable graphs."
---

<!-- Source presentation: vLLM, torch.compile and IR.pdf. Date is the blog publication date. -->

vLLM already uses `torch.compile`, so why add another intermediate representation? The issue becomes clearer when the same operation has a native PyTorch implementation, several custom kernels, and compiler passes that need to recognize all of them.

This is the slide-by-slide version of my vLLM IR talk. Click any slide for the full-size image. The explanations distinguish the design's core idea from features that were still proposed, such as autotuning across implementations.

## 1. Why another representation?

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-01.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-01.png" alt="Slide 1: Why another representation?" width="2400" height="1350" loading="eager" decoding="async">
</a>

I wanted to understand where the existing graph stopped being convenient for vLLM. An operation like RMSNorm has one intended meaning, but its graph can look very different depending on the selected implementation. If every compiler pass has to understand all those forms, adding another kernel becomes more work than it first appears.

## 2. The serving context

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-02.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-02.png" alt="Slide 2: The serving context" width="2400" height="1350" loading="lazy" decoding="async">
</a>

vLLM has to run models across different devices, dtypes, and execution configurations. That naturally creates several implementations of common operations. The serving engine needs to choose among them, while the compiler needs a representation it can analyze and transform. Those needs meet inside the model execution path.

## 3. What torch.compile already provides

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-03.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-03.png" alt="Slide 3: What torch.compile already provides" width="2400" height="1350" loading="lazy" decoding="async">
</a>

PyTorch supplies graph capture and compilation machinery. vLLM can use it to optimize supported tensor computation and combine it with custom kernels. The remaining problem is how vLLM-specific operations should appear to that machinery before a hardware-specific implementation has been chosen.

## 4. Where the IR fits in the compiler path

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-04.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-04.png" alt="Slide 4: Where the IR fits in the compiler path" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Dynamo captures the model, graph transformations prepare the computation, and Inductor eventually lowers and generates executable work. vLLM's representation lives within this process using FX nodes. It doesn't require replacing the entire PyTorch compiler stack; it gives vLLM's passes more meaningful operations to work with inside it.

## 5. Finding the model boundary again

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-05.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-05.png" alt="Slide 5: Finding the model boundary again" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The request still passes through the engine, scheduler, executor, and worker before reaching model execution. vLLM IR concerns the operations inside that computation. It doesn't become a new request scheduler or KV allocator. The stack diagram helps locate the scope before we look at the representation itself.

## 6. Building on the existing integration

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-06.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-06.png" alt="Slide 6: Building on the existing integration" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The wrapper, backend, shape dispatch, and CUDA graph machinery are already present in the compilation design. IR adds a more stable way to represent selected operations before they reach low-level implementation details. Graph splitting and CUDA graph capture remain separate concerns even though they interact with the same compiled model.

## 7. The original proposal

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-07.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-07.png" alt="Slide 7: The original proposal" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The [vLLM IR RFC](https://github.com/vllm-project/vllm/issues/32358) describes the motivation: simplify transformations and kernel registration by separating an operation's meaning from its implementation and dispatch. Reading that motivation first helps explain the API choices. The proposal also contains longer-term goals, so it shouldn't be read as a list of features all delivered at once.

## 8. Why compilation alone wasn't enough

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-08.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-08.png" alt="Slide 8: Why compilation alone wasn&#x27;t enough" width="2400" height="1350" loading="lazy" decoding="async">
</a>

A compiler can only transform the representation it receives. If RMSNorm appears as a collection of elementary operations on one backend and a different opaque custom call on another, a fusion pass has several patterns to handle. The extra IR gives that pass a common semantic operation to recognize before implementation selection changes the graph.

## 9. The questions to keep in mind

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-09.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-09.png" alt="Slide 9: The questions to keep in mind" width="2400" height="1350" loading="lazy" decoding="async">
</a>

There are four useful questions here: what was awkward about the old custom-op design, what Dynamo sees during tracing, when kernel selection happens, and which pieces are implemented. Keeping those questions separate helps avoid treating a cleaner graph representation as an automatic guarantee of faster kernels.

## 10. What is a semantic operation?

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-10.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-10.png" alt="Slide 10: What is a semantic operation?" width="2400" height="1350" loading="lazy" decoding="async">
</a>

A semantic operation defines what computation means, including its arguments and expected behavior. In vLLM IR, a native PyTorch implementation provides that definition and a reference path. The operation can then have several implementations that satisfy the same contract. This is useful for both compiler transformations and correctness testing.

## 11. What made the old CustomOp path awkward?

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-11.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-11.png" alt="Slide 11: What made the old CustomOp path awkward?" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Stateful layer dispatch and backend-specific graph patterns made matching operations harder. Mutating custom kernels also introduced functionalization details that passes had to account for. Custom operators themselves are useful and still participate in the newer design. The issue is how much implementation-specific structure a high-level pass needs to understand.

## 12. One meaning, several graph shapes

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-12.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-12.png" alt="Slide 12: One meaning, several graph shapes" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The diagram follows one layer into native decomposition, a CUDA kernel, a ROCm path, or a fused special case. A compiler pass looking for the same mathematical operation can encounter a different pattern in each case. That couples the pass to kernel choices that it may not otherwise need to care about.

## 13. Delay the implementation decision

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-13.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-13.png" alt="Slide 13: Delay the implementation decision" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The proposed separation lets early passes work with the operation's meaning and leaves concrete implementation selection for a later stage. A normalization-plus-quantization transformation can reason about those operations before provider-specific nodes appear. The chosen implementation still has to preserve the original contract after the graph is rewritten.

## 14. Four pieces working together

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-14.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-14.png" alt="Slide 14: Four pieces working together" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The semantic op defines behavior. The implementation registry records ways to execute it. Dispatch policy chooses an eligible implementation. Lowering replaces the semantic node with the selected implementation's graph. Each piece answers a different question, which is why the IR is more than a new name for a kernel registry.

## 15. Following an operation through the pipeline

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-15.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-15.png" alt="Slide 15: Following an operation through the pipeline" width="2400" height="1350" loading="lazy" decoding="async">
</a>

During capture, a registered operation can appear as a stable `vllm_ir` node. Transformations work on that representation, then lowering selects and traces an implementation using the graph's argument metadata. The resulting graph continues through compilation. The [`VllmIRLoweringPass` source](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/compilation/passes/ir/lowering_pass.py) is a useful place to follow that replacement.

## 16. Reading the color-coded pipeline

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-16.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-16.png" alt="Slide 16: Reading the color-coded pipeline" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The colors separate the semantic operation, registry, dispatcher, and lowering machinery. Follow the arrows from the stable node to the registry and back to the rewritten graph. Functionalization and clone cleanup also matter for implementations that reuse input storage: a functional representation needs a deliberate way to preserve its behavior while permitting that reuse.

## 17. Several implementations, one contract

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-17.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-17.png" alt="Slide 17: Several implementations, one contract" width="2400" height="1350" loading="lazy" decoding="async">
</a>

A native implementation and a custom kernel can compute the same operation while making different performance tradeoffs. Their outputs, shapes, dtypes, and relevant mutation behavior must agree with the operation's contract, allowing for supported numerical tolerances. A shared semantic name is useful only if implementations really honor that agreement.

## 18. What changes in the FX graph?

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-18.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-18.png" alt="Slide 18: What changes in the FX graph?" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Before lowering, the graph contains a semantic RMSNorm node. A native lowering can expand it into elementary tensor operations; another lowering can introduce allocation and a custom kernel call. Early passes see a common operation, while later stages see the selected implementation. Those stages have different information because they are solving different problems.

## 19. Why keep an implementation registry?

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-19.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-19.png" alt="Slide 19: Why keep an implementation registry?" width="2400" height="1350" loading="lazy" decoding="async">
</a>

One kernel may support only particular devices, layouts, or dtypes, while another covers more cases with different performance. The registry gives those implementations a common place to declare their availability and argument requirements. It also gives out-of-tree providers a way to add an implementation without requiring every model to learn a new dispatch path.

## 20. How dispatch chooses

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-20.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-20.png" alt="Slide 20: How dispatch chooses" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Priority configuration determines which implementations are considered first. Static availability and argument compatibility then determine what can run. Compile-time selection uses fake tensor metadata, while eager dispatch receives runtime arguments. In the [checked dispatcher](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/ir/op.py), an unavailable implementation left in the active priority list can raise an error; the slide's skip-and-continue drawing is a simplified policy view.

## 21. Who benefits from the separation?

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-21.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-21.png" alt="Slide 21: Who benefits from the separation?" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Kernel authors get a clearer registration and testing contract. Compiler authors get stable nodes for transformations. Platform authors can supply implementations without replacing entire layers. Users get more explicit control over selection. The checked [design document](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/docs/design/vllm_ir.md) still lists autotuning across implementations as future work, so priority-based dispatch shouldn't be confused with measuring every candidate automatically.

## 22. References for following the implementation

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-22.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-22.png" alt="Slide 22: References for following the implementation" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Start with the [RFC](https://github.com/vllm-project/vllm/issues/32358), then follow an operation from [`vllm/ir`](https://github.com/vllm-project/vllm/tree/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/ir) into the lowering pass. The slide also links to my [previous vLLM compilation talk](https://www.youtube.com/watch?v=qLtCDI6XOzg). Reading one operation across these files is more manageable than trying to understand the whole registry at once.

## 23. A useful way to read compiler designs

<a href="/assets/slides/vllm-torch-compile-and-ir/slide-23.png">
  <img src="/assets/slides/vllm-torch-compile-and-ir/slide-23.png" alt="Slide 23: A useful way to read compiler designs" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Thanks for following along. When I look at this design, the question I keep coming back to is which decisions should be made early and which should wait. Keeping an operation recognizable until the compiler has used that information is a useful idea well beyond vLLM.
