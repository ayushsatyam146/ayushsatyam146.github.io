---
title: "Understanding vLLM: An architectural deep dive"
date: 2025-10-08
description: "From token generation and KV caches to PagedAttention, scheduling, and advanced serving features in vLLM."
---

<!-- Source presentation: vLLM presentation.pdf. Date is the blog publication date. -->

Running one prompt through an LLM is fairly easy. Keeping a GPU busy while many requests arrive, grow, and finish at different times is a much more interesting systems problem. This talk looks at how vLLM approaches that problem, starting with the memory needed to generate tokens.

I've kept every slide from the original architecture presentation below. Each one has a short explanation, and clicking the image opens the full-size diagram. The architecture drawings are simplified views of the implementation discussed in the talk.

## 1. What makes an inference system fast?

<a href="/assets/slides/vllm-architecture-deep-dive/slide-01.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-01.png" alt="Slide 1: What makes an inference system fast?" width="2400" height="1350" loading="eager" decoding="async">
</a>

The model is only one part of serving. You also need to decide which requests run together, where their state lives, and when memory can be reused. I wanted to connect those decisions to the GPU work they enable. That's the thread running through this vLLM walkthrough.

## 2. Why serving gets expensive

<a href="/assets/slides/vllm-architecture-deep-dive/slide-02.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-02.png" alt="Slide 2: Why serving gets expensive" width="2400" height="1350" loading="lazy" decoding="async">
</a>

GPU memory holds model weights, temporary workspaces, and the state of active requests. As prompts and generated sequences get longer, request state can become a serious limit on concurrency. The number of requests a GPU can support depends on the model and workload; the broader point is that unused or poorly managed memory has a direct cost.

## 3. Generating one token at a time

<a href="/assets/slides/vllm-architecture-deep-dive/slide-03.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-03.png" alt="Slide 3: Generating one token at a time" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The model processes a prompt, produces a distribution for the next token, and continues using the token selected from that distribution. This repeats until a stopping condition is reached. The growing context matters at every step, but a KV cache lets us reuse earlier attention state instead of recomputing the full prefix each time.

## 4. What attention needs from the past

<a href="/assets/slides/vllm-architecture-deep-dive/slide-04.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-04.png" alt="Slide 4: What attention needs from the past" width="2400" height="1350" loading="lazy" decoding="async">
</a>

For a new token, its query is compared with the keys available in its causal context, and the attention weights combine the corresponding values. That explains why old keys and values remain useful during decoding. Saving them trades memory for avoiding repeated computation, which becomes especially valuable across many generation steps.

## 5. The transformer has many layers

<a href="/assets/slides/vllm-architecture-deep-dive/slide-05.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-05.png" alt="Slide 5: The transformer has many layers" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The diagram places attention alongside the feed-forward computation inside a transformer. In a typical decoder, the attention state is maintained across many layers, so the cache cost isn't just one set of vectors for the whole model. Layer count, KV head count, head dimension, sequence length, and cache dtype all affect the memory required.

## 6. Where KV-cache memory gets wasted

<a href="/assets/slides/vllm-architecture-deep-dive/slide-06.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-06.png" alt="Slide 6: Where KV-cache memory gets wasted" width="2400" height="1350" loading="lazy" decoding="async">
</a>

If each request reserves a large contiguous region, unknown output lengths make allocation awkward. Some reserved space is never used, some waits for future tokens, and gaps can remain between allocations. The 20–40% figure on the slide describes the earlier systems studied in the [PagedAttention paper](https://arxiv.org/abs/2309.06180), rather than a universal utilization number for today's serving stacks.

## 7. What vLLM brings together

<a href="/assets/slides/vllm-architecture-deep-dive/slide-07.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-07.png" alt="Slide 7: What vLLM brings together" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Paged KV storage is an important starting point, but a serving engine also needs scheduling, efficient kernels, and ways to avoid unnecessary work. Features such as chunked prefill and speculative decoding target different bottlenecks. Which combination helps depends on the request mix, model, and hardware; a feature list alone doesn't tell us the resulting throughput.

## 8. What gets cached?

<a href="/assets/slides/vllm-architecture-deep-dive/slide-08.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-08.png" alt="Slide 8: What gets cached?" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The cache stores key and value vectors from previous tokens at the relevant attention layers. On a normal decode step, we compute the new token's state and attend over the available cached context. We still perform attention against that history. Caching saves recomputation of earlier K/V projections; it doesn't make reading a long context free.

## 9. PagedAttention

<a href="/assets/slides/vllm-architecture-deep-dive/slide-09.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-09.png" alt="Slide 9: PagedAttention" width="2400" height="1350" loading="lazy" decoding="async">
</a>

A sequence has a logical token order, but its KV blocks don't have to occupy adjacent physical locations. PagedAttention uses a mapping to find the right blocks while computing attention. This gives the memory manager more freedom to allocate space as a sequence grows, instead of finding one large contiguous reservation at the start.

## 10. The operating-system analogy

<a href="/assets/slides/vllm-architecture-deep-dive/slide-10.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-10.png" alt="Slide 10: The operating-system analogy" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The block table connects each request's logical blocks to physical KV blocks. Two requests can grow independently, drawing blocks from the same pool, and released blocks can be reused. It's similar to the separation between virtual and physical memory in an OS, although the serving engine manages these mappings for its own attention execution.

## 11. The main components

<a href="/assets/slides/vllm-architecture-deep-dive/slide-11.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-11.png" alt="Slide 11: The main components" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Tokenization turns text into model inputs. The engine coordinates requests, the scheduler selects work, the KV-cache manager tracks storage, and workers execute the model through an executor. Keeping these responsibilities separate helps answer a practical question: is a request waiting for compute, for memory, or for its turn in the schedule?

## 12. How the pieces connect

<a href="/assets/slides/vllm-architecture-deep-dive/slide-12.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-12.png" alt="Slide 12: How the pieces connect" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Follow a request from the entry point through input processing into the engine core. The scheduler and cache manager prepare work that the executor sends to workers, and output processing turns results back into responses. The host-memory cache drawn here represents an optional configuration path; CPU offloading isn't a required step for every request.

## 13. Initialization builds the runtime

<a href="/assets/slides/vllm-architecture-deep-dive/slide-13.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-13.png" alt="Slide 13: Initialization builds the runtime" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Before serving can start, the engine prepares its configuration, workers, model, scheduler, and cache resources. Device initialization and model loading happen on the worker side, while scheduling state is managed by the engine core. The drawing is an approximate initialization map, so individual constructors and process boundaries can differ between versions and deployment modes.

## 14. Following an actual request

<a href="/assets/slides/vllm-architecture-deep-dive/slide-14.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-14.png" alt="Slide 14: Following an actual request" width="2400" height="1350" loading="lazy" decoding="async">
</a>

This expanded diagram puts request processing next to scheduling and execution. A request enters the queues, receives a token budget and cache allocation, and participates in model steps until it finishes. The result of one step updates what can run in the next. That repeated feedback is what makes continuous batching useful for requests of different lengths.

## 15. Beyond the basic execution loop

<a href="/assets/slides/vllm-architecture-deep-dive/slide-15.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-15.png" alt="Slide 15: Beyond the basic execution loop" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Once we can schedule requests and manage their KV state, there are several opportunities to improve the service. Long prompts can be processed in pieces, repeated prefixes can reuse work, output formats can constrain sampling, and prefill can run separately from decode. Each feature changes a different part of the request's journey.

## 16. Chunked prefill

<a href="/assets/slides/vllm-architecture-deep-dive/slide-16.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-16.png" alt="Slide 16: Chunked prefill" width="2400" height="1350" loading="lazy" decoding="async">
</a>

A long prompt doesn't have to consume its entire prefill budget in one scheduling step. Splitting it into chunks gives the scheduler room to mix prompt work with ongoing decode requests. The tradeoff is about latency and resource sharing: finishing one prompt as quickly as possible and keeping existing streams responsive are not always the same objective.

## 17. Prefix caching

<a href="/assets/slides/vllm-architecture-deep-dive/slide-17.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-17.png" alt="Slide 17: Prefix caching" width="2400" height="1350" loading="lazy" decoding="async">
</a>

If two prompts begin with a compatible, identical prefix, the second request may reuse KV blocks already computed for it. The diagram shows blocks indexed by hashes and shared while needed. Matching requires the relevant execution context as well as token content. This saves repeated prefill work; it doesn't mean the rest of the response has already been generated.

## 18. Guided decoding

<a href="/assets/slides/vllm-architecture-deep-dive/slide-18.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-18.png" alt="Slide 18: Guided decoding" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Before sampling the next token, a structured-output backend can mask tokens that would violate the allowed output structure. The bitmask in the slide illustrates setting invalid logits to negative infinity. A finite-state machine is one useful model, but grammar implementations can be more general. Constraining syntax also doesn't guarantee that the generated answer is factually correct.

## 19. Separating prefill and decode

<a href="/assets/slides/vllm-architecture-deep-dive/slide-19.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-19.png" alt="Slide 19: Separating prefill and decode" width="2400" height="1350" loading="lazy" decoding="async">
</a>

Prefill processes many prompt tokens together, while ordinary decode advances active sequences a small amount at a time. Their resource needs can differ enough to justify separate instances, with KV state transferred between them. That can improve control over time to first token and inter-token latency, but transfer costs and load balancing become part of the design.

## 20. References for going further

<a href="/assets/slides/vllm-architecture-deep-dive/slide-20.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-20.png" alt="Slide 20: References for going further" width="2400" height="1350" loading="lazy" decoding="async">
</a>

The [vLLM codebase](https://github.com/vllm-project/vllm/tree/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb) connects these diagrams to an implementation, and the [PagedAttention paper](https://arxiv.org/abs/2309.06180) develops the memory-management argument. The original slides also point to the [Anatomy of vLLM article](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) and a [talk by Woosuk Kwon and Zhuohan Li](https://www.youtube.com/watch?v=5ZlavKF_98U). I would start by following one request through the scheduler and engine core.

## 21. Thanks for reading

<a href="/assets/slides/vllm-architecture-deep-dive/slide-21.png">
  <img src="/assets/slides/vllm-architecture-deep-dive/slide-21.png" alt="Slide 21: Thanks for reading" width="2400" height="1350" loading="lazy" decoding="async">
</a>

I hope the diagrams help connect attention, memory management, and scheduling into one system. If you work on serving, the part I'd be curious to hear about is where your workload runs into a limit first: prompt processing, decoding, or keeping enough requests resident in memory.
