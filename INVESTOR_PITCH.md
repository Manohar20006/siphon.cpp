# Siphon Studio — Investor Overview

> **Siphon Studio turns storage bandwidth and model sparsity into practical,
> private AI capacity on hardware that is normally considered too small.**

## Executive summary

Siphon Studio is a local AI inference product for running large, sparse
Mixture-of-Experts (MoE) models on consumer, edge, and on-premises hardware.
It combines a desktop control surface, a terminal interface, a headless CLI,
live telemetry, resource-aware configuration, and the `siphon.cpp` inference
runtime in one product.

The core technical idea is simple: an MoE model does not use every expert for
every token. Siphon Studio keeps the model's active working set in GPU memory,
maintains bounded VRAM and RAM caches, and streams required expert weights from
fast local storage when they are needed. This allows a model whose complete
quantized artifact is much larger than available VRAM to remain useful locally,
without sending prompts to a cloud service.

Siphon Studio is not only a model runner and not only a UI. It is the product
layer that makes storage-aware inference understandable, configurable, and
operationally usable.

## The problem

### Large models exceed local GPU memory

Modern MoE models can have tens or hundreds of billions of total parameters,
while using only a much smaller active subset per token. Quantization reduces
the storage requirement, but a large model artifact can still be several times
larger than the VRAM available in a laptop or edge device.

When the whole model cannot remain resident, conventional approaches often
fall back to broad CPU/RAM offload or require a substantially larger GPU. The
result is lower throughput, unpredictable latency, difficult configuration,
and a high hardware or cloud bill.

### Cloud inference is not always acceptable

Healthcare, legal, public-sector, industrial, defense, and research workloads
may require local processing because of privacy, policy, connectivity, or
air-gap requirements. Even when cloud use is permitted, repeated inference
costs, data-transfer latency, and dependence on an external service can be
unattractive for small teams and edge deployments.

### Local inference is too operationally difficult

The user is normally left to assemble model files, runtime flags, cache sizes,
GPU settings, storage paths, and monitoring tools. A technically capable
runtime can therefore remain inaccessible to the people who could benefit from
private local AI.

## The Siphon Studio solution

Siphon Studio addresses the hardware and operational problems together.

1. **Inspect the model.** The model catalogue reads model metadata and exposes
   architecture, quantization, context, layers, expert counts, and runtime
   capabilities.
2. **Keep a useful working set resident.** Frequently used weights can remain
   in a bounded GPU cache, with additional expert data held in a bounded RAM
   cache when configured and available.
3. **Stream only what the router needs.** For supported MoE architectures, the
   runtime follows expert routing and fetches required data from local NVMe.
   On the Linux reference path, this uses NVIDIA CUDA and GDS/cuFile where the
   hardware, driver, filesystem, and model path are compatible.
4. **Respect the machine.** Resource-aware settings use detected VRAM and RAM,
   keep safety headroom, and warn or reject allocations that exceed the safe
   budget instead of silently pushing the system into failure.
5. **Make behavior visible.** The dashboard and chat surfaces expose live
   inference, cache, transfer, residency, and system telemetry so operators can
   distinguish actual runtime behavior from configuration assumptions.
6. **Operate the model as a product.** Users can import models, create or
   restore launch profiles, start and stop the runtime, change settings, chat
   with a running model, and inspect the result from one local application.

## What the product includes

Siphon Studio is the umbrella product. `siphon.cpp` is its inference runtime,
not a separate product competing with the Studio surface.

| Product layer | Purpose |
| --- | --- |
| **Desktop application** | Model catalogue, launch profiles, resource-aware settings, dashboard, chat, and live telemetry. |
| **Terminal TUI** | Lightweight local control for operators who prefer a terminal workflow. |
| **Headless CLI** | Scriptable start/stop, profiles, health checks, and machine-readable output. |
| **Inference runtime** | MoE routing, expert residency, NVMe streaming, caching, batching, and generation. |
| **Shared control core** | Keeps the desktop app, TUI, and CLI aligned instead of maintaining separate behavior. |

The desktop shell uses system GTK4/WebKitGTK components rather than bundling a
large browser runtime. That keeps the control surface lightweight while the
runtime remains independently usable in headless deployments.

## Who benefits

### Individuals and developers

People can experiment with capable local models using hardware they already
own, without first purchasing a datacenter-class GPU or sending private prompts
to a hosted service.

### Small teams

Teams can share a local model box for coding, analysis, document work, and
internal assistants. The dashboard makes capacity, cache pressure, and
throughput visible before a workload becomes a mystery.

### Private and regulated deployments

Organizations can keep data and model traffic on premises or at the edge. Siphon
Studio does not make an organization automatically compliant; it provides a
local execution foundation on which the organization's access control,
encryption, audit, and retention policies can be built.

### Offline and edge systems

Factories, field systems, laboratories, vehicles, and other constrained
environments benefit when inference can continue without a permanent cloud
connection.

## Why it matters

Siphon Studio changes the practical question from:

> “How much VRAM is required to hold the entire model?”

to:

> “What active working set, storage bandwidth, cache budget, and latency can
> this machine sustain?”

That shift matters because it can:

- lower the minimum hardware cost for useful local inference;
- make private and offline AI available in more locations;
- turn existing NVMe capacity into an active part of the serving system;
- provide predictable, bounded resource behavior instead of accidental swapping;
- reduce dependence on per-token cloud pricing and external availability; and
- give operators evidence about throughput, cache hits, transfers, and residency.

## How Siphon Studio differs from a conventional inference engine

Siphon Studio should not be positioned as “the only engine that can run large
models locally.” The defensible distinction is the combination of storage-aware
MoE execution and a complete operational product.

| Dimension | Conventional local inference focus | Siphon Studio focus |
| --- | --- | --- |
| **Primary assumption** | Keep as much of the model resident as available memory allows, or use general offload. | Treat storage, RAM, and VRAM as coordinated tiers for an MoE working set. |
| **Unit of movement** | Often layers, tensors, or broad model offload regions. | Routed expert data and other runtime-specific working sets. |
| **Storage path** | General host I/O or framework-managed staging. | Explicit NVMe-to-cache streaming, using GDS/cuFile on the Linux reference path when supported. |
| **Resource control** | Runtime flags and user-managed limits. | Detected system budgets, safety headroom, recommendations, validation, and launch profiles. |
| **Observability** | Usually token speed and basic logs. | Live generation, prefill/decode, cache, residency, transfer, system, and request-level telemetry. |
| **User experience** | Library, server, or command-line runtime. | Desktop application plus TUI and CLI around the same runtime and state. |
| **Deployment goal** | Make a model run. | Make constrained, private inference understandable and repeatable. |

The product's advantage is therefore not one isolated optimization. It is the
integration between model metadata, routing, cache policy, storage queues, GPU
synchronization, resource validation, and operator-facing telemetry.

## Evidence and current scope

The repository contains a Linux-first CUDA reference path and internal benchmark
evidence for Qwen3.6 35B-A3B in quantized GGUF form on an NVIDIA RTX 4050
Laptop GPU with approximately 6 GB of VRAM. Depending on cache, context,
storage, and runtime configuration, internal runs have reached roughly
17–20 tokens per second in decode.

Those numbers are reference measurements, not universal guarantees. Actual
performance depends on model architecture and quantization, SSD latency and
bandwidth, GPU and driver support, context length, cache budgets, batching,
concurrency, and operating-system behavior.

The current product direction is deliberately capability-gated:

- **Proven focus:** Linux, NVIDIA CUDA, fast local storage, and supported GGUF
  MoE model paths.
- **Product surface:** desktop model catalogue, model import and inspection,
  resource-aware settings, launch profiles, start/stop lifecycle, dashboard,
  chat, and live telemetry.
- **Architecture coverage:** several MoE model families are supported or under
  active validation; broad support is earned through metadata inspection and
  compatibility testing rather than assumed from a model name.
- **Windows path:** D3D12/DirectStorage-based support is an engineering
  workstream and should not be presented as having Linux performance parity
  until it is benchmarked and hardened.

This distinction is important to investors: Siphon Studio has a demonstrated
technical wedge, while cross-platform breadth, installer quality, and
enterprise hardening remain productization work.

## Defensibility

The long-term moat is the accumulated system knowledge required to make this
work reliably across models and machines:

1. **Runtime integration:** routing, cache residency, storage I/O, GPU
   synchronization, batching, and generation must agree at token time.
2. **Model capability knowledge:** imported models need architecture-aware
   inspection, safe defaults, and compatibility probes.
3. **Resource-aware operations:** recommendations and limits must reflect real
   VRAM, RAM, storage, context, and concurrency constraints.
4. **Telemetry and benchmarks:** per-model profiles, cache behavior, transfer
   behavior, and reference measurements improve recommendations over time.
5. **A unified product surface:** the desktop app, TUI, CLI, and runtime share
   one control model, reducing operational drift.

These capabilities become more valuable as the supported model set and hardware
matrix grow. They are also harder to reproduce than a single UI feature or a
single kernel optimization.

## Business opportunity

The initial commercial opportunity is private inference software and packaged
local AI systems for developers, small teams, and organizations that need
control over data and deployment.

Potential commercial forms include:

- per-device or per-deployment software licensing;
- enterprise support, validation, and on-premises deployment;
- reference appliances built from commodity GPU and NVMe components; and
- OEM or systems-integrator partnerships for edge AI products.

These are proposed routes to market, not claims of current revenue. The near-
term product goal is to turn the working Linux reference path into a dependable
installable product with clear hardware requirements and repeatable benchmarks.

## Roadmap to a production business

1. Harden the Linux reference product, packaging, diagnostics, and upgrade path.
2. Expand model import and architecture validation so new supported MoE models
   receive safe, model-specific recommendations automatically.
3. Productize multi-user scheduling, fairness, admission control, and workload
   isolation.
4. Benchmark and harden the Windows D3D12/DirectStorage backend rather than
   assuming parity with Linux.
5. Add enterprise deployment capabilities such as authentication, encryption,
   auditability, air-gap installation, and fleet management.
6. Publish reproducible performance reports across model families and hardware
   tiers.

## Investment thesis

Siphon Studio targets a clear inefficiency in local AI: capable sparse models
are often treated as impossible on smaller machines because their total weight
size is confused with their per-token active working set. Siphon Studio uses
that sparsity, fast local storage, bounded caches, and a purpose-built control
plane to make private inference more accessible.

Investment would accelerate the transition from a technically demonstrated
Linux runtime into a dependable cross-platform product: simpler installation,
broader model validation, multi-user operation, Windows backend maturity, and
enterprise deployment controls.

## One-line positioning

**Siphon Studio is the resource-aware local AI platform that lets large sparse
models use the GPU, RAM, and NVMe they actually need—without requiring the
entire model to fit in VRAM or sending private data to the cloud.**
