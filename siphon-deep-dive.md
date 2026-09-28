# Siphon.cpp

## How an AI model runs on your PC

When a model is created, its weights are stored in **FP16** (full precision). That means every weight takes up a lot of space, so the full model file is very large.

On a normal laptop or PC, there isn't enough memory to hold a full FP16 model. That's why we use **4-bit quantized models** instead.

Quantization is like compressing the model. The weights are reduced from 16-bit to 4-bit, making the file much smaller while keeping the model mostly as smart.

These compressed models come in a format called **GGUF**. This is the standard format used by **llama.cpp** and all the tools built on top of it.

So the rule is simple: **your model must fit inside your GPU memory (VRAM).**

For example, if a PC has **6 GB of VRAM**, it can only run a model that fits inside those 6 GB. A typical **8B or 9B model** quantized to 4-bit is around **5 GB** — so it fits. Realistically, you can't run anything bigger than that on 6 GB of VRAM.

## Another way: offloading layers to system RAM

There is a second way to run a model that is bigger than the VRAM cap.

The model is made of many **layers**. Here's the trick:

- Keep **as many layers as fit inside VRAM**.
- Move (offload) the **remaining layers** to system RAM.

This actually works pretty well — **but only if you offload just a few layers.**

The moment you offload too many layers, or use a bigger model with more layers, the speed drops drastically. It becomes far from usable.

So even though offloading lets you run a bigger model, there are two big problems:

1. **The speed drops.** The model has to move data between VRAM and RAM constantly, and that is slow.
2. **The whole PC becomes unusable.** While the model runs, the RAM is so busy that other applications lag, and the PC can even freeze sometimes.

So offloading to RAM is only really practical when you offload **a few layers** — never a lot.

There is one exception: **Apple MacBooks.**

Apple MacBooks have **unified memory** — the CPU and GPU share the same pool of memory. So a MacBook can offload more layers to RAM without hurting speed as much, because there is no slow copying between two separate memory pools.

On **non-Mac PCs** this doesn't work, because the GPU and CPU have **separate, dedicated memory pools**. Moving data between them is slow, so offloading only works when you offload just a few layers.

## Now, meet Siphon.cpp

So far we have seen the rules:

- A model must fit inside VRAM.
- Or, at best, a bit more by offloading a few layers to RAM.

But what if there is still another way?

Imagine a PC with **6 GB of VRAM** and **16 GB of RAM**. Together, that is **22 GB** of memory.

Following the rules above, we could only run a model that fits inside this 22 GB cap. And realistically, even less than that — because the operating system (like Ubuntu) itself uses about **2 to 3 GB**. So the real usable model size would be below **18 or 19 GB**.

Now here is the big claim:

> What if we could **comfortably run a model that is bigger than 22 GB**?

That sounds like fiction, but it is **true**. It is possible — by using **MoE (Mixture of Experts) models** with **Siphon.cpp**.

That is exactly what this document is about: how Siphon.cpp makes that possible.

## Dense models vs MoE models

Before we go any further, you really need to understand this next part. **If you miss this concept, you will not understand anything about Siphon.cpp.** So take it slowly.

### What is inside a model layer?

Every AI model is made of layers. Inside every layer there is a component called the **FFN** (Feed-Forward Network), also called the **MLP** (Multi-Layer Perceptron).

An FFN is basically an advanced **ANN** (Artificial Neural Network). It has **three matrices**: **up**, **gate**, and **down**. You don't need to know exactly what each one does. Just remember: **an FFN has three matrices.**

The FFN does the important work. It is the part that:

- transforms the representation of a token, and
- learns from data and stores the knowledge and facts inside the model.

So the FFN is like the "brain" of the model. That is also why the FFN makes up about **80% to 90% of the entire model's size.** Almost all the model's size is just these FFNs.

### Dense models

In a **dense model**, every layer has **one giant FFN** — one big neural network with its three matrices.

Here is the key thing about dense models:

> During computation, **the entire FFN is active for every token.**

No matter what the token is, the whole FFN is loaded, read, and used. That means every single token needs:

- all the FFN weights to be read from memory, and
- all the FFN computation to be done.

This is the problem with dense models: **you pay the full cost for every token.** Bigger model = bigger FFN = slower, even though most of that work is "wasted" on tokens that don't need it.

Examples of dense models are Qwen3.5 9B, or Qwen3.6 27B (the dense variant).

### MoE models

**MoE** (Mixture of Experts) models solve this problem.

Instead of one big FFN per layer, each layer's FFN is **split into many smaller FFNs.** Each small FFN is called an **expert.**

#### Why are they called "experts"?

Because each small FFN has its own **specialization.** One expert might be specialized in coding tasks, another in science, another in medical knowledge.

So it is not like every expert knows everything. Each expert has a specific specialty. The model mixes the right experts for the task — hence the name **Mixture of Experts.**

#### How many experts per layer?

Every layer has its own set of experts. A model with 40 layers can have **up to 256 experts in each layer.**

But the model does **not** use all of them. A small predictor called the **router** sits in each layer and picks only a **fraction** of the experts — called the **top-K.** K can be 4, 8, or 10 — whatever the model was trained with.

For example:

- **Qwen3.6 35B-A3B** has **41 layers** and **256 experts per layer**. Its router selects only **8 experts** out of 256 in each layer. If the next token is coding-related, the router picks the coding experts. If it's a medical token, it picks the medical experts.
- **Qwen3-Coder-Next 80B-A3B** has **512 experts per layer**, and its router picks the **top 10.**

So the number of experts and the top-K **varies from model to model.**

#### The shared expert

There is one more detail: most MoE layers also have a **shared expert** — one expert that is **always active** for every token, no matter what. It runs alongside the selected top-K experts. It is there to handle the general, everyday work that every token needs, while the routed experts handle the specialized work.

#### What does "A3B" mean?

Model names like "35B-A3B" have a hidden meaning:

- **35B** = total parameters of the model.
- **A3B** = only **3 billion parameters are active** for each token.

So for Qwen3.6 35B-A3B: out of all 35B parameters, only about **3B parameters are actually used per token.**

Those 3B active parameters include: **attention weights, embeddings, norms, residuals, and the few experts the router selected.**

That is the whole difference from a dense model: a dense model activates **all** of its parameters for every single token, while an MoE model activates only a small slice of them.

### The difference, simply put

| Dense model | MoE model |
|---|---|
| One giant FFN per layer | Many small FFNs (experts) per layer |
| Every token uses the entire FFN | Every token uses only a few experts |
| Bigger model = much slower | Can be huge, but stays fast |

### Why MoE makes sense

Because only a few experts are active per token, an MoE model can be **very large** (lots of total parameters, lots of stored knowledge) without being **very slow** (each token only touches a small part of the model).

That is why MoE models are so widely used today — they give you the brain of a huge model at the speed of a small one. And, as you will see next, this "only a few experts are used" idea is exactly what Siphon.cpp relies on.

### But here is the real bottleneck

MoE only reduces the **computation.** That part is solved.

But there is still one huge problem: **you still need enough memory to load the model.**

Even if only a few experts are active per token, the model weights still have to be **in memory** before you can run anything. And MoE models are big. That is the real bottleneck.

Usually we simply do not have that much memory on a laptop. To load one of these big MoE models comfortably, you would normally need **multiple GPUs**, or one giant **datacenter GPU**, just to hold the model.

Let's make this real with an example. Take **Qwen3.6 35B**. Even quantized to **4-bit**, it is about **22.7 GB**. So we need roughly **23 GB of memory just to load the model itself.**

Now remember our earlier example: **6 GB VRAM + 16 GB RAM = 22 GB combined.** We still cannot even load the model — it would not even fit, let alone run.

Could we use lower quantization, like 3-bit or 2-bit? Yes, but then we would **lose most of the model's quality.** That is not a real answer.

So it genuinely feels **impossible** to run a model this big comfortably on a laptop. And that is exactly the problem Siphon.cpp was built to solve.

## The key innovation: split the model in two

The trick to running a model bigger than your available memory is to **split the model into two parts.**

Here is why this works. We know an MoE model only needs a few experts per layer. But it still needs the **non-expert weights** — things like attention weights, embeddings, norms, and residuals — for **every token, in every layer.**

The good news: these non-expert (dense) weights are **really small** — only about **10% of the whole model.**

So here is what we do:

1. **At startup**, while loading the model, we load **only the dense weights** into VRAM. (The shared experts are also counted as dense weights here, because they need to be resident in every layer.)
2. We **leave the expert weights on the SSD**, because we do not have enough memory to load them.

Then, during computation:

3. Each layer's router selects its **top-K experts**.
4. We **stream those experts from the SSD to VRAM on demand**, right when they are needed.

That is how we can run models that are **bigger than the available memory** — the big expert weights never sit in memory all at once. They come and go as needed.

But this technique has its **own bottleneck** — and that is what we will talk about next.

## The problem: transfer latency

The main problem is **transfer latency** — it takes time to stream the required experts from the SSD to VRAM.

The standard way to move experts is step by step, through RAM:

> **SSD → RAM → VRAM**

On paper, the speeds look nice:

- A **PCIe Gen4 NVMe SSD** reads at around **6.6 GB/s**.
- The **RAM to VRAM** transfer runs at around **16 GB/s**.

Let's check with our Qwen3.6 35B example:

- Total model size: **22.7 GB**
- Dense weights: only about **2.4 GB**
- Everything else is expert weights.
- Each expert (out of 256 per layer) is about **1.8 MB** in size.

1.8 MB looks tiny. On paper, it seems like it should stream in instantly.

But here is the real problem: we still have to move it **through RAM**, and that creates **transfer latency.** And those nice on-paper speeds do not actually hold up for small transfers:

- The NVMe speed **drops to about 2–3 GB/s**.
- The RAM → VRAM speed **drops to about 10 GB/s**.

Why? Because those maximum speeds are only reached for **large reads**. For tiny reads like our 1.8 MB expert, the real speed is much lower — the transfer setup and latency costs dominate.

With all these delays, the GPU **sits idle half of the time**, waiting for experts to arrive. And the overall speed of the model **drops drastically.**

## The fix: stream experts directly from SSD to VRAM

To fix this, we need something that streams experts **directly from the SSD to VRAM** — skipping RAM entirely.

The problem is, **Windows** has no software or kernel support to stream data directly from the SSD to VRAM. So Windows is out of the question for this.

That is why **Linux (Ubuntu)** is chosen. Linux has something called **GPU Direct Storage** (also known as **cuFile**), which was developed by **NVIDIA** itself.

cuFile lets us stream data **directly from the SSD to VRAM**, without going through RAM and without any of those slowdowns.

By using this, we can **cut the transfer latency by a lot.** The experts arrive quickly, the GPU stays busy, and the model runs much faster.

But cuFile alone is **not enough.** We still need other tricks to scale the speed up. One of them is the **VRAM cache.**

## The VRAM cache (hot expert cache / L1 cache)

Remember our Qwen3.6 35B numbers:

- Dense weights are only **2.4 GB**.
- We have **6 GB of VRAM** total.

So after loading the dense weights, we still have about **3.6 GB free**. Out of that, we allocate roughly **3 GB for a VRAM cache.**

What does this cache do? It works because MoE models have a special habit:

> MoE models **reuse a lot of their previously used experts**, instead of calling for new unique experts all the time.

Think back to the router: it only picks coding experts when the next token is a coding token. So while the model is writing code, it keeps needing the **same coding experts** over and over — the ones it already used for the previous coding token.

So instead of throwing each expert back to the SSD right after using it, we **keep the most recently used and most frequently used experts in the VRAM cache.** That way:

- We do not have to stream them again from the SSD.
- The model can just grab them from the cache when needed.

When the cache gets full, some experts get evicted to make room — using **LRU** (Least Recently Used) and **LFU** (Least Frequently Used) style decisions. It is a bit like how a computer does **paging** — same idea, not exactly the same thing.

With our **3 GB cache**, we can hold about **1,500 experts** in VRAM at once.

This cache is the **hot expert cache** — also called the **L1 cache.**

The remaining **600 MB** of free VRAM is used for the **KV cache** and other necessary processes.

## Where we stand so far

At this point, we have already scaled the speed a lot:

- Before any of this, decode speed was around **3–4 tokens/sec**.
- After using cuFile and the VRAM cache, we pushed it **past 10 tokens/sec**, and it settled at a stable **11 tokens/sec**.

But we can push it even further. This is where the **RAM cache** comes in.

## The RAM cache (warm cache / L2 cache)

The RAM cache is the second level of caching — the **warm cache**, also called the **L2 cache.** It has **two types: static and dynamic.**

### Static RAM cache

First, we need a list of "popular" experts. Here is how it is built:

1. We ran the model across **a lot of coding tasks** (since we mostly use this model for coding).
2. We **traced** which experts were used most frequently for coding.
3. We created an **expert suite** — a list containing the IDs of the most frequently used coding experts.

Then, at **startup**:

- The RAM cache is given an adjustable size — for example, **4096 MB (4 GB)**.
- That 4096 MB cache is **filled with the experts listed in the expert suite.**

So the warmest, most-used coding experts are already sitting in RAM, waiting.

### How an expert is looked up (the flow)

When a token comes in, this is the exact flow for each expert the router picked:

```
        Token comes in
              |
              v
    Router selects top-K experts
              |
              v
    For each selected expert:
              |
              +-----> Is it in the VRAM cache (L1)?
              |              |
              |              +--- YES ---> HIT  ---> use it directly from VRAM
              |              |
              |              +--- NO  (L1 MISS)
              |                         |
              |                         v
              +-----> Is it in the RAM cache (L2)?
                         |
                         +--- YES ---> HIT  ---> quickly copy it to VRAM, then use it
                         |
                         +--- NO  (L2 MISS too)
                                    |
                                    v
                         Stream it directly from the SSD
                         to VRAM using cuFile, then use it
```

To put it simply:

- **VRAM cache hit** → use it, nothing to transfer.
- **VRAM miss, but RAM cache hit** → copy from RAM to VRAM (fast).
- **Both miss** → pull it straight from the SSD with cuFile (slowest path).

### Dynamic RAM cache

The static cache is filled from the expert suite. The **dynamic** cache works differently — it **fills itself over time:**

- When an expert is **evicted from the VRAM cache**, instead of throwing it back to the SSD, we **store it in the RAM cache**.
- Later, if that expert is needed again, we get a **hit** and copy it from RAM back to VRAM quickly.

So the dynamic cache learns which experts to keep, based on how the model is actually being used right now.

## The results

By adding the RAM cache, we pushed the speed even higher:

- **Decode: past 15 tokens/sec**
- **Prefill: 30 tokens/sec**

Just to be clear on the two speeds:

- **Prefill** is the **reading speed** — how fast the model processes the input prompt tokens.
- **Decode** is the **generation speed** — how fast the model produces the output tokens.

## One more lever: MTP (Multi-Token Prediction)

There is one more trick that helped us push the speed even higher: **MTP**, which stands for **Multi-Token Prediction.**

Normally, a model generates text **one token at a time** — it cannot start the next token until it has finished the previous one. That is why generation feels slow.

MTP changes that. With MTP, the model **guesses several next tokens at once** — like writing a few words ahead instead of one letter at a time. Then it checks all the guesses together in a single step.

Here is the clever part:

- If the guesses are **correct** (which happens a lot — for example, when the model is confidently writing code), the model earns **several tokens in a single step.** So it finishes the whole sentence much faster.
- If a guess is **wrong**, the model just discards the wrong ones and continues normally.

So MTP does not change what the model says — it just lets the model **write faster**, by doing more than one token per step when it can.

With MTP added, our speeds jumped again:

- **Decode: past 20 tokens/sec**
- **Prefill: past 50 tokens/sec**
- And when the cache is **fully warm** (a lot of experts are being reused), decode even reaches **25–30 tokens/sec.**

### The main lever: lowering top-K from 8 to 4

There is one more lever that gave us the biggest speed boost — and it is a simple one.

Remember that the router picks a certain number of experts per layer (the **top-K**). For Qwen3.6 35B, the default is **8 experts** per layer.

We tried lowering it to **4 experts** per layer instead.

And it turned out great:

- It showed **no real quality loss**.
- It gave us a **significant speed increase** — because fewer experts per token means less compute and, most importantly, **less streaming load** (fewer experts to pull from the caches and the SSD for every single token).

So cutting the default top-K from 8 to 4 was one of the main levers behind the speed we achieved.

There are also other factors, hacks, and techniques that contributed to this achievement — but these are the core ones.

## To be clear about the target

Siphon.cpp was originally developed to run the **Qwen3.6 35B-A3B** model. Every speed mentioned above was measured on that model.

And the whole thing was built and tested entirely on a **laptop with 6 GB of VRAM and 16 GB of RAM** — the exact hardware we used as our example from the start.

## Other supported models

Siphon.cpp currently also supports **two more models:**

1. **KAT-Coder V2.5 Dev 35B** — this is a **coding-finetuned version of Qwen3.6**, and it runs at **around the same speeds** as Qwen3.6.

2. **Qwen3-Coder-Next 80B-A3B** — and this one is a **huge milestone.** This model alone is about **47 GB on disk**, yet it runs on this laptop at **7 tokens/sec.**

Think about that for a second: running an **80B model entirely on a 6 GB VRAM laptop** is something that would normally be **far from reality.** Siphon.cpp makes it possible.

And the journey is not over — we are still looking forward to running **even bigger models** and scaling the speed **even further.**

## Why the name "Siphon.cpp"?

This was just a thorough explanation of Siphon.cpp — but why that name?

Think about a real **siphon**: a tube that moves liquid from one place to another, continuously, without any pump.

That is exactly what Siphon.cpp does — but with **experts instead of liquid.** It continuously siphons the expert weights up from the SSD into the GPU, on demand, as the model runs. No need to hold everything in memory at once — just siphon what you need, when you need it.

Hence the name: **Siphon.cpp.**