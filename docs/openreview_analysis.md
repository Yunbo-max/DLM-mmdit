# OpenReview Reviewer Analysis: Diffusion Language Model Gaps

## Overview

This document synthesizes reviewer comments, rejected paper lessons, and reviewer suggestions from OpenReview for papers related to diffusion language models (DLMs), organized by paper. The goal is to identify: (1) what went wrong with rejected papers, (2) reviewer suggestions pointing to missing work, and (3) papers cited by reviewers as better alternatives.

---

## 1. CCDD: Coevolutionary Continuous Discrete Diffusion (REJECTED)

**Paper:** "Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner"
**Venue:** ICLR 2026 | **Status:** REJECTED
**Scores:** 4 (shnm) / 2 (SRFH) / 6 (8q5Z) / 4 (RTsX)
**OpenReview:** https://openreview.net/forum?id=mwAkJJ4NBD

### Why It Was Rejected (Area Chair Summary)

> "The rebuttal only partially resolves key reviewer concerns. On novelty, a reviewer argues that combining discrete and continuous diffusion is not totally new, so the current positioning needs a clearer, more explicit differentiation from prior hybrid discrete+continuous diffusion work. On evaluation and practicality, several reviewers remain unconvinced because the comparison scope is still not sufficiently comprehensive, and the paper still lacks the efficiency accounting reviewers explicitly asked for."

### Key Reviewer Criticisms

**Reviewer shnm (Rating: 4):**
- "The experimental scope is limited - this is a major one. While results on LM1B/OWT (ppl) are compelling, the paper lacks evaluation on complex reasoning tasks (e.g., Sudoku, coding or even GSM8k) that it claims DLMs excel at."
- "The analysis of CCDD's architecture variants (MDiT/MMDiT/MoEDiT) is underdeveloped: Table 2/3 only compare performance but not efficiency (e.g., FLOPs per step, inference time)."
- "There is no comparison to loop transformers."

**Reviewer SRFH (Rating: 2):**
- "Definitions are underspecified: 'decision spaces,' 'representation spaces,' and 'combinatorial complexity in decoding' are invoked without precise formal definitions."
- "The claim that CCDD 'preserves full semantics in previous denoising steps' while masked DDMs 'discard' them lacks a formal definition of 'semantics' and empirical evidence isolating this factor."

**Reviewer 8q5Z (Rating: 6):**
- "Despite fruitful theoretical discussions, the empirical evaluation is weak. The author only do some toy scales experiment (lm1b) and evaluate with metrics like perplexity."
- "Combining discrete and continuous diffusion is not a new idea, which has been studied in previous work like **Diffuseq-v2**. The author should include discussions on these related work."

**Reviewer RTsX (Rating: 4):**
- "The technical aspects of increased training and inference costs were not adequately explored. Algorithm 2 indicates that CCDD requires denoising with both DDM and CDM at each sampling step, yet CDM outputs appear to be discarded during inference."
- "The model's expressivity might not be the most compelling argument... the real advantage of integrating DDM with CDM likely lies in improving the representational efficiency of latent embeddings... the CDM's auxiliary loss can facilitate learning of more efficient latent embeddings through continuous domain learning, which may mitigate information loss caused by logit quantization."
- "With the same architecture, comparing pure DDM, pure CDM and CCDD [ablation needed]."

### Papers Cited by Reviewers as Missing Comparisons
- **DiffuSeq-v2** (hybrid discrete+continuous diffusion, prior work)
- **MDLM** and **SEDD** (as stronger baselines)
- Looped transformers

### Lessons for Our Work
1. **Must include efficiency metrics** (FLOPs, inference time, memory) alongside accuracy
2. **Must demonstrate on downstream tasks** beyond perplexity (reasoning, conditional generation, text editing)
3. **Must clearly differentiate** from prior hybrid continuous-discrete approaches
4. **Must include ablation** isolating contribution of continuous vs. discrete components
5. **Reviewer RTsX's insight is gold:** "The real advantage lies in improving representational efficiency of latent embeddings... mitigating information loss caused by logit quantization" -- frame contribution as representation improvement, not expressivity

---

## 2. DICE / Discrete Inversion (WITHDRAWN)

**Paper:** "Discrete Inversion: A Controllable Latent Space for Multinomial Diffusion and Masked Generative Models"
**Venue:** ICLR 2025 | **Status:** WITHDRAWN
**OpenReview:** https://openreview.net/forum?id=NFEnBqknoX

### What We Know
- First approach to enable precise inversion for discrete diffusion models
- Records noise sequences and masking patterns during forward diffusion for accurate reconstruction and controlled edits
- Evaluated on VQ-Diffusion, Paella, and RoBERTa
- Withdrawn before review completion -- suggests authors identified issues or planned resubmission

### Implications for Our Work
- Text editing via discrete diffusion inversion is an open problem
- The withdrawal suggests technical challenges in making this work robustly
- This is a gap we can potentially fill with our latent-space approach

---

## 3. Simple Guidance Mechanisms for Discrete Diffusion (ACCEPTED)

**Paper:** "Simple Guidance Mechanisms for Discrete Diffusion Models"
**Venue:** ICLR 2025 Poster | **Status:** ACCEPTED
**Scores:** 6 (LFVk) / 6 (2Wfu) / 6 (9Khq) / 3 (S2DB) -- Final: 6/6/6/3
**OpenReview:** https://openreview.net/forum?id=i5MrJ6g5G1

### Meta-Review Summary
> "The reviewers raised concerns about novelty compared to concurrent works, particularly MDLM, leading to scepticism about the contributions, the performance of UDML on large-scale language tasks (e.g., LM1B) showed limited improvement compared to MDLM, scepticism about scalability, and missing complexity analysis and runtime comparisons."

### Key Reviewer Criticisms

**Reviewer LFVk (Rating: 6):**
- "The primary limitation is the scope of improvements being mainly restricted to multinomial diffusion with **small vocabulary settings, with a persistent performance gap in larger vocabulary NLP tasks**."
- "The paper lacks computational complexity analysis and detailed runtime comparisons."

**Reviewer S2DB (Rating: 3, most critical):**
- "I very respectfully disagree with the author's contributions. The uniform prior and the corresponding [formulation] are widely known and exist in the literature. I challenge the authors to demonstrate how their formulation differs from **Discrete Flow Matching (Gat et al. 2024)** and **Dirichlet Flow/Diffusion (Stark et al.)**"
- "The fact that we need to compute the normalization constant means that **this cannot be extended to actual larger-scale systems** where we would want to apply discrete diffusion models."
- "Note that **Dirichlet Flow Matching** (Stark et al. 2024) point out this exact issue with uniform prior paths, and hence suggest their Dirichlet paths."

**Reviewer 9Khq (Rating: 6, raised from 5):**
- "Very similar discrete diffusion works are not discussed and compared: [Zheng et al. 2023] A Reparameterized Discrete Diffusion Model and [Zhao et al. 2024] Unified Discrete Diffusion for Categorical Data"

### Key Author Rebuttal Results (important data)
- UDLM outperforms MDLM **specifically in guidance and fast-sampling settings**
- At higher guidance strength (gamma=2), UDLM achieves FID 9.07 vs MDLM 9.43 on ImageNet
- UDLM is much more robust to reduced sampling steps: at T=32, UDLM gets FID 20.34 vs MDLM 32.86
- "Diffusion models with our guidance outperform AR" on controllable NLP tasks (F1: 86 vs 20 for AR)

### Papers Cited by Reviewers
- **Discrete Flow Matching** (Gat et al. 2024) -- reviewer says formulations are equivalent
- **Dirichlet Flow Matching** (Stark et al. 2024) -- suggested as better path for large vocabularies
- **Zheng et al. 2023** "Reparameterized Discrete Diffusion" -- similar approach
- **Zhao et al. 2024** "Unified Discrete Diffusion for Categorical Data"

### Lessons for Our Work
1. **Guidance for large vocabularies remains unsolved** -- normalization constant is a bottleneck
2. **Uniform noise diffusion enables better guidance** than absorbing (masking) diffusion
3. **Fast sampling benefits of guidance** are a strong selling point
4. **Must benchmark against Discrete Flow Matching** as a baseline

---

## 4. EDLM: Energy-Based Diffusion Language Models (ACCEPTED)

**Paper:** "Energy-Based Diffusion Language Models for Text Generation"
**Venue:** ICLR 2025 Poster | **Status:** ACCEPTED
**Scores:** 8 (LAtw) / 8 (r3kY) / 6 (9hNB) / 5 (wiWZ)
**OpenReview:** https://openreview.net/forum?id=sL2F9YCMXf

### Key Reviewer Criticisms

**Reviewer LAtw (Rating: 8):**
- "This method requires a pre-trained discrete diffusion model, which increases the overall computational requirements. It may be **unfair to compare it directly with simpler methods like MDLM**."
- "While the proposed method reduces the Gen PPL metric, it also **decreases the entropy** of generated texts."
- Suggested: "A separate figure with Entropy/Gen PPL axis would address concerns about the notably reduced entropy."

**Reviewer r3kY (Rating: 8):**
- "There is insufficient examination of relevant prior research also involving EBMs for language modeling, e.g., [Deng et al. 2020] **Residual Energy-Based Models for Text Generation**."
- "The bidirectional attention mechanism in transformers can already capture dependencies among tokens. Feeding the decoded sequence back (i.e., the next denoising step) could potentially identify erroneous tokens. There is extensive literature on **filtering and remasking tokens** at each denoising step."
- Cited: **Mask-Predict** (Ghazvininejad 2019), **Step-unrolled Denoising Autoencoders** (Savinov 2021), **Reparameterized Discrete Diffusion** (Zheng 2023)

**Reviewer 9hNB (Rating: 6):**
- "The core technique introduced in this work has been present in the literature for a long time... The energy function and Importance Sampling method are **identical to reranking (or noisy parallel decoding) methods used in MLM generation/non-autoregressive generation**."
- Cited: **Non-autoregressive Neural Machine Translation** (2017), **Mask-Predict** (2019), **Fully Non-autoregressive NMT** (2020)

**Reviewer wiWZ (Rating: 5):**
- "This paper only evaluates perplexity, and it remains uncertain whether such energy-based reranking **translates effectively to downstream tasks**."
- "I tend to have concerns about the **inference cost** linked to repetitive sampling and energy function computations."

### Lessons for Our Work
1. **Energy-based reranking doubles parameters** -- must account for this in comparisons
2. **Entropy vs. quality tradeoff** must be measured and reported
3. **Remasking/filtering** at each step is a well-established alternative to energy functions
4. **Must evaluate on downstream tasks**, not just perplexity

---

## 5. MDLM: Simple and Effective Masked Diffusion Language Models (ACCEPTED)

**Paper:** "Simple and Effective Masked Diffusion Language Models"
**Venue:** NeurIPS 2024 | **Status:** ACCEPTED
**Scores:** 5 (p1DW) / 5 (BSmY) / 5 (DasC) / 6 (wXn1)
**OpenReview:** https://openreview.net/forum?id=L4uaAR4ArM

### Key Reviewer Criticisms

**Reviewer p1DW (Rating: 5):**
- "Lacks a deeper theoretical analysis of why the proposed MDLM approach outperforms previous methods."
- "Lacks a comprehensive comparison to state-of-the-art non-diffusion language models."
- "Experiments focus on **relatively small models**. It's unclear how well the approach scales to larger models."
- "Primarily focuses on **perplexity as an evaluation metric**. Including human evaluations or other metrics that assess quality and coherence would provide a more comprehensive view."
- "Unclear if MDLM can achieve practical speed-up comparing to AR models with same FLOPs budget."
- Asked: "How does MDLM compare to **MaskGIT**? They share lots of same design space but I didn't see any reference."
- Asked: "What are the authors' thoughts on MDLM versus **continuous-diffusion models** for text, such as **Plaid** and **CDCD**?"

**Reviewer BSmY (Rating: 5):**
- "There also lacks some discussions with some previous works, including **MAGE** [CVPR 2023], which also conducts masked modeling on tokens with a varying mask ratio."

**Reviewer DasC (Rating: 5):**
- "The comparative results are **not fair comparisons**. The use of a more advanced backbone (DiT) and a low-variance sampler weakens the claim that the performance and stability are sourced from the proposed method."
- "Other than the simplified objective and new backbone and sampler, what's the difference between your method and absorbing state D3MM or more specifically **DiffusionBert**?"

**Reviewer wXn1 (Rating: 6):**
- "Several works about using flow-based method are missing: **Language Rectified Flow** and **Flow Matching for Conditional Text Generation**."
- "Better discuss with this work: [arXiv:2406.03736]"

### Papers Cited by Reviewers as Missing
- **MaskGIT** (shares design space with MDLM)
- **MAGE** (MAsked Generative Encoder, CVPR 2023)
- **Plaid** and **CDCD** (continuous diffusion for text)
- **DiffusionBert** (absorbing state discrete diffusion)
- **Language Rectified Flow** (flow-based text generation)
- **Flow Matching for Conditional Text Generation**

### Lessons for Our Work
1. **Scaling experiments are essential** -- reviewers want large model results
2. **Fair comparison requires matching compute** (backbone, sampler, training tokens)
3. **Must compare to MaskGIT, MAGE, and flow-matching approaches**
4. **Human evaluation or downstream metrics** needed beyond perplexity

---

## 6. LLaDA: Large Language Diffusion Models (ACCEPTED)

**Paper:** "Large Language Diffusion Models"
**Venue:** NeurIPS 2025 Oral | **Status:** ACCEPTED
**Scores:** 6 (YafR) / 5 (vGqQ) / 5 (mRMF) / 5 (APN5)
**OpenReview:** https://openreview.net/forum?id=KnqiC0znVF

### Key Reviewer Criticisms

**Reviewer YafR (Rating: 6):**
- "I find the evaluation procedure of LLaDA somewhat unclear, particularly in how its performance is made comparable to that of autoregressive models."
- "When employing the **confidence-based decoding strategy**, how much does the generation behavior diverge from the autoregressive paradigm?"
- "Have you tried to scale a **continuous diffusion model** instead of the MDM?"

**Reviewer vGqQ (Rating: 5):**
- "How is the **output length** controlled during evaluation? ...do you set a fixed length?"
- "Do you have any insights on **why diffusion LM outperforms autoregressive LMs** with the same training FLOPs?"

**Reviewer mRMF (Rating: 5):**
- "Outside of the reversal curse, **what advantages are there to using diffusion models?** Why might they have different capability profiles than their autoregressive counterparts?"
- "Can the authors comment on the novelty of the paper with regards to other diffusion models such as **Inception's Mercury**?"

**Reviewer APN5 (Rating: 5):**
- "The key difference between language and image is in the **variable length**. Although the generation length is easy to decide for most of the benchmarks, it should still be fixed for real deployment. Is there any elegant solution for this?"
- "There is a **misalignment between training and inference** on LLaDA. In training, all tokens are treated equally, while in inference, the prompt is given as a condition and always the clean token."

### Lessons for Our Work
1. **Variable-length generation** is an unsolved problem for masked DLMs
2. **Training-inference misalignment** (all tokens masked in training vs. prompt given clean in inference) is a recognized weakness
3. **Must articulate clear advantages** beyond reversal curse (controllability, editing, etc.)
4. **Continuous diffusion alternative** remains under-explored at scale

---

## 7. BD3-LM: Block Diffusion (ACCEPTED)

**Paper:** "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models"
**Venue:** ICLR 2025 Oral | **Status:** ACCEPTED
**Scores:** 8 (Wz2T) / 8 (HEpL) / 8 (ix6w) / 8 (ZiXH)
**OpenReview:** https://openreview.net/forum?id=tyEyYT267x

### Key Reviewer Criticisms

**Reviewer Wz2T (Rating: 8):**
- "What is the motivation to use diffusion models for language modeling? Maybe potentially even better PPL than auto-regressive models? We don't see that here. Or **better speed?** I assume this is the case, but **this is not shown here**."
- "No speed measurements (for the different cases: Training, sampling, evaluating log probs for given seqs)."

**Reviewer HEpL (Rating: 8):**
- "Since SAD3-LM is first pre-trained with standard autoregression for 850K steps, **is it fair to posit it as a semi-autoregressive discrete diffusion model?** It can be that it learns representations especially on the pre-training part and the fine-tuning part is more accessory."
- Cited missing works: **Accelerating Transformer Inference for Translation via Parallel Decoding** (ACL 2023), consistency-like models for Jacobi iterations
- "What is the point on doing discrete diffusion **if there isn't any gain on standard autoregression?**"

**Reviewer ZiXH (Rating: 8):**
- "SAD3-LM has **significantly increased NFEs** compared to AR, but only with limited PPL improvement."
- "The paper could be improved by involving an experiment on **controlled text generation** as in SSD-LM to further showcase its advantages of integrating diffusion models in AR."

### Lessons for Our Work
1. **Must demonstrate practical speed/quality advantage** over pure AR
2. **Controlled text generation experiments** are expected
3. **Pre-training with AR then fine-tuning as diffusion** raises fairness questions
4. High scores (all 8s) came from clean execution, not from solving controllability

---

## 8. Masked Diffusion Models are Secretly Time-Agnostic (ACCEPTED)

**Paper:** "Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling"
**Venue:** ICLR 2025 | **Status:** ACCEPTED
**Scores:** 6 (nW6z) / 8 (Z3xj) / 8 (KDgp) / 6 (o25N)
**OpenReview:** https://openreview.net/forum?id=CTC7CmirNr

### Key Findings (from paper itself, relevant to our gaps)
- MDMs' training and sampling are time-agnostic -- equivalent to masked models
- **Attention in masked models is incompatible with KV caching** -- a crucial limitation largely overlooked
- 32-bit Gumbel sampling introduces numerical precision issues that artificially improve MDM perplexity
- "MDMs lack a clear and compelling prospect to replace ARMs" given infrastructure/inference limitations

### Key Reviewer Criticisms

**Reviewer Z3xj (Rating: 8):**
- "The paper shows that MDMs do not outperform ARMs in text generation. **It would be beneficial to propose improvements for MDMs.**"
- "The experiments are only conducted on text generation; **more discrete data generation should be considered** (image, music)."

**Reviewer KDgp (Rating: 8):**
- "Is it possible to mitigate the time-agnostic issue by **re-designing or regulating the training objective?**"

### Lessons for Our Work
1. **Time-agnostic nature of MDMs** is now established -- latent conditioning could restore meaningful temporal structure
2. **KV caching incompatibility** is a real deployment barrier
3. **Numerical precision** matters -- must use 64-bit Gumbel sampling

---

## 9. Plug-and-Play Controllable Generation for Discrete Masked Models (MIXED)

**Paper:** "Plug-and-Play Controllable Generation for Discrete Masked Models"
**Venue:** ICLR 2025 | **Status:** Appears rejected or low-scored
**Scores:** 1 (rJsN) / 6 (MPEF) / 5 (kZRj) / 3 (kUek)
**OpenReview:** https://openreview.net/forum?id=4hFT4rfG40

### Key Reviewer Criticisms

**Reviewer rJsN (Rating: 1, harshest):**
- "Several key aspects of the paper lack theoretical justification. The proposed reward equation is provided with **no theoretical grounding**."
- "The mean-field approximation assumes that the probabilities of masked inputs are independent conditioned on the observed values. **This is clearly not the case for domains such as images**, which exhibit strong local structure."
- "The authors mention that the proposed method is beneficial when the complexity of querying the masked model is much higher than evaluating the reward function. Unfortunately, **this is only true for trivial objective functions.** Most problems of interest will not have a closed form/cheap objective function."

**Reviewer MPEF (Rating: 6):**
- "No limitations are presented... The **Monte Carlo samples seem to be quite high** -- performance at 10k samples is still improving."
- "Why are we only looking at protein sequence generation? **Why not also look at natural language applications?**"

**Reviewer kZRj (Rating: 5):**
- "Experimental results only include protein generation benchmarks. **There are no experiments on text, images or audio**."
- "There is **no baseline to compare against** and there are no ablation experiments."

### Lessons for Our Work
1. **Plug-and-play controllable generation for text** remains undertested
2. **Mean-field approximation fails** for structured/correlated domains
3. **High MC sample count** makes these methods impractical at scale
4. **Must include NLP experiments** for controllable generation claims

---

## 10. Soft-Masked Diffusion Language Models (ACCEPTED)

**Paper:** "Soft-Masked Diffusion Language Models"
**Venue:** ICLR 2026 Poster | **Status:** ACCEPTED
**Scores:** 6 (wkhr) / 4 (rBVW) / 8 (GihD) / 6 (E5oT)
**OpenReview:** https://openreview.net/forum?id=Gba02UMvrG

### Key Reviewer Criticisms

**Reviewer wkhr (Rating: 6):**
- "The proposed method requires **two evaluations of the denoiser network** in each training iteration. This makes the comparison to pure masked models unfair given the same batch size."
- "There is a very closely-related prior method **'self-conditioning'** that the authors failed to prominently highlight."

**Reviewer rBVW (Rating: 4):**
- "The training methodology lacks a detailed derivation. Since the authors introduce an embedding for masked tokens, **both the forward process and backward process have been changed** [but no theoretical analysis]."
- "The authors claim that DLMs beat AR in terms of accelerated sampling and controllable generation. **Can the authors provide more evidence to support this claim?**"

**Reviewer E5oT (Rating: 6):**
- "Soft-masking's 'activation' of top-k tokens creates a **biased learning environment**, which could aid constrained inference but hinder generalization."
- "This method, if its low-compute gains stem from **greedy top-k token exploitation**, might be better suited for finetuning or post-training MDLMs, rather than pre-training from scratch as claimed."

### Lessons for Our Work
1. **Soft-masking is related to self-conditioning** -- must cite and differentiate
2. **Two forward passes per training step** is a cost concern -- must address efficiency
3. **Pre-training vs. fine-tuning** distinction matters for claims
4. Soft-masking provides a way to **retain information in masked states** -- complementary to our latent approach

---

## 11. Unveiling the Potential of Diffusion LLMs in Controllable Generation (ACCEPTED)

**Paper:** "Unveiling the Potential of Diffusion Large Language Model in Controllable Generation"
**Venue:** ICLR 2026 Poster | **Status:** ACCEPTED
**Scores:** 6 (3M4f) / 6 (bXwN) / 2 (fQDV) / 6 (CuYm)
**OpenReview:** https://openreview.net/forum?id=qhd0qv6L0k

### Key Finding (from paper)
> "The hallucination issue stems from distributional shifts introduced by the imposed structural constraints, with the rigid schema acting as a structural prior that misaligned with the diffusion language model's learned denoising process. This distributional mismatch forces the model into suboptimal denoising trajectories."

### Key Reviewer Criticisms

**Reviewer 3M4f (Rating: 6):**
- "The authors should provide experimental results from an **autoregressive model baseline**."

**Reviewer bXwN (Rating: 6):**
- "Experiments only use **one dataset (Wikibio)**... unclear how well the proposed method generalizes to other datasets."
- "It would be nice to include experiments on other types of structured outputs, such as **XML or YAML**."

**Reviewer fQDV (Rating: 2):**
- "**Limited experimental scope**: Only one dataset and one diffusion model."
- "**Lack of comparison to AR-LM baselines**: doesn't include comparison with strong AR methods like structured prompting or constrained decoding (CodeLLaMA, T5, GPT-style JSON control)."

**Reviewer CuYm (Rating: 6):**
- "Only one dLLM (LLaDA) is evaluated; cross-model tests (e.g., **Dream** or **Mercury**) would strengthen claims."
- "dLLM inference (O(nL^2)) may still lag AR models (O(L)), which could affect scalability."
- "**Would hybrid diffusion-autoregressive setups (e.g., block diffusion) inherit similar controllability advantages?**"

### Lessons for Our Work
1. **Schema scaffolding causes distributional mismatch** with diffusion denoising -- important failure mode
2. **Must compare against AR baselines** for controllable generation
3. **Must test across multiple dLLMs** (LLaDA, Dream, Mercury)
4. **Block diffusion + controllability** is an open question worth exploring

---

## 12. What Exactly Does Guidance Do in Masked Discrete Diffusion Models (ACCEPTED)

**Paper:** "What Exactly Does Guidance Do in Masked Discrete Diffusion Models"
**Venue:** ICLR 2026 Poster | **Status:** ACCEPTED
**OpenReview:** https://openreview.net/forum?id=h06nffFJqi

### Key Finding
- Guidance amplifies class-specific regions while suppressing shared regions
- Effect depends on guidance strength and induces distinct covariance structures
- Large guidance strength: total variation decay rate is double-exponential

### Relevance
- Provides theoretical grounding for guidance in masked diffusion
- Companion paper "Improving Classifier-Free Guidance in Masked Diffusion" (ICLR 2026) proposes fix: high guidance early in sampling harms quality, late-stage guidance has larger effect

---

## Cross-Paper Synthesis: Recurring Gaps Identified by Reviewers

### Gap 1: Controllable Generation Beyond Perplexity
Nearly every paper was criticized for evaluating only on perplexity. Reviewers consistently asked for:
- Controllable text generation experiments (sentiment, topic, style)
- Structured output generation (JSON, code)
- Conditional generation (seq2seq, QA, machine translation)
- Human evaluation of quality and coherence

### Gap 2: Efficiency and Scalability
Reviewers flagged missing efficiency metrics in almost every paper:
- FLOPs per step, inference time, memory usage
- Fair iso-compute comparisons to AR models
- Scaling behavior with model size and vocabulary size
- KV caching incompatibility for masked models

### Gap 3: Joint Continuous-Discrete Diffusion
CCDD was rejected partly because combining discrete+continuous diffusion "is not new" (DiffuSeq-v2), but reviewers acknowledged the gap remains:
- Better framing: representation efficiency, not expressivity
- Need downstream task evaluation (not just perplexity)
- Need proper ablation isolating continuous vs. discrete contributions

### Gap 4: Text Editing with Diffusion Models
- DICE/Discrete Inversion was withdrawn from ICLR 2025
- No accepted paper demonstrates robust text editing with DLMs
- Schema scaffolding for dLLMs causes distributional mismatch/hallucination
- This remains a wide-open gap

### Gap 5: Guidance at Scale
- Guidance normalization constant prevents scaling to large vocabularies
- High guidance early in sampling harms quality (theory confirmed)
- Dirichlet paths suggested as alternative to uniform paths for large vocabularies
- Mean-field approximation fails for structured domains

### Gap 6: Missing Baselines Frequently Cited by Reviewers
Papers that reviewers repeatedly said should be compared against:
- **MaskGIT** (image discrete generation, similar to MDMs)
- **MAGE** (masked generative encoder)
- **Discrete Flow Matching** (Gat et al. 2024)
- **Dirichlet Flow Matching** (Stark et al. 2024)
- **DiffuSeq-v2** (hybrid continuous-discrete)
- **Mask-Predict** (parallel decoding baseline)
- **Plaid / CDCD** (continuous text diffusion)
- **DiffusionBert** (absorbing state diffusion)
- **Language Rectified Flow** (flow-based text generation)

---

## Actionable Recommendations Based on Reviewer Feedback

1. **Frame contribution as representation efficiency, not expressivity** (RTsX's insight from CCDD reviews)
2. **Include efficiency accounting** in all experiments (FLOPs, wall-clock time, memory)
3. **Evaluate on downstream tasks**: controllable generation, text editing, conditional generation, structured output
4. **Compare to AR baselines** with matched compute budgets
5. **Compare to flow-matching baselines** (Discrete Flow Matching, Dirichlet Flow Matching)
6. **Address variable-length generation** -- a known weakness of masked DLMs
7. **Report entropy alongside perplexity** to detect diversity collapse
8. **Use 64-bit Gumbel sampling** to avoid numerical precision artifacts
9. **Include human evaluation** or LLM-judge metrics alongside automatic metrics
10. **Test across multiple dLLMs** (not just one model) for generalizability

Sources:
- [CCDD OpenReview](https://openreview.net/forum?id=mwAkJJ4NBD)
- [Simple Guidance OpenReview](https://openreview.net/forum?id=i5MrJ6g5G1)
- [EDLM OpenReview](https://openreview.net/forum?id=sL2F9YCMXf)
- [MDLM OpenReview](https://openreview.net/forum?id=L4uaAR4ArM)
- [LLaDA OpenReview](https://openreview.net/forum?id=KnqiC0znVF)
- [BD3-LM OpenReview](https://openreview.net/forum?id=tyEyYT267x)
- [Time-Agnostic MDMs OpenReview](https://openreview.net/forum?id=CTC7CmirNr)
- [Plug-and-Play Controllable OpenReview](https://openreview.net/forum?id=4hFT4rfG40)
- [Soft-Masked Diffusion OpenReview](https://openreview.net/forum?id=Gba02UMvrG)
- [Unveiling Potential OpenReview](https://openreview.net/forum?id=qhd0qv6L0k)
- [Guidance Theory OpenReview](https://openreview.net/forum?id=h06nffFJqi)
- [Improving CFG Masked Diffusion OpenReview](https://openreview.net/forum?id=mMK9pvQJxf)
- [Discrete Inversion OpenReview](https://openreview.net/forum?id=NFEnBqknoX)
