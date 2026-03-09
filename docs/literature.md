# Related Work: Coevolutionary Continuous Discrete Diffusion (CCDD)

## Seed Paper Analysis

**Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner** | ICLR 2026 submission (Rejected) | [arXiv:2510.03206](https://arxiv.org/abs/2510.03206)

**Authors:** Cai Zhou, Chenxiao Yang, Yi Hu, Chenyu Wang, Chubin Zhang, Muhan Zhang, Lester Mackey, Tommi Jaakkola, Stephen Bates, Dinghuai Zhang

**Contributions:**
1. Proves continuous diffusion models have stronger expressivity than discrete diffusions and looped transformers
2. Proposes CCDD: a joint multimodal diffusion process on the union of continuous representation space + discrete token space, using a single model to simultaneously denoise in the joint space
3. Combines rich semantics from continuous latent space with good trainability and sample quality from discrete tokens
4. Develops effective architectures and advanced training/sampling techniques for joint continuous-discrete diffusion
5. Achieves strong empirical performance on LM1B and OpenWebText (significantly lower perplexity vs. baselines)

**Key Methods:** Joint continuous-discrete diffusion process, coevolutionary denoising, continuous latent representation space coupled with discrete token space, single-model dual-space denoising

**Limitations / Future Work:** No official code released. Only evaluated on language modeling (perplexity). No controllable generation experiments. No conditional generation (seq2seq, Q&A). No text editing demonstrations. Architecture details differ from standard MMDiT.

**OpenReview Reviews:** Scores: 4/6/2/4. Decision: **Reject**. Key weaknesses: "underspecified definitions," "limited evaluation scope (only OWT/LM1B, only perplexity)," "lacks evaluation on complex reasoning tasks (Sudoku, coding, GSM8k)," "combining discrete and continuous diffusion not novel (DiffuSeq-v2 exists)," "no comparison to looped transformers," "training/inference cost inadequately explored." Reviewers wanted broader benchmarks, efficiency metrics, and clearer formal definitions.

**Exploration Direction:** Diffusion language models for controllable text generation, conditional generation, and text editing using continuous latent conditioning — how continuous latent channels improve DLM capabilities beyond unconditional generation.

---

## Other Related Papers

Overflow papers that are relevant but did not make either top 10 list.

| Paper | Venue | Year | Link | Code? | Summary |
|-------|-------|------|------|-------|---------|
| SVDD: Soft Value-Based Decoding | arXiv | 2024 | [2408.08252](https://arxiv.org/abs/2408.08252) | [masa-ue/SVDD](https://github.com/masa-ue/SVDD) | Training-free derivative-free guidance via soft value functions for both continuous and discrete diffusion |
| DGLM: Diffusion Guided Language Modeling | ACL Findings | 2024 | [2408.04220](https://arxiv.org/abs/2408.04220) | [justinlovelace/Diffusion-Guided-LM](https://github.com/justinlovelace/Diffusion-Guided-LM) | Diffusion generates continuous semantic proposals to guide AR generation; plug-and-play attribute control |
| DiffuSeq: Seq2Seq with Diffusion | ICLR | 2023 | [2210.08933](https://arxiv.org/abs/2210.08933) | [Shark-NLP/DiffuSeq](https://github.com/Shark-NLP/DiffuSeq) | Continuous diffusion for conditional seq2seq tasks with higher diversity than AR baselines |
| Cosmos: Compressed Latent Space for Text Diffusion | NeurIPS | 2025 | [2506.21170](https://arxiv.org/abs/2506.21170) | [MeshchaninovViacheslav/cosmos](https://github.com/MeshchaninovViacheslav/cosmos) | Autoencoder compresses text into smooth latent space; 2x faster inference matching token-level quality |
| RDLM: Riemannian Diffusion LM | NeurIPS | 2025 | [2502.11564](https://arxiv.org/abs/2502.11564) | [harryjo97/RDLM](https://github.com/harryjo97/RDLM) | Continuous diffusion on statistical manifold geometry; outperforms both continuous and discrete DLMs |
| SMDM: Scaling Masked Diffusion | ICLR | 2025 | [2410.18514](https://arxiv.org/abs/2410.18514) | [ML-GSAI/SMDM](https://github.com/ML-GSAI/SMDM) | First scaling law for masked DLMs + unsupervised CFG for conditional inference on unpaired data |
| NeoDiff: Unifying Continuous and Discrete Text Diffusion | ACL | 2025 | [2505.22165](https://arxiv.org/abs/2505.22165) | :x: | Poisson forward process with per-token adaptive denoising; unifies discrete and continuous diffusion |
| RegDiff: Attribute-Regularized Diffusion | arXiv | 2025 | [2510.06386](https://arxiv.org/abs/2510.06386) | :x: | Train-time attribute regularization avoids need for pretrained classifier at sampling |
| DDOT: Flexible-Length Text Infilling | EMNLP | 2025 | [2506.13579](https://arxiv.org/abs/2506.13579) | :x: | Joint denoising of token values and positions with optimal transport for flexible-length infilling |

---

## Top 10 Most Related Papers (with Code)

Ranked by relevance to the seed paper + exploration direction (DLM controllable/conditional generation and text editing via continuous latent conditioning).

**STRICT RULE: Every paper here MUST have a working GitHub link.**

| Rank | Paper | Venue | Year | Link | GitHub | Why Related | Reviews Summary |
|------|-------|-------|------|------|--------|-------------|-----------------|
| 1 | **Simple Guidance Mechanisms for Discrete Diffusion Models** | ICLR | 2025 | [2412.10193](https://arxiv.org/abs/2412.10193) | [kuleshov-group/discrete-diffusion-guidance](https://github.com/kuleshov-group/discrete-diffusion-guidance) | **Most directly relevant.** Derives CFG + classifier-based guidance for discrete DLMs. Introduces UDLM. Enables controllable generation — the core gap our latent channel addresses differently (continuous guidance vs discrete guidance). | Scores: 3/6/6/6. Weakness: "limited to small vocab, missing runtime analysis." |
| 2 | **Diffusion-LM Improves Controllable Text Generation** | NeurIPS | 2022 | [2205.14217](https://arxiv.org/abs/2205.14217) | [XiangLi1999/Diffusion-LM](https://github.com/XiangLi1999/Diffusion-LM) | **Foundation paper.** Pioneered continuous diffusion for text with gradient-based classifier control for 6 fine-grained tasks. Proves continuous space enables gradient control that discrete tokens cannot. Our latent channel brings this advantage to modern masked DLMs. | |
| 3 | **Latent Diffusion for Language Generation (LD4LG)** | NeurIPS | 2023 | [2212.09462](https://arxiv.org/abs/2212.09462) | [justinlovelace/latent-diffusion-for-language](https://github.com/justinlovelace/latent-diffusion-for-language) | Learns VAE latent (32x256), runs continuous diffusion in latent space. CFG for class-conditional and seq2seq. Shows latent space enables controllability. But uses continuous diffusion for text — our approach keeps discrete masked text. | |
| 4 | **STAR-LDM: Stop-Think-AutoRegress with Latent Diffusion Planning** | COLM | 2025 | [2602.20528](https://arxiv.org/abs/2602.20528) | [justinlovelace/STAR-LDM](https://github.com/justinlovelace/STAR-LDM) | Integrates latent diffusion "thinking" with AR generation. Sentence-T5 (768D) → 8 soft prompt tokens → GPT-2 conditioning. Classifier-guided latent steering. Most recent latent-augmented DLM. | |
| 5 | **LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning** | arXiv | 2025 | [2510.04573](https://arxiv.org/abs/2510.04573) | [mk322/LaDiR](https://github.com/mk322/LaDiR) | VAE compresses reasoning into latent tokens (3x128D); flow-match diffusion refines with CFG + reward + diversity guidance. 30%+ improvement on planning. Shows continuous latent improves DLM reasoning. | |
| 6 | **PLANNER: Latent Language Diffusion for Paragraph Generation** | NeurIPS | 2023 | [2306.02531](https://arxiv.org/abs/2306.02531) | [apple/ml-planner](https://github.com/apple/ml-planner) | Apple. Latent diffusion plans paragraph semantics (16x1024D), AR decoder generates text. Demonstrates latent interpolation producing smoothly varying text. Latent for coarse-to-fine controllable generation. | |
| 7 | **ReMDM: Remasking Discrete Diffusion Models** | arXiv | 2025 | [2503.00307](https://arxiv.org/abs/2503.00307) | [kuleshov-group/remdm](https://github.com/kuleshov-group/remdm) | Iterative refinement via confidence-based remasking. Closest to "editing" in discrete DLMs. But has no continuous signal to steer edits — our latent channel provides richer edit guidance. | |
| 8 | **CANDI: Hybrid Discrete-Continuous Diffusion Models** | arXiv | 2025 | [2510.22510](https://arxiv.org/abs/2510.22510) | [patrickpynadath1/candi-diffusion](https://github.com/patrickpynadath1/candi-diffusion) | Decouples discrete masking from continuous Gaussian noise via token identifiability. Hybrid disc+cont at token level (V-dim one-hot + noise). But not a learned latent — our approach uses compressed semantic latent via MMDiT. | |
| 9 | **EDLM: Energy-Based Diffusion Language Models** | ICLR | 2025 | [2410.21357](https://arxiv.org/abs/2410.21357) | [MinkaiXu/Energy-Diffusion-LLM](https://github.com/MinkaiXu/Energy-Diffusion-LLM) | Energy function captures inter-token dependencies via importance sampling. Energy-based formulation naturally supports conditional generation via energy composition. But discrete-only, no learned latent. | Scores: 5/8/8/8. Weakness: "doubled params, only perplexity eval." |
| 10 | **MDLM: Simple and Effective Masked Diffusion Language Models** | NeurIPS | 2024 | [2406.07524](https://arxiv.org/abs/2406.07524) | [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm) | Foundation masked diffusion model. SUBS parameterization, log-linear schedule. Our text-side builds on MDLM. Purely unconditional — continuous latent conditioning must improve upon this baseline. | Scores: 6/5/5/5. Weakness: "unclear scalability, no human eval." |

---

## Top 10 Most Related Papers (without Code)

Ranked by relevance. These papers have NO public GitHub repo but are important for understanding the field.

| Rank | Paper | Venue | Year | Link | Key Method/Result | Why Related |
|------|-------|-------|------|------|-------------------|-------------|
| 1 | **LDDM: Latent Discrete Diffusion Models** | arXiv | 2025 | [2510.18114](https://arxiv.org/abs/2510.18114) | Couples masked discrete diffusion with continuous latent channel; FUJI variant does fully joint denoising, SEQ does sequential | **Most similar to our approach.** Joint latent+discrete diffusion. But no code, no MMDiT architecture, no editing experiments. |
| 2 | **EdiText: Controllable Text Editing with DLMs** | ACL | 2025 | [2502.19765](https://arxiv.org/abs/2502.19765) | SDEdit-style noising/denoising for text with self-conditioning for fine-grained attribute control | **First SDEdit-for-text paper.** Directly validates our Direction 2 (latent-steered editing). But uses continuous embedding space, not discrete masked + latent. |
| 3 | **Reasoning with Latent Tokens in Diffusion LMs** | arXiv | 2026 | [2602.03769](https://arxiv.org/abs/2602.03769) | Shows masked tokens act as auxiliary computational states; modulates latent token count for speed-quality tradeoff | Demonstrates that latent tokens in DLMs carry computational meaning beyond token prediction. Relevant to understanding our latent channel's role. |
| 4 | **Diffusion-EAGS: Conditional [MASK] Discrete Diffusion LM** | EMNLP | 2025 | [2411.06438](https://arxiv.org/abs/2411.06438) | Conditional masked DLM via conditional MRF theory; entropy-adaptive Gibbs sampling for best quality-diversity tradeoff | Shows how to make masked DLMs conditional via MRF formulation. Alternative to our latent-based conditioning. |
| 5 | **Plug-and-Play Controllable Generation for Discrete Masked Models** | NeurIPS | 2024 | [2410.02143](https://arxiv.org/abs/2410.02143) | Training-free importance-sampling guidance for masked DLMs; agnostic to control criteria, requires no gradients | Training-free guidance alternative. Our latent channel provides gradient-based control; this paper shows non-gradient approaches exist for discrete DLMs. |
| 6 | **ILRR: Inference-Time Steering for Masked Diffusion LMs** | arXiv | 2026 | [2601.21647](https://arxiv.org/abs/2601.21647) | Steers DLM generation by aligning internal activations with reference sequence activations | Latent-space guidance without continuous diffusion. Shows activation steering works for DLMs — our continuous latent could be seen as a learnable version. |
| 7 | **SLD: Segment-Level Diffusion for Long-Form Generation** | ACL | 2025 | [2412.11333](https://arxiv.org/abs/2412.11333) | Segments outputs into latent representations with adversarial/contrastive training and latent-space guidance | Latent-level planning for long-form controllable generation. Related to PLANNER but with better training. No discrete DLM connection. |
| 8 | **Token Assorted: Mixing Latent and Text Tokens** | arXiv | 2025 | [2502.03275](https://arxiv.org/abs/2502.03275) | VQ-VAE compresses initial reasoning into discrete latent tokens mixed with text for shorter reasoning traces | Shows latent tokens can replace explicit reasoning steps. Relevant to our latent channel's potential for compressed reasoning. |
| 9 | **Theory-Informed CFG for Discrete Diffusion** | arXiv | 2025 | [2507.08965](https://arxiv.org/abs/2507.08965) | High early-stage guidance harms quality; late-stage has larger effect; column normalization stabilizes guidance | Theoretical understanding of when guidance helps/hurts in masked DLMs. Directly informs how to apply CFG on our latent channel. |
| 10 | **Debiasing Guidance for Discrete Diffusion with SMC** | ICLR | 2025 | [2502.06079](https://arxiv.org/abs/2502.06079) | SMC algorithm for unbiased guided sampling in discrete diffusion; validated on controlled text and image generation | Principled unbiased guidance for discrete DLMs. Complementary to our latent-based guidance approach. |

---

## Additional Papers with Code (Honorable Mentions)

These papers have public code and are relevant but ranked outside top 10.

| Paper | Venue | Year | GitHub | Why Relevant | Reviews Summary |
|-------|-------|------|--------|--------------|-----------------|
| LLaDA: Large Language Diffusion Models | NeurIPS (Oral) | 2025 | [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) | First 8B masked DLM competitive with LLaMA3 8B. Foundation model that latent conditioning could extend. | ICML rejected (4/3/3/1), NeurIPS accepted oral (5/5/5/6). Weakness: "just scaling, fixed output length." |
| Beyond Autoregression (MGDM) | ICLR | 2025 | [HKUNLP/diffusion-vs-ar](https://github.com/HKUNLP/diffusion-vs-ar) | Multi-granularity diffusion for reasoning; 91.5% Countdown, 100% Sudoku vs 45.8%/20.7% AR. Shows DLM advantage for planning. | Scores: 6/8/5/6. Weakness: "limited task diversity." |
| HDLM: Hierarchical Diffusion LMs | NeurIPS | 2025 | [zhouc20/HDLM](https://github.com/zhouc20/HDLM) | By CCDD authors (Cai Zhou, Dinghuai Zhang, Jaakkola). Hierarchical vocabulary for coarse-to-fine discrete diffusion. | |
| SPG: Sandwiched Policy Gradient | ICLR | 2026 | [facebookresearch/SPG](https://github.com/facebookresearch/SPG) | By CCDD authors (Jaakkola et al.). RL alignment for dLLMs; applicable to latent-augmented models. | |
| BD3-LM: Block Diffusion | ICLR (Oral) | 2025 | [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms) | Semi-AR diffusion over blocks; interpolates AR-diffusion tradeoff. SOTA likelihoods among DLMs. | Scores: 8/8/8/8 (unanimous). |
| DiffuLLaMA: AR-to-DLM Adaptation | ICLR | 2025 | [HKUNLP/DiffuLLaMA](https://github.com/HKUNLP/DiffuLLaMA) | Converts pretrained AR models to DLMs via attention mask annealing. Strong infilling and in-context learning. | Scores: 8/6/6/6. Weakness: "no human eval, no FLOPs." |
| Dream 7B | arXiv | 2025 | [DreamLM/Dream](https://github.com/DreamLM/Dream) | Most powerful open dLLM; 7B params. Key baseline model. | |
| d1: Scaling Reasoning via RL | arXiv | 2025 | [dllm-reasoning/d1](https://github.com/dllm-reasoning/d1) | diffu-GRPO for RL alignment of masked DLMs on reasoning. | |
| SDEdit | ICLR | 2022 | [ermongroup/SDEdit](https://github.com/ermongroup/SDEdit) | Original noising-denoising editing paradigm for images. Concept our "SDEdit for text" extends. | |
| SEDD: Score Entropy Discrete Diffusion | ICML (Best Paper) | 2024 | [louaaron/Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) | Foundation discrete diffusion model; score entropy loss. Many controllable methods build on this. | |
| E2D2: Encoder-Decoder Diffusion LMs | NeurIPS | 2025 | [kuleshov-group/e2d2](https://github.com/kuleshov-group/e2d2) | Encoder-decoder for efficient discrete diffusion; halves training cost. | |
| SSD-LM: Simplex Diffusion LM | ACL | 2023 | [xhan77/ssd-lm](https://github.com/xhan77/ssd-lm) | Diffusion on vocab simplex with modular plug-and-play classifier control. | |
| SLCD: Efficient Controllable Diffusion | arXiv | 2025 | [Owen-Oertell/slcd](https://github.com/Owen-Oertell/slcd) | Iteratively trains small classifier to guide discrete diffusion; provable convergence. | |
| DiffusionBERT | ACL | 2023 | [Hzfinfdu/Diffusion-BERT](https://github.com/Hzfinfdu/Diffusion-BERT) | BERT + discrete diffusion with linguistic noise schedule; improves over D3PM. | |
| Discrete Flow Matching | NeurIPS (Spotlight) | 2024 | [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) | Novel discrete flow paradigm; 1.7B model achieves 6.7% Pass@1 on HumanEval. | |
