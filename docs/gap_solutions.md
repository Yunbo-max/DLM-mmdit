# Research Gap Solutions

Generated from: `docs/research_gap.tex`
Date: 2026-03-03

---

## Gap 1: Joint Discrete-Continuous Architecture (M1, M2, M4, CC1)

**Type:** Methods / Cross-cutting
**Core problem:** No model combines discrete masked text diffusion with a learned continuous latent channel through joint attention (MMDiT). Two disconnected worlds exist: 12 discrete masked DLM papers vs 8 continuous latent DLM papers.
**Sub-problems:** (a) How to do shared QKV attention between discrete tokens and continuous vectors, (b) How to handle different noise processes in one forward pass, (c) How to design the latent space.

### Direction 1.1: MMDiT / DiT for Multimodal Generation

**Reasoning:** SD3 and similar models already handle joint attention between image and text modalities. The MMDiT block (separate per-modality weights, concatenated QKV) is the core primitive for text+latent joint denoising.
**Search domains:** Image generation, multimodal generation, vision-language models

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **SD3: Scaling Rectified Flow Transformers** (MMDiT) | ICML | 2024 | [2403.03206](https://arxiv.org/abs/2403.03206) | [Stability-AI/sd3.5](https://github.com/Stability-AI/sd3.5), [lucidrains/mmdit](https://github.com/lucidrains/mmdit) | Defines the MMDiT block: separate per-modality weights with joint attention via concatenated QKV -- the core architectural primitive |
| 2 | **Transfusion: Predict Next Token and Diffuse Images** | ICLR | 2025 | [2408.11039](https://arxiv.org/abs/2408.11039) | [lucidrains/transfusion-pytorch](https://github.com/lucidrains/transfusion-pytorch) | Trains single transformer with next-token loss on text + diffusion loss on images simultaneously; proves mixed discrete+continuous objectives can share one backbone |
| 3 | **Show-o: One Transformer for Understanding + Generation** | NeurIPS | 2025 | [2408.12528](https://arxiv.org/abs/2408.12528) | [showlab/Show-o](https://github.com/showlab/Show-o) | Discrete diffusion for image tokens + AR for text in one transformer; demonstrates mixed attention masking (causal for text, full for diffusion) |
| 4 | **OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows** | CVPR | 2025 | [2412.01169](https://arxiv.org/abs/2412.01169) | [jacklishufan/OmniFlows](https://github.com/jacklishufan/OmniFlows) | Extends MMDiT to audio and text with modality-specific modules; shows how to add new modality branches to existing MMDiT |
| 5 | **Dual Diffusion (D-DiT)** | CVPR | 2025 | [2501.00289](https://arxiv.org/abs/2501.00289) | [zijieli-Jlee/Dual-Diffusion](https://github.com/zijieli-Jlee/Dual-Diffusion) | Two diffusion branches (image + text) inside single DiT with cross-modal maximum likelihood |

### Direction 1.2: Coupled Diffusion Processes (Masking + Gaussian)

**Reasoning:** Coupling two different noise processes (discrete masking + continuous Gaussian) requires mathematical frameworks. Papers that solve this for tabular/molecular data can inspire text+latent coupling.
**Search domains:** Mixed-type generation, tabular data, molecular design, population genetics

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **Duo: The Diffusion Duality** | ICML + ICLR | 2025/2026 | [2506.10892](https://arxiv.org/abs/2506.10892) | [s-sahoo/duo](https://github.com/s-sahoo/duo) | Proves discrete diffusion naturally emerges from underlying Gaussian diffusion; provides the mathematical bridge to transfer continuous techniques to discrete |
| 2 | **TabDiff: Mixed-type Diffusion for Tabular Data** | ICLR | 2025 | [2410.20626](https://arxiv.org/abs/2410.20626) | [MinkaiXu/TabDiff](https://github.com/MinkaiXu/TabDiff) | Joint continuous-time process for numerical (Gaussian) + categorical (discrete) features with feature-wise learnable noise schedules |
| 3 | **CANDI: Hybrid Discrete-Continuous Diffusion** | arXiv | 2025 | [2510.22510](https://arxiv.org/abs/2510.22510) | [patrickpynadath1/candi-diffusion](https://github.com/patrickpynadath1/candi-diffusion) | Decouples discrete masking from Gaussian noise into independently controlled corruption mechanisms |
| 4 | **Unification of Discrete, Gaussian, Simplicial Diffusion** | arXiv | 2025 | [2512.15923](https://arxiv.org/abs/2512.15923) | [yucenli/unify-diffusion](https://github.com/yucenli/unify-diffusion) | Unifies all three diffusion types as parameterizations of Wright-Fisher model; single framework for mixed processes |
| 5 | **Discrete Copula Diffusion** | ICLR | 2025 | [2410.01949](https://arxiv.org/abs/2410.01949) | [liuanji/Copula-Diffusion](https://github.com/liuanji/Copula-Diffusion) | Copula model captures inter-variable dependencies during denoising; 8-32x fewer steps |

### Direction 1.3: Hybrid Discrete-Continuous Generative Models

**Reasoning:** Models that jointly generate discrete and continuous variables from other domains (images, molecules, floorplans) provide architectural templates.
**Search domains:** Multimodal generation, molecular design, layout generation

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **DisCo-Diff: Continuous Diffusion + Discrete Latents** | ICML | 2024 | [2407.03300](https://arxiv.org/abs/2407.03300) | [gcorso/disco-diffdock](https://github.com/gcorso/disco-diffdock) | Augments continuous diffusion with learnable discrete codes; discrete+continuous co-training improves both quality and efficiency |
| 2 | **UniDiffuser: One Transformer Fits All Distributions** | ICML | 2023 | [2303.06555](https://arxiv.org/abs/2303.06555) | [thu-ml/unidiffuser](https://github.com/thu-ml/unidiffuser) | Per-modality timesteps for joint/marginal/conditional distributions; foundational for variable-rate multimodal denoising |
| 3 | **UniDisc: Unified Multimodal Discrete Diffusion** | arXiv | 2025 | [2503.20853](https://arxiv.org/abs/2503.20853) | [alexanderswerdlow/unidisc](https://github.com/alexanderswerdlow/unidisc) | Joint text+image discrete diffusion; outperforms AR on controllability and editability |
| 4 | **MMaDA: Multimodal Large Diffusion Language Models** | NeurIPS | 2025 | [2505.15809](https://arxiv.org/abs/2505.15809) | [Gen-Verse/MMaDA](https://github.com/Gen-Verse/MMaDA) | 8B unified diffusion with modality-agnostic design and unified RL post-training (UniGRPO) |
| 5 | **BD3-LM: Block Diffusion** | ICLR (Oral) | 2025 | [2503.09573](https://arxiv.org/abs/2503.09573) | [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms) | Block decomposition with per-block diffusion + AR across blocks; applicable to mixed text+latent sequences |

---

## Gap 2: Joint End-to-End Training (M5)

**Type:** Methods
**Core problem:** Two-stage training (autoencoder then diffusion) dominates; joint training of latent encoder + discrete text diffusion is undemonstrated.
**Sub-problems:** (a) Balancing discrete CE + continuous MSE losses, (b) Gradient flow between discrete and continuous, (c) Training stability.

### Direction 2.1: Joint VAE + Diffusion Training

**Reasoning:** End-to-end training of autoencoder + diffusion avoids representation mismatch. Recent work in image generation has cracked this.
**Search domains:** Image generation, representation learning

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **REPA-E: End-to-End Tuning of Latent Diffusion** | ICCV | 2025 | [2504.10483](https://arxiv.org/abs/2504.10483) | [End2End-Diffusion/REPA-E](https://github.com/End2End-Diffusion/REPA-E) | First stable joint VAE+diffusion training via representation-alignment loss; 17-45x training speedup |
| 2 | **REPA: Representation Alignment for Generation** | ICLR (Oral) | 2025 | [2410.06940](https://arxiv.org/abs/2410.06940) | [sihyun-yu/REPA](https://github.com/sihyun-yu/REPA) | Aligns noisy hidden states with pretrained encoder representations; foundation for REPA-E |
| 3 | **Improving Diffusability of Autoencoders** | ICML | 2025 | [2502.14831](https://arxiv.org/abs/2502.14831) | [snap-research/diffusability](https://github.com/snap-research/diffusability) | Scale-equivariance regularization aligns VAE latent and target frequency spaces; reduces FID 19% |
| 4 | **LD4LG: Latent Diffusion for Language** | NeurIPS | 2023 | [2212.09462](https://arxiv.org/abs/2212.09462) | [justinlovelace/latent-diffusion-for-language](https://github.com/justinlovelace/latent-diffusion-for-language) | End-to-end language autoencoder + continuous latent diffusion for text |
| 5 | **UniDisc: Unified Multimodal Discrete Diffusion** | arXiv | 2025 | [2503.20853](https://arxiv.org/abs/2503.20853) | [alexanderswerdlow/unidisc](https://github.com/alexanderswerdlow/unidisc) | Joint text-image discrete diffusion trained end-to-end |

### Direction 2.2: Multi-Task Loss Balancing

**Reasoning:** Balancing discrete cross-entropy and continuous MSE losses requires multi-task optimization. Existing libraries provide plug-and-play solutions.
**Search domains:** Multi-task learning, optimization

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **LibMTL: Python Library for Multi-Task Learning** | JMLR | 2024 | [2203.14338](https://arxiv.org/abs/2203.14338) | [median-research-group/LibMTL](https://github.com/median-research-group/LibMTL) | Unified PyTorch library with 12+ loss-weighting methods (GradNorm, UW, PCGrad, FAMO, NashMTL) |
| 2 | **LibMOON: Gradient-based Multi-Objective Optimization** | NeurIPS | 2024 | [2409.02969](https://arxiv.org/abs/2409.02969) | [xzhang2523/libmoon](https://github.com/xzhang2523/libmoon) | 20+ gradient-based MOO algorithms with GPU acceleration; Pareto-optimal solutions |
| 3 | **FAMO: Fast Adaptive Multitask Optimization** | NeurIPS | 2023 | [2306.03792](https://arxiv.org/abs/2306.03792) | [Cranial-XIX/FAMO](https://github.com/Cranial-XIX/FAMO) | O(1) space/time dynamic weighting; scalable to large models |
| 4 | **NashMTL: Multi-Task Learning as Bargaining** | ICML | 2022 | [2202.01017](https://arxiv.org/abs/2202.01017) | [AvivNavon/nash-mtl](https://github.com/AvivNavon/nash-mtl) | Game-theoretic Nash Bargaining for multi-task gradient combination with convergence guarantees |
| 5 | **PCGrad: Gradient Surgery** | NeurIPS | 2020 | [2001.06782](https://arxiv.org/abs/2001.06782) | [tianheyu927/PCGrad](https://github.com/tianheyu927/PCGrad) | Projects conflicting task gradients; prevents destructive interference between CE and MSE |

### Direction 2.3: Differentiable Discrete-Continuous Bridges

**Reasoning:** Gradient flow between discrete tokens and continuous latent requires differentiable relaxations. Gumbel-Softmax and score-based bridges provide this.
**Search domains:** Discrete optimization, NLP, molecular design

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **DRAKES: Reward Optimization for Discrete Diffusion** | ICLR | 2025 | [2410.13643](https://arxiv.org/abs/2410.13643) | [ChenyuWang-Monica/DRAKES](https://github.com/ChenyuWang-Monica/DRAKES) | Gumbel-Softmax makes discrete diffusion trajectories differentiable for reward backpropagation |
| 2 | **SEDD: Score Entropy Discrete Diffusion** | ICML (Best Paper) | 2024 | [2310.16834](https://arxiv.org/abs/2310.16834) | [louaaron/Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) | Concrete score ratios bridge continuous score matching to discrete tokens |
| 3 | **GST: Gapped Straight-Through Estimator** | ICML | 2022 | [2206.07235](https://arxiv.org/abs/2206.07235) | [chijames/GST](https://github.com/chijames/GST) | Reduces gradient variance of STE Gumbel-Softmax without resampling overhead |
| 4 | **STGFlow: Gumbel-Softmax Flow Matching** | arXiv | 2025 | [2503.17361](https://arxiv.org/abs/2503.17361) | [ChatterjeeLab/GumbelFlow](https://huggingface.co/ChatterjeeLab/GumbelFlow) | Gumbel-Softmax interpolant with time-dependent temperature for discrete flow matching |
| 5 | **BD3-LM: Block Diffusion** | ICLR (Oral) | 2025 | [2503.09573](https://arxiv.org/abs/2503.09573) | [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms) | Bridges AR and diffusion via block decomposition; tunable quality-efficiency tradeoff |

---

## Gap 3: Guidance Unification (M6, C2, C5, C6, CC3)

**Type:** Methods / Cross-cutting
**Core problem:** 11+ guidance methods exist in silos (CFG, CBG, energy IS, remasking, MRF, importance sampling, activation alignment, SMC, classifier gradients, reward, latent guidance). No architecture supports both continuous-latent and discrete-token guidance.

### Direction 3.1: CFG in Latent Space for Text

**Reasoning:** Classifier-free guidance is well-understood in continuous latent diffusion (images). Adapting it for text latent space is the foundation for guidance unification.
**Search domains:** Image generation, text generation

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **Simple Guidance for Discrete Diffusion** | ICLR | 2025 | [2412.10193](https://arxiv.org/abs/2412.10193) | [kuleshov-group/discrete-diffusion-guidance](https://github.com/kuleshov-group/discrete-diffusion-guidance) | Clean derivation of CFG + classifier guidance for discrete diffusion; baseline to compare latent-space guidance against |
| 2 | **LD4LG: Latent Diffusion for Language** | NeurIPS | 2023 | [2212.09462](https://arxiv.org/abs/2212.09462) | [justinlovelace/latent-diffusion-for-language](https://github.com/justinlovelace/latent-diffusion-for-language) | Demonstrates CFG in continuous latent text space for class-conditional and seq2seq generation |
| 3 | **beta-CFG: Adaptive Scaling** | arXiv | 2025 | [2502.10574](https://arxiv.org/abs/2502.10574) | [gmum/beta-CFG](https://github.com/gmum/beta-CFG) | Dynamic guidance strength via beta-distribution time curves; applicable to latent text diffusion |
| 4 | **ReMDM: Remasking with Inference-Time Scaling** | arXiv | 2025 | [2503.00307](https://arxiv.org/abs/2503.00307) | [kuleshov-group/remdm](https://github.com/kuleshov-group/remdm) | Inference-time compute scaling for masked diffusion via remasking; orthogonal to latent guidance |
| 5 | **MDLM: Masked Diffusion Language Model** | NeurIPS | 2024 | [2406.07524](https://arxiv.org/abs/2406.07524) | [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm) | SOTA discrete diffusion LM with conditional dropout enabling CFG; the discrete-side baseline |

### Direction 3.2: Reward-Guided Diffusion Sampling

**Reasoning:** RL reward guidance enables arbitrary objective optimization during diffusion sampling. Combining reward guidance on continuous latent with discrete text diffusion is unexplored.
**Search domains:** RL alignment, image generation, protein design

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **DRAKES: Discrete Diffusion Reward Optimization** | ICLR | 2025 | [2410.13643](https://arxiv.org/abs/2410.13643) | [ChenyuWang-Monica/DRAKES](https://github.com/ChenyuWang-Monica/DRAKES) | Reward optimization for discrete diffusion via Gumbel-Softmax; applicable to text/protein/DNA |
| 2 | **DDPO: Training Diffusion Models with RL** | ICLR | 2024 | [2305.13301](https://arxiv.org/abs/2305.13301) | [jannerm/ddpo](https://github.com/jannerm/ddpo) | Foundational RL reward optimization for diffusion: denoising as multi-step MDP with policy gradients |
| 3 | **Diffusion-DPO: Direct Preference Optimization** | CVPR | 2024 | [2311.12908](https://arxiv.org/abs/2311.12908) | [SalesforceAIResearch/DiffusionDPO](https://github.com/SalesforceAIResearch/DiffusionDPO) | DPO adapted to diffusion likelihood for preference-based alignment without reward models |
| 4 | **D3PO: Human Feedback for Diffusion** | CVPR | 2024 | [2311.17911](https://arxiv.org/abs/2311.17911) | [yk7333/d3po](https://github.com/yk7333/d3po) | Direct human feedback fine-tuning without separate reward model |
| 5 | **DDPO-PyTorch (LoRA variant)** | Community | 2024 | [2305.13301](https://arxiv.org/abs/2305.13301) | [kvablack/ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch) | Lightweight PyTorch DDPO with LoRA; practical reward-guided diffusion on <10GB GPU |

### Direction 3.3: Universal Guidance Frameworks

**Reasoning:** Papers that unify multiple guidance types (CFG + classifier + reward) in one framework provide the template for guidance unification through a latent channel.
**Search domains:** Image generation, composable control

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **Universal Guidance for Diffusion Models** | ICLR | 2024 | [2302.07121](https://arxiv.org/abs/2302.07121) | [arpitbansal297/Universal-Guided-Diffusion](https://github.com/arpitbansal297/Universal-Guided-Diffusion) | Enables arbitrary guidance modalities (segmentation, face, style, CLIP, classifier) in one unified algorithm without retraining |
| 2 | **UniCoDe: Unified Control for Inference-Time Guidance** | arXiv | 2025 | [2512.12339](https://arxiv.org/abs/2512.12339) | [maurya-goyal10/UniCoDe](https://github.com/maurya-goyal10/UniCoDe) | Unifies sampling-based and gradient-based guidance; integrates local gradients during blockwise sampling |
| 3 | **UniDisc: Unified Multimodal Discrete Diffusion** | arXiv | 2025 | [2503.20853](https://arxiv.org/abs/2503.20853) | [alexanderswerdlow/unidisc](https://github.com/alexanderswerdlow/unidisc) | Single discrete diffusion handling text+images with guidance, inpainting, controllable generation across modalities |
| 4 | **Awesome Alignment of Diffusion Models (Survey)** | ACM | 2025 | Survey | [xie-lab-ml/awesome-alignment-of-diffusion-models](https://github.com/xie-lab-ml/awesome-alignment-of-diffusion-models) | Comprehensive curated collection of all diffusion alignment methods with code links |
| 5 | **Simple Guidance for Discrete Diffusion** | ICLR | 2025 | [2412.10193](https://arxiv.org/abs/2412.10193) | [kuleshov-group/discrete-diffusion-guidance](https://github.com/kuleshov-group/discrete-diffusion-guidance) | Unified derivation of CFG + classifier guidance for discrete diffusion under one framework |

---

## Gap 4: Latent-Steered Text Editing (C4, CC2)

**Type:** Capabilities / Cross-cutting
**Core problem:** No discrete DLM supports SDEdit-style editing with continuous latent steering. ReMDM edits via remasking (no continuous signal); EdiText uses continuous embeddings (no discrete masking).

### Direction 4.1: SDEdit Adaptations for Text

**Reasoning:** The noise-then-denoise editing paradigm (SDEdit) has been adapted for text in several ways. These provide templates for latent-steered editing.
**Search domains:** Text generation, style transfer, machine translation

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **DiffusER: Diffusion via Edit-Based Reconstruction** | ICLR | 2023 | [2209.14453](https://arxiv.org/abs/2209.14453) | [machelreid/diffuser](https://github.com/machelreid/diffuser) | Edit-based diffusion process (INSERT/DELETE/KEEP/REPLACE) for iterative text revision; SDEdit-like conditioning on incomplete sequences |
| 2 | **SSD-LM: Simplex-based Diffusion LM** | ACL | 2023 | [2210.17432](https://arxiv.org/abs/2210.17432) | [xhan77/ssd-lm](https://github.com/xhan77/ssd-lm) | Diffusion on token-probability simplex with classifier guidance; direct text analogue of SDEdit |
| 3 | **Diffusion-LM: Controllable Text Generation** | NeurIPS | 2022 | [2205.14217](https://arxiv.org/abs/2205.14217) | [XiangLi1999/Diffusion-LM](https://github.com/XiangLi1999/Diffusion-LM) | Foundation continuous text diffusion with gradient-based editing; established the noise-then-denoise paradigm for text |
| 4 | **DICE: Discrete Inversion for Controllable Editing** | CVPR | 2025 | [2410.08207](https://arxiv.org/abs/2410.08207) | [hexiaoxiao-cs/DICE](https://hexiaoxiao-cs.github.io/DICE/) | First precise inversion for discrete diffusion/masked models; records noise sequences for accurate reconstruction + editing |
| 5 | **DiffuLLaMA: AR-to-DLM Adaptation** | ICLR | 2025 | [2410.17891](https://arxiv.org/abs/2410.17891) | [HKUNLP/DiffuLLaMA](https://github.com/HKUNLP/DiffuLLaMA) | Converts AR LLMs to diffusion models with fill-in-the-middle capability; large-scale backbone for SDEdit-style editing |

### Direction 4.2: Masked Editing with Conditioning

**Reasoning:** Text infilling conditioned on a signal (class, attribute, latent) is the core mechanism for latent-steered editing.
**Search domains:** Masked language models, code generation, text infilling

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **SMDM: Scaling Masked Diffusion Models** | ICLR | 2025 | [2410.18514](https://arxiv.org/abs/2410.18514) | [ML-GSAI/SMDM](https://github.com/ML-GSAI/SMDM) | 1.1B masked DLM with unsupervised CFG for conditional infilling |
| 2 | **DreamOn: Diffusion LMs for Code Infilling** | arXiv | 2026 | [2602.01326](https://arxiv.org/abs/2602.01326) | [DreamLM/DreamOn](https://github.com/DreamLM/DreamOn) | Expand/delete control states for variable-length conditioned infilling |
| 3 | **DINOISER: Conditional Sequence Learning** | TACL | 2024 | [2302.10025](https://arxiv.org/abs/2302.10025) | [yegcjs/DINOISER](https://github.com/yegcjs/DINOISER) | Amplifies noise scales to leverage source conditions; improves conditioned seq2seq in diffusion |
| 4 | **ADLM: Anchored Diffusion Language Model** | NeurIPS | 2025 | [2505.18456](https://arxiv.org/abs/2505.18456) | [LituRout/ADLM](https://github.com/LituRout/ADLM) | Predicts anchor tokens first, then infills conditioned on anchors; 25% PPL improvement over SEDD |
| 5 | **MDLM: Masked Diffusion Language Model** | NeurIPS | 2024 | [2406.07524](https://arxiv.org/abs/2406.07524) | [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm) | SOTA masked diffusion with infilling via classifier-free guidance |

### Direction 4.3: Style Transfer via Latent Manipulation

**Reasoning:** Changing text attributes by manipulating latent representations demonstrates the mechanism our latent channel would use for steered editing.
**Search domains:** Text style transfer, controllable generation, VAE

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **LatentOps: Composable Text Controls with ODEs** | ACL | 2023 | [2208.00638](https://arxiv.org/abs/2208.00638) | [guangyliu/LatentOps](https://github.com/guangyliu/LatentOps) | ODE-based style transfer (sentiment, tense, formality) in VAE latent space; demonstrates latent-space attribute manipulation |
| 2 | **MAGIC: Multi-Aspect Control via Disentangled Augmentation** | ACL | 2024 | [2405.19958](https://arxiv.org/abs/2405.19958) | [nju-websoft/MAGIC](https://github.com/nju-websoft/MAGIC) | Disentangled attribute latent space with counterfactual augmentation for multi-aspect control |
| 3 | **PLANNER: Latent Language Diffusion** | NeurIPS | 2023 | [2306.02531](https://arxiv.org/abs/2306.02531) | [apple/ml-planner](https://github.com/apple/ml-planner) | Latent interpolation steers global text structure and style |
| 4 | **LD4LG: Latent Diffusion for Language** | NeurIPS | 2023 | [2212.09462](https://arxiv.org/abs/2212.09462) | [justinlovelace/latent-diffusion-for-language](https://github.com/justinlovelace/latent-diffusion-for-language) | Latent vectors can be interpolated/manipulated for attribute control |
| 5 | **SLD: Segment-Level Diffusion** | ACL | 2025 | [2412.11333](https://arxiv.org/abs/2412.11333) | [SpaceHunterInf/Segment_Level_Diffusion](https://github.com/SpaceHunterInf/Segment_Level_Diffusion) | Segment-level latent with adversarial/contrastive training; enables style/coherence control |

---

## Gap 5: Conditional/Seq2Seq Generation (C1, TD1, TD4)

**Type:** Capabilities / Tasks
**Core problem:** No discrete masked DLM supports encoder-decoder conditioning. Only continuous diffusion models (LD4LG, DiffuSeq) support seq2seq.

### Direction 5.1: Encoder-Decoder Diffusion for Text

**Reasoning:** Seq2seq via diffusion is well-explored in continuous space. Adapting these encoder-conditioned architectures for a discrete DLM with latent conditioning is the target.
**Search domains:** Machine translation, summarization, question generation

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **DiffuSeq: Seq2Seq with Diffusion** | ICLR | 2023 | [2210.08933](https://arxiv.org/abs/2210.08933) | [Shark-NLP/DiffuSeq](https://github.com/Shark-NLP/DiffuSeq) | Foundational conditional diffusion for seq2seq (QG, paraphrase, dialogue) via partial noising |
| 2 | **DiffuSeq-v2: Bridging Discrete and Continuous** | arXiv | 2023 | [2310.05793](https://arxiv.org/abs/2310.05793) | [Shark-NLP/DiffuSeq](https://github.com/Shark-NLP/DiffuSeq/tree/diffuseq-v2) | Soft absorbing state + ODE solvers; 4x faster training, 800x faster sampling |
| 3 | **SeqDiffuSeq: Encoder-Decoder Transformers** | NAACL | 2024 | [2212.10325](https://arxiv.org/abs/2212.10325) | [Yuanhy1997/SeqDiffuSeq](https://github.com/Yuanhy1997/SeqDiffuSeq) | Encoder-decoder with self-conditioning and adaptive noise; outperforms DiffuSeq |
| 4 | **Meta-DiffuB: Contextualized Seq2Seq Diffusion** | NeurIPS | 2024 | [2410.13201](https://arxiv.org/abs/2410.13201) | [Meta-DiffuB/Meta-DiffuB](https://github.com/Meta-DiffuB/Meta-DiffuB) | Meta-scheduler for contextualized noise per sentence; SOTA on 4 seq2seq benchmarks |
| 5 | **GENIE: Continuous Paragraph Denoise** | ICML | 2023 | [2212.11685](https://arxiv.org/abs/2212.11685) | [microsoft/ProphetNet/GENIE](https://github.com/microsoft/ProphetNet/tree/master/GENIE) | Large-scale encoder-decoder diffusion pre-trained for summarization, QA, completion |

### Direction 5.2: Soft Prompt / Latent Conditioning

**Reasoning:** Compressed latent representations as conditioning signals for text generation are the mechanism our MMDiT model uses.
**Search domains:** Prompt compression, latent conditioning, planning

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **PLANNER: Latent Language Diffusion** | NeurIPS | 2023 | [2306.02531](https://arxiv.org/abs/2306.02531) | [apple/ml-planner](https://github.com/apple/ml-planner) | Latent semantic diffusion generates paragraph-level "plan" conditioning AR decoder |
| 2 | **Plaid 1B: Likelihood-Based Diffusion LM** | arXiv | 2023 | [2305.18619](https://arxiv.org/abs/2305.18619) | [igul222/plaid](https://github.com/igul222/plaid) | 1B continuous diffusion LM; zero-shot controllable generation via continuous latent |
| 3 | **LD4LG: Latent Diffusion for Language** | NeurIPS | 2023 | [2212.09462](https://arxiv.org/abs/2212.09462) | [justinlovelace/latent-diffusion-for-language](https://github.com/justinlovelace/latent-diffusion-for-language) | Diffuses in compressed latent; supports class-conditional and seq2seq with 250 steps |
| 4 | **Gist Tokens: Learning to Compress Prompts** | NeurIPS | 2023 | [2304.08467](https://arxiv.org/abs/2304.08467) | [jayelm/gisting](https://github.com/jayelm/gisting) | Compresses prompts into compact "gist" tokens (26x compression); applicable to diffusion decoders |
| 5 | **SLD: Segment-Level Diffusion** | ACL | 2025 | [2412.11333](https://arxiv.org/abs/2412.11333) | [SpaceHunterInf/Segment_Level_Diffusion](https://github.com/SpaceHunterInf/Segment_Level_Diffusion) | Compresses text segments into latent vectors; diffuses over compressed representations |

### Direction 5.3: DLMs for Reasoning Tasks

**Reasoning:** Discrete DLMs are increasingly applied to reasoning (math, code, planning). Adding latent conditioning could further improve reasoning capabilities.
**Search domains:** Math reasoning, code generation, planning

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **d1: Scaling Reasoning via RL in Diffusion LLMs** | arXiv | 2025 | [2504.12216](https://arxiv.org/abs/2504.12216) | [dllm-reasoning/d1](https://github.com/dllm-reasoning/d1) | diffu-GRPO for masked DLMs; +26.2% Countdown, +10% Sudoku over LLaDA-8B |
| 2 | **DiffuCoder: Masked Diffusion for Code** | arXiv | 2025 | [2506.20639](https://arxiv.org/abs/2506.20639) | [apple/ml-diffucoder](https://github.com/apple/ml-diffucoder) | 7B code dLLM; coupled-GRPO RL; +4.4% EvalPlus; analyzes generation order |
| 3 | **Diffusion of Thoughts: CoT in Diffusion LMs** | NeurIPS | 2024 | [2402.07754](https://arxiv.org/abs/2402.07754) | [HKUNLP/diffusion-of-thoughts](https://github.com/HKUNLP/diffusion-of-thoughts) | CoT reasoning steps diffuse over time; small DLM outperforms larger AR on math/logic |
| 4 | **Dream 7B: Diffusion Large Language Model** | arXiv | 2025 | [2508.15487](https://arxiv.org/abs/2508.15487) | [DreamLM/Dream](https://github.com/DreamLM/Dream) | Most powerful open dLLM; outperforms all prior on math/code/general |
| 5 | **MDPO: Overcoming Training-Inference Divide** | arXiv | 2025 | [2508.13148](https://arxiv.org/abs/2508.13148) | [autonomousvision/mdpo](https://github.com/autonomousvision/mdpo) | RL for MDMs; +9.6% MATH500, +54.2% Countdown with 60x fewer gradient updates |

---

## Gap 6: Latent Space Geometry (C3, TD5)

**Type:** Tasks / Evaluation
**Core problem:** Does the learned latent space have meaningful geometric structure (interpolation, arithmetic)? Only PLANNER demonstrates this, using continuous text diffusion.

### Direction 6.1: Latent Interpolation for Text

**Reasoning:** Smooth latent spaces that produce meaningful text variations when interpolated demonstrate the latent channel has geometric structure.
**Search domains:** VAE, text generation, sentence embeddings

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **Cosmos: Smooth Latent Space for Text Diffusion** | NeurIPS | 2025 | [2506.21170](https://arxiv.org/abs/2506.21170) | [MeshchaninovViacheslav/cosmos](https://github.com/MeshchaninovViacheslav/cosmos) | 8x-compressed smooth latent for text diffusion; perturbation augmentations promote smooth interpolation |
| 2 | **Smoothie: Smoothing Diffusion on Token Embeddings** | arXiv | 2025 | [2505.18853](https://arxiv.org/abs/2505.18853) | [ashaba1in/smoothie](https://github.com/ashaba1in/smoothie) | Combines Gaussian and categorical diffusion via semantic similarity; creates continuous space respecting token semantics |
| 3 | **LD4LG: Latent Diffusion for Language** | NeurIPS | 2023 | [2212.09462](https://arxiv.org/abs/2212.09462) | [justinlovelace/latent-diffusion-for-language](https://github.com/justinlovelace/latent-diffusion-for-language) | Compact latent space enabling smooth interpolation |
| 4 | **Optimus: Pretrained VAE Latent Space** | EMNLP | 2020 | [2004.04092](https://arxiv.org/abs/2004.04092) | [ChunyuanLI/Optimus](https://github.com/ChunyuanLI/Optimus) | First large-scale pretrained text VAE (BERT+GPT-2); universal sentence latent supporting interpolation and analogy |
| 5 | **AdaVAE: Adaptive GPT-2 VAEs** | arXiv | 2022 | [2205.05862](https://arxiv.org/abs/2205.05862) | [ImKeTT/adavae](https://github.com/ImKeTT/adavae) | Latent Attention mechanism; demonstrates linear and arithmetic interpolation in text latent space |

### Direction 6.2: Disentangled Representations for Text

**Reasoning:** Disentangled latent variables that control different attributes separately (sentiment, topic, style) demonstrate meaningful latent geometry.
**Search domains:** Style transfer, controllable generation, representation learning

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **MAGIC: Multi-Aspect Controllable with Disentanglement** | ACL | 2024 | [2405.19958](https://arxiv.org/abs/2405.19958) | [nju-websoft/MAGIC](https://github.com/nju-websoft/MAGIC) | Disentangles attribute correlations via counterfactual feature vectors for multi-aspect control |
| 2 | **LatentOps: Composable Controls with ODEs** | ACL | 2023 | [2208.00638](https://arxiv.org/abs/2208.00638) | [guangyliu/LatentOps](https://github.com/guangyliu/LatentOps) | ODE-based diffusion in VAE latent for composable sentiment/tense/formality control |
| 3 | **MacLaSa: Multi-Aspect Control from Compact Latent** | EMNLP | 2023 | [2305.12785](https://arxiv.org/abs/2305.12785) | [TrustedLLM/MacLaSa](https://github.com/TrustedLLM/MacLaSa) | Compact latent + ODE sampling with pluggable attribute discriminators |
| 4 | **Disentangled Representation for Style Transfer** | ACL | 2019 | [1808.04339](https://arxiv.org/abs/1808.04339) | [h3lio5/linguistic-style-transfer-pytorch](https://github.com/h3lio5/linguistic-style-transfer-pytorch) | Multi-task + adversarial disentanglement of style and content |
| 5 | **Diffusion-LM: Controllable Text Generation** | NeurIPS | 2022 | [2205.14217](https://arxiv.org/abs/2205.14217) | [XiangLi1999/Diffusion-LM](https://github.com/XiangLi1999/Diffusion-LM) | Hierarchical intermediate latent variables for gradient-based attribute control |

---

## Gap 7: Comprehensive Evaluation (E1-E5, CC4, CC5)

**Type:** Evaluation / Cross-cutting
**Core problem:** No standardized evaluation suite for controllable DLMs. No model evaluated on PPL + control accuracy + editing quality + reasoning + efficiency simultaneously.

### Direction 7.1: Controllable Text Generation Benchmarks

**Reasoning:** Existing benchmark suites define how to evaluate controllable generation quality.
**Search domains:** NLP evaluation, benchmark design

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **ConGenBench: Controllable Text Generation Benchmark** | arXiv | 2024 | [2405.01490](https://arxiv.org/abs/2405.01490) | [DhananjayAshok/ConGenBench](https://github.com/DhananjayAshok/ConGenBench) | 17 controllable tasks, 18 constraint datasets, 9 methods benchmarked |
| 2 | **CoDI-Eval: Diversified Instructions for LLMs** | AAAI | 2024 | [2401.00690](https://arxiv.org/abs/2401.00690) | [Xt-cyh/CoDI-Eval](https://github.com/Xt-cyh/CoDI-Eval) | Fine-grained constraint evaluation across instruction types |
| 3 | **LCTG Bench: LLM Controlled Text Generation** | arXiv | 2025 | [2501.15875](https://arxiv.org/abs/2501.15875) | [CyberAgentAILab/LCTG-Bench](https://github.com/CyberAgentAILab/LCTG-Bench) | Unified framework for format, character, keyword, prohibited-word constraints |
| 4 | **CTRLEval: Unsupervised Reference-Free Metric** | ACL | 2022 | [2204.00862](https://arxiv.org/abs/2204.00862) | [thu-coai/CTRLEval](https://github.com/thu-coai/CTRLEval) | Reference-free coherence, consistency, attribute relevance via text infilling |
| 5 | **CTG Survey: Controllable Text Generation** | arXiv | 2024 | [2408.12599](https://arxiv.org/abs/2408.12599) | [IAAR-Shanghai/CTGSurvey](https://github.com/IAAR-Shanghai/CTGSurvey) | Systematic taxonomy of CTG techniques and evaluation practices |

### Direction 7.2: Text Editing Quality Metrics

**Reasoning:** Measuring edit faithfulness, locality, and semantic preservation is critical for evaluating latent-steered editing.
**Search domains:** Knowledge editing, text simplification, style transfer evaluation

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **EasyEdit: Knowledge Editing Framework** | ACL (Demo) | 2024 | [2308.07269](https://arxiv.org/abs/2308.07269) | [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit) | Unified edit evaluation: reliability, generalization, locality, portability metrics |
| 2 | **EditEval: Instruction-Based Editing Benchmark** | CoNLL | 2024 | [2209.13331](https://arxiv.org/abs/2209.13331) | [facebookresearch/EditEval](https://github.com/facebookresearch/EditEval) | First instruction-based benchmark for text editing (cohesion, paraphrase, neutralization) |
| 3 | **TSTBench: Text Style Transfer Benchmark** | Entropy | 2025 | N/A | [FayeXXX/A-Benchmark-of-Text-Style-Transfer](https://github.com/FayeXXX/A-Benchmark-of-Text-Style-Transfer) | 13 TST algorithms, standardized protocol for style accuracy + content preservation + fluency |
| 4 | **FaithEval: Contextual Faithfulness** | ICLR | 2025 | [2410.03727](https://arxiv.org/abs/2410.03727) | [SalesforceAIResearch/FaithEval](https://github.com/SalesforceAIResearch/FaithEval) | Evaluates faithfulness across counterfactual contexts; 4.9K samples |
| 5 | **EASSE: Sentence Simplification Evaluation** | EMNLP | 2019 | [ACL Anthology](https://aclanthology.org/D19-3009/) | [feralvam/easse](https://github.com/feralvam/easse) | Standardized toolkit for SARI, BLEU, SAMSA, FKGL metrics on text simplification |

### Direction 7.3: Efficiency Evaluation for Diffusion Models

**Reasoning:** Fair efficiency comparison is essential. CCDD was criticized for missing efficiency metrics.
**Search domains:** Systems, inference optimization, benchmarking

| Rank | Paper | Venue | Year | Link | GitHub | How It Helps |
|------|-------|-------|------|------|--------|--------------|
| 1 | **dInfer: Efficient Inference for Diffusion LMs** | arXiv | 2025 | [2510.08666](https://arxiv.org/abs/2510.08666) | [inclusionAI/dInfer](https://github.com/inclusionAI/dInfer) | Modular inference framework with TPS/TPF metrics; 10x speedup; standardized efficiency measurement |
| 2 | **Fast-dLLM: KV Cache + Parallel Decoding** | ICLR | 2026 | [2505.22618](https://arxiv.org/abs/2505.22618) | [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) | Block-wise KV cache + confidence-aware parallel decoding; wall-clock benchmarks as baselines |
| 3 | **Efficient Diffusion Models Survey** | TMLR | 2025 | [2502.06805](https://arxiv.org/abs/2502.06805) | [AIoT-MLSys-Lab/Efficient-Diffusion-Model-Survey](https://github.com/AIoT-MLSys-Lab/Efficient-Diffusion-Model-Survey) | Comprehensive taxonomy of efficiency methods; reference for fair comparison methodology |
| 4 | **Prism: Hierarchical Search for Discrete Diffusion** | arXiv | 2025 | [2602.01842](https://arxiv.org/abs/2602.01842) | [viiika/Prism](https://github.com/viiika/Prism) | NFE-vs-accuracy tradeoff curves across 4 benchmarks and 3 DLMs |
| 5 | **MDLM: Masked Diffusion Language Model** | NeurIPS | 2024 | [2406.07524](https://arxiv.org/abs/2406.07524) | [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm) | ddpm_cache sampler 3-4x faster; PPL-vs-NFE evaluations as efficiency baselines |

---

## OpenReview Reviewer Insights

Key findings from analyzing reviewer comments on accepted and rejected DLM papers:

### CCDD (ICLR 2026, Rejected, scores 4/2/6/4)
- "Limited evaluation scope (only OWT/LM1B, only perplexity)"
- "Lacks Sudoku, coding, GSM8k benchmarks"
- "Training/inference cost inadequately explored"
- Reviewer RTsX: "The real advantage lies in improving representational efficiency of latent embeddings... mitigating information loss caused by logit quantization"

### Recurring Reviewer Demands Across All DLM Papers
1. **Every paper** criticized for insufficient downstream task evaluation beyond perplexity
2. **Every paper** missing efficiency/compute comparisons to AR models
3. **Controllable text generation** repeatedly requested but never adequately demonstrated
4. **Text editing** completely unaddressed in accepted DLM papers
5. **Variable-length generation** remains unsolved
6. **Large vocabulary guidance** does not scale due to normalization constants

### Papers Repeatedly Cited by Reviewers as Missing Baselines
- Discrete Flow Matching (Gat et al. 2024)
- Dirichlet Flow Matching (Stark et al. 2024)
- MaskGIT / MAGE
- DiffuSeq-v2
- Mask-Predict / Non-autoregressive MT baselines
- Plaid / CDCD (continuous text diffusion)
- DiffusionBERT

---

## Summary: All Solution Papers

Total unique papers found: ~100
Total with GitHub repos: ~85

| Gap | Direction | Top Paper | GitHub | Domain |
|-----|-----------|-----------|--------|--------|
| 1 (Architecture) | 1.1 MMDiT | SD3 / MMDiT | [Stability-AI/sd3.5](https://github.com/Stability-AI/sd3.5) | Image gen. |
| 1 (Architecture) | 1.2 Coupled Diffusion | Duo (Diffusion Duality) | [s-sahoo/duo](https://github.com/s-sahoo/duo) | Theory |
| 1 (Architecture) | 1.3 Hybrid Disc+Cont | DisCo-Diff | [gcorso/disco-diffdock](https://github.com/gcorso/disco-diffdock) | Molecular |
| 2 (Training) | 2.1 Joint VAE+Diff | REPA-E | [End2End-Diffusion/REPA-E](https://github.com/End2End-Diffusion/REPA-E) | Image gen. |
| 2 (Training) | 2.2 Loss Balancing | LibMTL | [median-research-group/LibMTL](https://github.com/median-research-group/LibMTL) | MTL |
| 2 (Training) | 2.3 Diff. Bridges | DRAKES | [ChenyuWang-Monica/DRAKES](https://github.com/ChenyuWang-Monica/DRAKES) | Bio/Text |
| 3 (Guidance) | 3.1 CFG in Latent | Simple Guidance | [kuleshov-group/discrete-diffusion-guidance](https://github.com/kuleshov-group/discrete-diffusion-guidance) | Text |
| 3 (Guidance) | 3.2 Reward Guidance | DRAKES | [ChenyuWang-Monica/DRAKES](https://github.com/ChenyuWang-Monica/DRAKES) | Bio/Text |
| 3 (Guidance) | 3.3 Universal | Universal Guidance | [arpitbansal297/Universal-Guided-Diffusion](https://github.com/arpitbansal297/Universal-Guided-Diffusion) | Image gen. |
| 4 (Editing) | 4.1 SDEdit for Text | DiffusER | [machelreid/diffuser](https://github.com/machelreid/diffuser) | Text |
| 4 (Editing) | 4.2 Masked + Cond. | SMDM | [ML-GSAI/SMDM](https://github.com/ML-GSAI/SMDM) | Text |
| 4 (Editing) | 4.3 Latent Style | LatentOps | [guangyliu/LatentOps](https://github.com/guangyliu/LatentOps) | Text |
| 5 (Seq2Seq) | 5.1 Enc-Dec Diff | DiffuSeq | [Shark-NLP/DiffuSeq](https://github.com/Shark-NLP/DiffuSeq) | Text |
| 5 (Seq2Seq) | 5.2 Soft Prompt | PLANNER | [apple/ml-planner](https://github.com/apple/ml-planner) | Text |
| 5 (Seq2Seq) | 5.3 DLM Reasoning | d1 | [dllm-reasoning/d1](https://github.com/dllm-reasoning/d1) | Reasoning |
| 6 (Geometry) | 6.1 Interpolation | Cosmos | [MeshchaninovViacheslav/cosmos](https://github.com/MeshchaninovViacheslav/cosmos) | Text |
| 6 (Geometry) | 6.2 Disentangled | MAGIC | [nju-websoft/MAGIC](https://github.com/nju-websoft/MAGIC) | Text |
| 7 (Evaluation) | 7.1 CTG Benchmark | ConGenBench | [DhananjayAshok/ConGenBench](https://github.com/DhananjayAshok/ConGenBench) | Eval |
| 7 (Evaluation) | 7.2 Edit Metrics | EasyEdit | [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit) | Eval |
| 7 (Evaluation) | 7.3 Efficiency | dInfer | [inclusionAI/dInfer](https://github.com/inclusionAI/dInfer) | Systems |
