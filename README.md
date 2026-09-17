# 🗺️ AI Topics Map — Complete Guide

> A structured, deduplicated, and link-rich map of Artificial Intelligence — from mathematical foundations to frontier research, engineering, safety, and real-world applications.
>
> **Legend:** 📄 paper · 🛠️ tool/docs · 📚 course/book · 🔗 resource hub

## 📑 Contents

- [1. Foundations](#1-foundations)
- [2. Core Architectures](#2-core-architectures)
- [3. Learning Paradigms](#3-learning-paradigms)
- [4. LLM and Generative AI Engineering](#4-llm-and-generative-ai-engineering)
- [5. Efficient AI and Systems](#5-efficient-ai-and-systems)
- [6. Data for AI](#6-data-for-ai)
- [7. MLOps and LLMOps](#7-mlops-and-llmops)
- [8. Evaluation, Interpretability, Safety and Governance](#8-evaluation-interpretability-safety-and-governance)
- [9. Applications by Domain](#9-applications-by-domain)
- [10. Frontier and Interdisciplinary AI](#10-frontier-and-interdisciplinary-ai)
- [11. Learning Roadmap](#11-learning-roadmap)

---

## 1. Foundations

> 🧩 The mathematical and conceptual building blocks of ML and DL.

### 1.1 Mathematical Foundations

- **Linear Algebra** — vectors, matrices, eigenvalues/eigenvectors, SVD, matrix decompositions
- **Probability and Statistics** — Bayes' theorem, distributions, MLE/MAP, hypothesis testing, confidence intervals
- **Information Theory** — entropy, mutual information, KL divergence, cross-entropy
- **Calculus and Optimization** — gradients, Jacobians, convex optimization, Lagrange multipliers, gradient descent variants
- **Statistical Learning Theory** — bias–variance tradeoff, VC dimension, PAC learning, generalization
- **Graph Theory** — graph algorithms, spectral graph theory, centrality (for GNNs)
- **Numerical Methods** — numerical linear algebra, floating-point stability, iterative solvers
- 📚 [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) · [Mathematics for Machine Learning](https://mml-book.github.io/) · [Probabilistic ML (Murphy)](https://probml.github.io/pml-book/) · [Convex Optimization (Boyd)](https://web.stanford.edu/~boyd/cvxbook/) · [Information Theory, Inference, and Learning Algorithms (MacKay)](https://www.inference.org.uk/mackay/itila/) · [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) · [An Introduction to Statistical Learning](https://www.statlearning.com/)
- 🔗 [StatQuest](https://www.youtube.com/@statquest) · [NetworkX (graphs)](https://networkx.org/documentation/stable/)

### 1.2 Classical Machine Learning

- **Supervised** — linear/logistic regression, SVM, k-NN, decision trees, random forests, gradient boosting (XGBoost, LightGBM, CatBoost)
- **Unsupervised** — k-means, DBSCAN, GMM, hierarchical clustering, PCA, t-SNE, UMAP, anomaly detection (Isolation Forest, One-Class SVM)
- **Feature engineering** — scaling, encoding, imputation, feature selection, target leakage
- **Model selection** — cross-validation, hyperparameter tuning (grid, random, Bayesian), metrics (ROC-AUC, PR-AUC, F1, calibration)
- 🛠️ [scikit-learn](https://scikit-learn.org/stable/) · [XGBoost](https://xgboost.readthedocs.io/) · [LightGBM](https://lightgbm.readthedocs.io/) · [Optuna](https://optuna.readthedocs.io/) · [UMAP](https://umap-learn.readthedocs.io/)

### 1.3 Deep Learning Fundamentals

- MLP, backpropagation, activation functions, initialization, vanishing/exploding gradients
- Regularization: Dropout, weight decay, early stopping, data augmentation
- Normalization: BatchNorm, LayerNorm, RMSNorm, GroupNorm
- Optimizers: SGD + Momentum, AdaGrad, RMSProp, Adam/AdamW, Lion, Muon
- Loss functions: cross-entropy, MSE, focal, contrastive (InfoNCE), triplet
- Embeddings and tokenization: Word2Vec, GloVe, fastText, BPE, WordPiece, SentencePiece
- Attention and the Transformer: self-attention, multi-head attention, positional encodings
- 📄 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [Adam](https://arxiv.org/abs/1412.6980) · [BatchNorm](https://arxiv.org/abs/1502.03167) · [LayerNorm](https://arxiv.org/abs/1607.06450) · [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) · [He initialization](https://arxiv.org/abs/1502.01852) · [Word2Vec](https://arxiv.org/abs/1301.3781) · [fastText](https://arxiv.org/abs/1607.04606)
- 📚 [Dive into Deep Learning](https://d2l.ai/) · [Deep Learning book](https://www.deeplearningbook.org/) · [CS231n](https://cs231n.stanford.edu/) · [Karpathy: Neural Networks Zero to Hero](https://karpathy.ai/zero-to-hero.html) · [fast.ai](https://course.fast.ai/)

### 1.4 Scaling and Training Dynamics

- Scaling laws, compute-optimal training, emergent abilities
- Pretraining vs fine-tuning vs post-training; data mixtures; compute budgets
- 📄 [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) · [Chinchilla](https://arxiv.org/abs/2203.15556) · [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)

---

## 2. Core Architectures

> 🏗️ Model families across vision, language, graphs, multimodality, and generation.

### 2.1 Computer Vision

- **Classification CNNs** — LeNet, AlexNet, VGG, GoogLeNet/Inception, ResNet, DenseNet, MobileNet, EfficientNet, ConvNeXt
- **Object detection** — R-CNN family, YOLO family, SSD, RetinaNet, DETR
- **Segmentation** — FCN, U-Net, DeepLab, Mask R-CNN, SAM / SAM 2
- **Tracking and MOT** — SORT, DeepSORT, ByteTrack, OC-SORT, MOTR
- **Vision Transformers** — ViT, DeiT, Swin, BEiT, MAE, DINO/DINOv2
- **3D vision** — NeRF, Gaussian Splatting, PointNet/PointNet++, point clouds, SLAM, depth/stereo
- 📄 [ResNet](https://arxiv.org/abs/1512.03385) · [VGG](https://arxiv.org/abs/1409.1556) · [EfficientNet](https://arxiv.org/abs/1905.11946) · [YOLO](https://arxiv.org/abs/1506.02640) · [Faster R-CNN](https://arxiv.org/abs/1506.01497) · [DETR](https://arxiv.org/abs/2005.12872) · [U-Net](https://arxiv.org/abs/1505.04597) · [Mask R-CNN](https://arxiv.org/abs/1703.06870) · [SAM](https://arxiv.org/abs/2304.02643) · [DeepSORT](https://arxiv.org/abs/1703.07402) · [ByteTrack](https://arxiv.org/abs/2110.06864) · [ViT](https://arxiv.org/abs/2010.11929) · [Swin Transformer](https://arxiv.org/abs/2103.14030) · [NeRF](https://arxiv.org/abs/2003.08934) · [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) · [PointNet](https://arxiv.org/abs/1612.00593)
- 🛠️ [torchvision](https://pytorch.org/vision/stable/index.html) · [Detectron2](https://detectron2.readthedocs.io/) · [MMDetection](https://mmdetection.readthedocs.io/) · [Ultralytics](https://docs.ultralytics.com/) · [OpenCV](https://docs.opencv.org/) · [Albumentations](https://albumentations.ai/docs/) · [Open3D](https://www.open3d.org/docs/)

### 2.2 Sequence and Language Models

- **Foundations** — RNN, LSTM, GRU, seq2seq, attention, Transformer
- **Encoders** — BERT, RoBERTa, DistilBERT, ELECTRA, DeBERTa
- **Encoder–decoders** — T5, BART, mT5
- **Decoders / LLMs** — GPT, LLaMA, Mistral, Qwen, DeepSeek, Gemma, Phi, OLMo, Falcon
- **Code models** — Codex, StarCoder, CodeLlama, DeepSeek-Coder
- **Tokenization** — BPE, WordPiece, SentencePiece, byte-level BPE
- 📄 [BERT](https://arxiv.org/abs/1810.04805) · [RoBERTa](https://arxiv.org/abs/1907.11692) · [T5](https://arxiv.org/abs/1910.10683) · [BART](https://arxiv.org/abs/1910.13461) · [GPT-3](https://arxiv.org/abs/2005.14165) · [LLaMA](https://arxiv.org/abs/2302.13971) · [Llama 3](https://arxiv.org/abs/2407.21783) · [Mistral 7B](https://arxiv.org/abs/2310.06825) · [DeepSeek-V3](https://arxiv.org/abs/2412.19437) · [StarCoder](https://arxiv.org/abs/2305.06161) · [BPE](https://arxiv.org/abs/1508.07909) · [SentencePiece](https://arxiv.org/abs/1808.06226)
- 📚 [CS224n: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) · [Speech and Language Processing (Jurafsky & Martin)](https://web.stanford.edu/~jurafsky/slp3/)

### 2.3 Generative Models

- **GANs** — DCGAN, StyleGAN, conditional GANs
- **VAEs and flows** — VAE, NICE, RealNVP, Glow
- **Diffusion** — DDPM, score-based SDEs, latent diffusion, Diffusion Transformers (DiT), flow matching, rectified flow
- **Controllable generation** — ControlNet, IP-Adapter, LoRA for diffusion, guidance (CFG)
- **Video and 3D generation** — diffusion transformers for video, DreamFusion, Gaussian Splatting pipelines
- 📄 [GAN](https://arxiv.org/abs/1406.2661) · [StyleGAN](https://arxiv.org/abs/1812.04948) · [VAE](https://arxiv.org/abs/1312.6114) · [DDPM](https://arxiv.org/abs/2006.11239) · [Score-Based SDEs](https://arxiv.org/abs/2011.13456) · [Latent Diffusion](https://arxiv.org/abs/2112.10752) · [DiT](https://arxiv.org/abs/2212.09748) · [Flow Matching](https://arxiv.org/abs/2210.02747) · [ControlNet](https://arxiv.org/abs/2302.05543) · [DreamFusion](https://arxiv.org/abs/2209.14988)
- 🛠️ [Diffusers](https://huggingface.co/docs/diffusers)

### 2.4 Graph Neural Networks

- Message passing, GCN, GAT, GraphSAGE, GIN, PNA
- Graph transformers, heterogeneous graphs, temporal/dynamic graphs
- Applications: molecules, knowledge graphs, recommendation, fraud, traffic
- 📄 [GCN](https://arxiv.org/abs/1609.02907) · [GAT](https://arxiv.org/abs/1710.10903) · [GraphSAGE](https://arxiv.org/abs/1706.02216) · [GIN](https://arxiv.org/abs/1810.00826) · [GraphGPS](https://arxiv.org/abs/2205.12454)
- 🛠️ [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) · [DGL](https://docs.dgl.ai/)

### 2.5 Multimodal Models

- **Vision–language** — CLIP, BLIP-2, LLaVA, Flamingo, PaLM-E, Qwen-VL
- **Audio–language and video–language** — AudioLM, VideoLLM families
- **Any-to-any models** — unified tokenizers, early/late fusion, cross-attention
- **Capabilities** — grounding, OCR/document understanding, VQA, captioning
- 📄 [CLIP](https://arxiv.org/abs/2103.00020) · [Flamingo](https://arxiv.org/abs/2204.14198) · [BLIP-2](https://arxiv.org/abs/2301.12597) · [LLaVA](https://arxiv.org/abs/2304.08485) · [PaLM-E](https://arxiv.org/abs/2303.03378) · [ImageBind](https://arxiv.org/abs/2305.05665) · [AudioLM](https://arxiv.org/abs/2209.03143)

### 2.6 Emerging Architectures

- **State Space Models** — S4, Mamba, Mamba-2, RWKV
- **Mixture of Experts** — Switch Transformer, GShard, Mixtral, expert routing/balancing
- **Long context** — Longformer, RoPE scaling, ring attention, memory layers
- **Alternatives** — RetNet, Hyena, recurrent-depth, hybrid attention–SSM stacks
- **World models and VLA** — JEPA, Genie, RT-2, OpenVLA, diffusion policies
- 📄 [Mamba](https://arxiv.org/abs/2312.00752) · [Mamba-2](https://arxiv.org/abs/2405.21060) · [RWKV](https://arxiv.org/abs/2305.13048) · [Switch Transformer](https://arxiv.org/abs/2101.03961) · [Mixtral](https://arxiv.org/abs/2401.04088) · [Longformer](https://arxiv.org/abs/2004.05150) · [RetNet](https://arxiv.org/abs/2307.08621) · [I-JEPA](https://arxiv.org/abs/2301.08243) · [Genie](https://arxiv.org/abs/2402.15391) · [RT-2](https://arxiv.org/abs/2307.15818) · [OpenVLA](https://arxiv.org/abs/2406.09246)

---

## 3. Learning Paradigms

> 🎯 How models learn: supervision, rewards, adaptation, and beyond.

### 3.1 Core Paradigms

- Supervised, unsupervised, semi-supervised
- Self-supervised: contrastive (SimCLR, MoCo), predictive (BYOL), masked modeling (BERT, MAE)
- Reinforcement learning, imitation learning, offline RL
- Transfer, multi-task, and multi-modal learning

### 3.2 Reinforcement Learning

- MDPs, value/policy iteration, Q-learning, DQN
- Policy gradients: REINFORCE, A2C/A3C, TRPO, PPO
- Actor–critic: DDPG, TD3, SAC
- Model-based: Dreamer, MuZero, AlphaZero
- Offline RL, multi-agent RL, hierarchical RL, exploration
- RL for LLMs: RLHF, RLAIF, RLVR, GRPO, reward modeling
- 📄 [DQN](https://www.nature.com/articles/nature14236) · [PPO](https://arxiv.org/abs/1707.06347) · [SAC](https://arxiv.org/abs/1801.01290) · [TD3](https://arxiv.org/abs/1802.09477) · [AlphaZero](https://www.science.org/doi/10.1126/science.aar6404) · [MuZero](https://www.nature.com/articles/s41586-020-03051-4) · [DreamerV3](https://arxiv.org/abs/2301.04104) · [Decision Transformer](https://arxiv.org/abs/2106.01345)
- 📚 [Sutton & Barto: Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html) · [OpenAI Spinning Up](https://spinningup.openai.com/)
- 🛠️ [Gymnasium](https://gymnasium.farama.org/) · [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) · [CleanRL](https://docs.cleanrl.dev/) · [PettingZoo](https://pettingzoo.farama.org/) · [RLlib](https://docs.ray.io/en/latest/rllib/index.html)

### 3.3 Adaptation and Fine-Tuning

- Full fine-tuning vs parameter-efficient fine-tuning (PEFT)
- LoRA, QLoRA, DoRA, adapters, prefix/prompt tuning, IA3
- Instruction tuning (FLAN, Self-Instruct), chat templates
- Knowledge distillation, model merging, model soups, task arithmetic
- 📄 [LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA](https://arxiv.org/abs/2305.14314) · [DoRA](https://arxiv.org/abs/2402.09353) · [Adapters](https://arxiv.org/abs/1902.00751) · [Prefix-Tuning](https://arxiv.org/abs/2101.00190) · [FLAN](https://arxiv.org/abs/2109.01652) · [Self-Instruct](https://arxiv.org/abs/2212.10560) · [Knowledge Distillation](https://arxiv.org/abs/1503.02531) · [Model Soups](https://arxiv.org/abs/2203.05482)
- 🛠️ [HF PEFT](https://huggingface.co/docs/peft) · [HF TRL](https://huggingface.co/docs/trl) · [Unsloth](https://docs.unsloth.ai/)

### 3.4 Other Learning Settings

- Few-shot / zero-shot / in-context learning
- Meta-learning (MAML, ProtoNets), continual learning, curriculum learning
- Active learning, weakly supervised learning, noisy-label learning
- Federated learning, differential privacy, privacy-preserving ML
- 📄 [MAML](https://arxiv.org/abs/1703.03400) · [ProtoNets](https://arxiv.org/abs/1703.05175) · [SimCLR](https://arxiv.org/abs/2002.05709) · [BYOL](https://arxiv.org/abs/2006.07733) · [MAE](https://arxiv.org/abs/2111.06377) · [FedAvg](https://arxiv.org/abs/1602.05629) · [DP-SGD](https://arxiv.org/abs/1607.00133) · [EWC](https://www.pnas.org/doi/10.1073/pnas.1611835114)
- 🛠️ [Flower (federated)](https://flower.ai/docs/framework/) · [Opacus (DP)](https://opacus.ai/)

---

## 4. LLM and Generative AI Engineering

> 🤖 The modern applied stack: prompting, retrieval, agents, training, serving, and evaluation.

### 4.1 Prompt and Context Engineering

- Zero/few-shot prompting, system prompts, role prompting
- Chain-of-Thought, zero-shot CoT, self-consistency, least-to-most
- ReAct, Tree of Thoughts, Graph of Thoughts, Reflexion, self-refine
- Context engineering: context budgets, compaction, memory, structured output
- Prompt optimization: DSPy, automatic prompt tuning
- 📄 [Chain-of-Thought](https://arxiv.org/abs/2201.11903) · [Self-Consistency](https://arxiv.org/abs/2203.11171) · [ReAct](https://arxiv.org/abs/2210.03629) · [Tree of Thoughts](https://arxiv.org/abs/2305.10601) · [Reflexion](https://arxiv.org/abs/2303.11366)
- 🔗 [Prompt Engineering Guide](https://www.promptingguide.ai/) · [DSPy](https://dspy.ai/) · [OpenAI Cookbook](https://cookbook.openai.com/)

### 4.2 Retrieval-Augmented Generation (RAG)

- Embeddings and semantic search; chunking; hybrid search (BM25 + dense)
- Reranking (cross-encoders), query rewriting, HyDE, multi-hop retrieval
- Vector databases: FAISS, Chroma, Qdrant, Weaviate, Milvus, pgvector, Pinecone
- GraphRAG, agentic RAG, multimodal RAG; long-context vs RAG tradeoffs
- RAG evaluation: faithfulness, relevance, RAGAS, LLM-as-a-judge
- 📄 [RAG](https://arxiv.org/abs/2005.11401) · [DPR](https://arxiv.org/abs/2004.04906) · [ColBERT](https://arxiv.org/abs/2004.12832) · [Sentence-BERT](https://arxiv.org/abs/1908.10084) · [HNSW](https://arxiv.org/abs/1603.09320) · [Lost in the Middle](https://arxiv.org/abs/2307.03172) · [GraphRAG](https://arxiv.org/abs/2404.16130) · [RAGAS](https://arxiv.org/abs/2309.15217)
- 🛠️ [LlamaIndex](https://docs.llamaindex.ai/) · [LangChain](https://python.langchain.com/docs/introduction/) · [Qdrant](https://qdrant.tech/documentation/) · [Chroma](https://docs.trychroma.com/) · [pgvector](https://github.com/pgvector/pgvector)

### 4.3 Agents and Tool Use

- Tool/function calling, structured outputs, sandboxing
- Planning, reasoning, reflection, memory (short-term, long-term, episodic)
- Multi-agent systems, orchestration, human-in-the-loop
- Computer-use / browser / coding agents
- Standards and protocols: Model Context Protocol (MCP), A2A
- 📄 [Toolformer](https://arxiv.org/abs/2302.04761) · [Gorilla](https://arxiv.org/abs/2305.15334) · [Generative Agents](https://arxiv.org/abs/2304.03442) · [Voyager](https://arxiv.org/abs/2305.16291) · [MemGPT](https://arxiv.org/abs/2310.08560) · [SWE-agent](https://arxiv.org/abs/2405.15793) · [Agent Survey](https://arxiv.org/abs/2308.11432)
- 🛠️ [MCP](https://modelcontextprotocol.io/) · [LangGraph](https://langchain-ai.github.io/langgraph/) · [AutoGen](https://microsoft.github.io/autogen/) · [CrewAI](https://docs.crewai.com/) · [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)

### 4.4 LLM Training: Pretraining, Post-Training, Alignment

- Pretraining objectives, data curation, tokenizer training, compute budgets
- SFT, instruction tuning, chat templates
- Reward modeling, RLHF, RLAIF, Constitutional AI, DPO/ORPO/KTO/SimPO, GRPO, RLVR
- Reasoning models and test-time compute / inference scaling
- Model editing, unlearning, distillation into smaller models
- 📄 [InstructGPT](https://arxiv.org/abs/2203.02155) · [Constitutional AI](https://arxiv.org/abs/2212.08073) · [DPO](https://arxiv.org/abs/2305.18290) · [ORPO](https://arxiv.org/abs/2403.07691) · [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300) · [DeepSeek-R1](https://arxiv.org/abs/2501.12948) · [Scaling Test-Time Compute](https://arxiv.org/abs/2408.03314) · [ROME (model editing)](https://arxiv.org/abs/2202.05262) · [Learning to Summarize (reward models)](https://arxiv.org/abs/2009.01325)

### 4.5 Inference, Serving, and Cost

- KV cache, PagedAttention, continuous batching, prefix caching
- Speculative decoding, Medusa, EAGLE; quantization (see §5.4)
- Structured/constrained generation (grammars, JSON schema)
- Runtimes: vLLM, SGLang, TensorRT-LLM, TGI, llama.cpp, Ollama, LMDeploy
- Cost/latency optimization: routing, caching, small models, distillation
- 📄 [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) · [Speculative Decoding](https://arxiv.org/abs/2211.17192) · [Medusa](https://arxiv.org/abs/2401.10774) · [EAGLE](https://arxiv.org/abs/2401.15077) · [GQA](https://arxiv.org/abs/2305.13245)
- 🛠️ [vLLM](https://docs.vllm.ai/) · [SGLang](https://docs.sglang.ai/) · [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [Ollama](https://ollama.com/)

### 4.6 Foundation Model Families

- **Open weights** — LLaMA 2/3, Mistral/Mixtral, Qwen, DeepSeek, Gemma, Phi, OLMo, Falcon
- **Closed** — GPT-4/4o, Claude, Gemini, Grok
- **Scientific** — AlphaFold, ESM, GraphCast, AlphaGeometry (see §10)
- 📄 [GPT-4](https://arxiv.org/abs/2303.08774) · [Gemini 1.5](https://arxiv.org/abs/2403.05530) · [Gemma](https://arxiv.org/abs/2403.08295) · [Qwen2.5](https://arxiv.org/abs/2412.15115) · [Mixtral](https://arxiv.org/abs/2401.04088)

---

## 5. Efficient AI and Systems

> ⚡ Hardware, distributed training, attention efficiency, compression, and edge deployment.

### 5.1 Hardware and Kernels

- GPUs, TPUs, accelerators; memory hierarchy, roofline model
- CUDA/Triton kernel programming; fused kernels; profiling (Nsight)
- 🔗 [CUDA docs](https://docs.nvidia.com/cuda/) · [CUDA Mode lectures](https://github.com/cuda-mode/lectures) · [Triton](https://triton-lang.org/main/index.html) · [TPU docs](https://cloud.google.com/tpu/docs)

### 5.2 Distributed Training

- Data / tensor / pipeline / expert parallelism; 3D parallelism
- ZeRO, FSDP, DeepSpeed, Megatron-LM, sequence parallelism
- Mixed precision (fp16/bf16/fp8), gradient checkpointing, activation offloading
- 📄 [ZeRO](https://arxiv.org/abs/1910.02054) · [Megatron-LM](https://arxiv.org/abs/1909.08053) · [GPipe](https://arxiv.org/abs/1811.06965) · [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- 🛠️ [DeepSpeed](https://www.deepspeed.ai/) · [FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) · [Ray](https://docs.ray.io/)

### 5.3 Attention and Context Efficiency

- FlashAttention 1/2/3, memory-efficient attention
- MQA, GQA, MLA; KV-cache compression/eviction; sliding window; sparse attention
- Positional encoding: RoPE, ALiBi, NTK scaling, YaRN
- 📄 [FlashAttention](https://arxiv.org/abs/2205.14135) · [FlashAttention-2](https://arxiv.org/abs/2307.08691) · [RoPE](https://arxiv.org/abs/2104.09864) · [ALiBi](https://arxiv.org/abs/2108.12409) · [Ring Attention](https://arxiv.org/abs/2310.01889)

### 5.4 Model Compression and Acceleration

- Quantization: post-training (GPTQ, AWQ, SmoothQuant, LLM.int8), QAT, GGUF formats
- Pruning: structured/unstructured, SparseGPT, Wanda
- Distillation, low-rank factorization, early exit, MoE routing efficiency
- 📄 [GPTQ](https://arxiv.org/abs/2210.17323) · [AWQ](https://arxiv.org/abs/2306.00978) · [SmoothQuant](https://arxiv.org/abs/2211.10438) · [LLM.int8()](https://arxiv.org/abs/2208.07339) · [SparseGPT](https://arxiv.org/abs/2301.00774)

### 5.5 Edge, Mobile, and On-Device AI

- TinyML, edge quantization/pruning, IoT AI, on-device learning
- Runtimes: ONNX Runtime, TensorRT, OpenVINO, LiteRT, ExecuTorch, MLC-LLM, MLX
- 🛠️ [ONNX Runtime](https://onnxruntime.ai/docs/) · [TensorRT](https://developer.nvidia.com/tensorrt) · [OpenVINO](https://docs.openvino.ai/) · [LiteRT](https://ai.google.dev/edge) · [ExecuTorch](https://pytorch.org/executorch/) · [MLX](https://ml-explore.github.io/mlx/) · [TinyML Foundation](https://www.tinyml.org/)

---

## 6. Data for AI

> 📊 Pipelines, quality, versioning, and data-centric methods.

### 6.1 Data Engineering and Infrastructure

- Batch/stream processing: Spark, Flink, Kafka, dbt
- Lakehouse: Delta Lake, Iceberg, Hudi; object storage; ELT
- Orchestration: Airflow, Prefect, Dagster
- 🛠️ [Spark](https://spark.apache.org/docs/latest/) · [Kafka](https://kafka.apache.org/documentation/) · [Flink](https://flink.apache.org/) · [Airflow](https://airflow.apache.org/docs/) · [Dagster](https://docs.dagster.io/) · [Prefect](https://docs.prefect.io/) · [Delta Lake](https://docs.delta.io/) · [Iceberg](https://iceberg.apache.org/docs/latest/)

### 6.2 Data Management and Versioning

- Versioning: DVC, lakeFS, Git LFS; lineage and catalogs
- Feature stores: Feast, Tecton; online/offline consistency
- 🛠️ [DVC](https://dvc.org/doc) · [lakeFS](https://docs.lakefs.io/) · [Feast](https://docs.feast.dev/) · [Tecton](https://docs.tecton.ai/) · [HF Datasets](https://huggingface.co/docs/datasets)

### 6.3 Data Quality, Labeling, and Synthetic Data

- Cleaning, dedup (MinHash), PII handling, filtering, quality metrics
- Labeling: annotation tools, guidelines, inter-annotator agreement, weak supervision
- Synthetic data: LLM-generated data, GAN augmentation, self-instruct, distillation data
- Data-centric AI: error analysis, slice-based evaluation, data debugging
- 🔗 [Label Studio](https://labelstud.io/) · [Argilla](https://docs.argilla.io/) · [Cleanlab](https://docs.cleanlab.ai/) · [Great Expectations](https://docs.greatexpectations.io/) · [Pandera](https://pandera.readthedocs.io/) · [MIT Data-Centric AI](https://dcai.csail.mit.edu/)

### 6.4 Data for Foundation Models

- Pretraining corpora: Common Crawl, The Pile, LAION, FineWeb
- Curation and dedup at scale, quality classifiers, decontamination
- Data mixtures, curriculum, tokenizer training, licensing
- 🔗 [Common Crawl](https://commoncrawl.org/) · [The Pile](https://arxiv.org/abs/2101.00027) · [LAION](https://laion.ai/)

---

## 7. MLOps and LLMOps

> 🔧 From experimentation to reliable production systems.

### 7.1 Experimentation and Reproducibility

- Experiment tracking, model registry, artifact/environment management
- Reproducibility: seeds, configs, dataset versions, containerization
- 🛠️ [MLflow](https://mlflow.org/docs/latest/) · [Weights & Biases](https://docs.wandb.ai/) · [TensorBoard](https://www.tensorflow.org/tensorboard) · [Hydra](https://hydra.cc/docs/intro/) · [Docker](https://docs.docker.com/)

### 7.2 Deployment and Serving

- Patterns: batch, online, streaming, serverless; canary, blue/green, shadow
- Containers, orchestration, autoscaling, GPU scheduling
- Serving stacks: Triton, KServe, BentoML, Ray Serve, Modal
- 🛠️ [Kubernetes](https://kubernetes.io/docs/home/) · [KServe](https://kserve.github.io/website/) · [BentoML](https://docs.bentoml.com/) · [Triton Inference Server](https://developer.nvidia.com/triton-inference-server)

### 7.3 Monitoring and Reliability

- Data drift, concept drift, bias monitoring, performance tracking
- Logging, tracing, alerting, incident response, rollback
- LLM observability: prompt/response tracing, token/cost tracking, eval-in-production
- 🛠️ [Evidently](https://docs.evidentlyai.com/) · [Arize](https://docs.arize.com/) · [NannyML](https://docs.nannyml.com/) · [Langfuse](https://langfuse.com/docs) · [LangSmith](https://docs.smith.langchain.com/) · [Phoenix](https://docs.arize.com/phoenix)

### 7.4 CI/CD and Quality for ML

- Testing ML: data tests, model tests, behavioral tests, integration tests
- CI/CD/CT pipelines, automated retraining, feature/model registries
- Prompt and agent evaluation in CI; regression gates
- 🛠️ [Kubeflow](https://www.kubeflow.org/docs/) · [Metaflow](https://docs.metaflow.org/) · [ZenML](https://docs.zenml.io/) · [Promptfoo](https://www.promptfoo.dev/docs/intro/)

---

## 8. Evaluation, Interpretability, Safety and Governance

> 🛡️ Measuring, understanding, and controlling AI systems.

### 8.1 Evaluation

- Benchmarks: MMLU, BIG-bench, GPQA, MATH, HumanEval, SWE-bench, MMMU, HELM
- Human and model-based evaluation: Chatbot Arena, LLM-as-a-judge, pairwise preference
- Methodology: contamination, statistical significance, robustness, eval harnesses
- 📄 [MMLU](https://arxiv.org/abs/2009.03300) · [BIG-bench](https://arxiv.org/abs/2206.04615) · [HumanEval](https://arxiv.org/abs/2107.03374) · [SWE-bench](https://arxiv.org/abs/2310.06770) · [GPQA](https://arxiv.org/abs/2311.12022) · [MATH](https://arxiv.org/abs/2103.03874) · [Chatbot Arena](https://arxiv.org/abs/2403.04132) · [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)
- 🛠️ [HELM](https://crfm.stanford.edu/helm/) · [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) · [OpenAI Evals](https://github.com/openai/evals) · [Inspect (UK AISI)](https://inspect.aisi.org.uk/)

### 8.2 Interpretability and Explainability

- Post-hoc: SHAP, LIME, Grad-CAM, integrated gradients, feature visualization
- Mechanistic interpretability: circuits, induction heads, superposition, sparse autoencoders
- Concept-based and causal interpretability; probing
- 📄 [LIME](https://arxiv.org/abs/1602.04938) · [SHAP](https://arxiv.org/abs/1705.07874) · [Grad-CAM](https://arxiv.org/abs/1610.02391) · [Integrated Gradients](https://arxiv.org/abs/1703.01365) · [Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) · [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- 🔗 [Captum](https://captum.ai/) · [Transformer Circuits](https://transformer-circuits.pub/) · [Getting Started in Mech Interp](https://www.neelnanda.io/mechanistic-interpretability/getting-started) · [Distill: Feature Visualization](https://distill.pub/2017/feature-visualization/)

### 8.3 Robustness and Security

- Adversarial examples and defenses, OOD detection, calibration, conformal prediction
- Prompt injection, jailbreaks, data poisoning, backdoors
- Privacy: membership inference, differential privacy, unlearning, watermarking
- 📄 [Adversarial Examples](https://arxiv.org/abs/1312.6199) · [PGD](https://arxiv.org/abs/1706.06083) · [Calibration](https://arxiv.org/abs/1706.04599) · [Conformal Prediction](https://arxiv.org/abs/2107.07511) · [Prompt Injection](https://arxiv.org/abs/2302.12173) · [Universal Jailbreaks](https://arxiv.org/abs/2307.02483) · [BadNets](https://arxiv.org/abs/1708.06733) · [Membership Inference](https://arxiv.org/abs/1610.05820) · [Watermarking](https://arxiv.org/abs/2301.10226)
- 🛡️ [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) · [MITRE ATLAS](https://atlas.mitre.org/)

### 8.4 Alignment and Safety

- Alignment: RLHF/RLAIF, Constitutional AI, scalable oversight, superalignment
- Failure modes: deception, reward hacking, sycophancy, goal misgeneralization
- Safety engineering: red teaming, evals, guardrails, model cards, incident reporting
- 📄 [Risks from Learned Optimization](https://arxiv.org/abs/1906.01820) · [AI Safety via Debate](https://arxiv.org/abs/1805.00899) · [Red Teaming LMs](https://arxiv.org/abs/2209.07858) · [Weak-to-Strong Generalization](https://arxiv.org/abs/2312.09390) · [Model Cards](https://arxiv.org/abs/1810.03993) · [Datasheets](https://arxiv.org/abs/1803.09010)
- 🔗 [Alignment Forum](https://www.alignmentforum.org/)

### 8.5 Fairness, Ethics, and Governance

- Bias and fairness metrics, fairness toolkits, disparate impact
- Transparency, accountability, audits, documentation
- Regulation: EU AI Act, NIST AI RMF, ISO/IEC 42001, OECD principles
- 🔗 [Fair ML Book](https://fairmlbook.org/) · [EU AI Act](https://artificialintelligenceact.eu/) · [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) · [ISO/IEC 42001](https://www.iso.org/standard/42001) · [OECD AI Principles](https://oecd.ai/en/ai-principles)

---

## 9. Applications by Domain

> 🌍 Where AI is applied, with representative methods and resources.

- **Computer Vision** — classification, detection, segmentation, tracking, OCR, pose, depth, 3D reconstruction, medical imaging
- **Video** — action recognition, temporal localization, video understanding, video generation, tracking
- **Speech and Audio** — ASR, TTS, voice cloning, diarization, music generation
  - 📄 [Whisper](https://arxiv.org/abs/2212.04356) · [wav2vec 2.0](https://arxiv.org/abs/2006.11477) · [VITS (TTS)](https://arxiv.org/abs/2106.06103) · [MusicGen](https://arxiv.org/abs/2306.05284) · 🛠️ [SpeechBrain](https://speechbrain.readthedocs.io/)
- **NLP** — classification, NER, translation, summarization, QA, information extraction, LLM applications
- **Code** — completion, generation, review, repository-level tasks, SWE agents
- **Time Series** — forecasting, anomaly detection, nowcasting
  - 📄 [Chronos](https://arxiv.org/abs/2403.07815) · [TimesFM](https://arxiv.org/abs/2310.10688) · [DeepAR](https://arxiv.org/abs/1704.04110) · [Informer](https://arxiv.org/abs/2012.07436) · 🛠️ [Nixtla](https://nixtlaverse.nixtla.io/) · [sktime](https://www.sktime.net/)
- **Graph Data** — social networks, knowledge graphs, molecules, fraud, traffic, supply chains
- **Recommenders and Search** — collaborative filtering, two-tower models, sequential recommendation, ranking, personalization
  - 📄 [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) · [BERT4Rec](https://arxiv.org/abs/1904.06690) · [DLRM](https://arxiv.org/abs/1906.00091) · 🛠️ [Microsoft Recommenders](https://github.com/microsoft/recommenders) · [RecBole](https://recbole.io/)
- **Robotics and Autonomy** — motion planning, manipulation, navigation, self-driving, drones, sim2real
  - 🛠️ [ROS 2](https://docs.ros.org/en/rolling/) · [MuJoCo](https://mujoco.readthedocs.io/) · [CARLA](https://carla.readthedocs.io/) · [LeRobot](https://github.com/huggingface/lerobot) · [nuScenes](https://www.nuscenes.org/)
- **Healthcare and Biology** — medical imaging, drug discovery, genomics, protein folding
  - 📄 [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) · [BioBERT](https://arxiv.org/abs/1901.08746) · 🛠️ [ESM](https://github.com/facebookresearch/esm) · [OpenFold](https://github.com/aqlaboratory/openfold) · [MONAI](https://monai.io/) · [RDKit](https://www.rdkit.org/docs/)
- **Finance** — trading, fraud detection, credit risk, AML, document processing
- **Cybersecurity** — intrusion detection, malware analysis, phishing detection, SOC automation
- **Industry** — predictive maintenance, quality control, supply chain, manufacturing vision
- **Energy and Climate** — smart grids, renewable forecasting, climate modeling
  - 📄 [GraphCast](https://arxiv.org/abs/2212.12794) · 🛠️ [PyPSA](https://pypsa.readthedocs.io/) · [Grid2Op](https://grid2op.readthedocs.io/)
- **Agriculture** — precision farming, crop monitoring, yield prediction, remote sensing
- **Education, Legal, Government, Defense** — tutoring, document review, public services, surveillance (with strict governance)
- **Entertainment and Media** — generative content, gaming AI, dubbing, VFX

---

## 10. Frontier and Interdisciplinary AI

> 🔮 Research frontiers beyond the standard applied stack.

- **World Models and Embodied AI** — 📄 [World Models](https://arxiv.org/abs/1803.10122) · [DreamerV3](https://arxiv.org/abs/2301.04104) · [Genie](https://arxiv.org/abs/2402.15391) · [I-JEPA](https://arxiv.org/abs/2301.08243)
- **AGI and Superintelligence** — 📄 [Levels of AGI](https://arxiv.org/abs/2311.02462) · 🔗 [Situational Awareness](https://situational-awareness.ai/)
- **Brain-Computer Interfaces** — neural signal processing, decoding, neuroprosthetics
- **Causal AI** — causal discovery, causal inference, causal representation learning
  - 📚 [Causal Inference: What If](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/) · 📄 [Causal Representation Learning](https://arxiv.org/abs/2102.11107) · 🛠️ [DoWhy](https://py-why.github.io/dowhy/)
- **Neuro-Symbolic AI** — program synthesis, formal reasoning, theorem proving, verification
- **Quantum Machine Learning** — 🔗 [PennyLane QML](https://pennylane.ai/qml/) · [Qiskit Machine Learning](https://qiskit-community.github.io/qiskit-machine-learning/)
- **Neuromorphic Computing** — spiking neural networks, event-based vision · 🛠️ [snnTorch](https://snntorch.readthedocs.io/)
- **AI for Science** — 📄 [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) · [AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-5) · [GraphCast](https://arxiv.org/abs/2212.12794) · 🛠️ [ESM](https://github.com/facebookresearch/esm)
- **Human-AI Collaboration** — copilots, mixed-initiative systems, augmented intelligence
- **Autonomous Agents and Self-Improvement** — long-horizon agents, self-play, self-improvement, agent economies
- **Multimodal and Any-to-Any** — real-time multimodal interaction, speech-to-speech, embodied multimodal models

---

## 11. Learning Roadmap

> 🎯 A practical progression from beginner to expert.

### 11.1 By Role

| Role | Core Path |
|------|-----------|
| Data Scientist | Statistics → Classical ML → Feature Engineering → Experimentation |
| ML Engineer | Python/SQL → Deep Learning → MLOps → Distributed Systems |
| AI/GenAI Engineer | LLM APIs → Prompting → RAG → Agents → Fine-tuning → Serving |
| Research Scientist | Math → ML Theory → Papers → Novel Algorithms → Publications |
| MLOps/Platform Engineer | Docker/K8s → CI/CD → Orchestration → Monitoring → Cost/Scale |
| AI Product/Strategy | Business → Use-Case Scoping → Evaluation → Governance/ROI |

### 11.2 Skill Priority

- **High** — Python, SQL, linear algebra and probability, classical ML, deep learning, Git, cloud basics, prompting and RAG
- **Medium** — MLOps, distributed training, optimization, evaluation, security, system design
- **Low (specialize later)** — quantum ML, neuromorphic computing, niche domains, exotic architectures

### 11.3 Progression

**Beginner (0–6 months)**
- Skills: Python, NumPy/Pandas, SQL, basic ML, scikit-learn, visualization, Git
- Resources: 📚 [Andrew Ng: Machine Learning](https://www.coursera.org/specializations/machine-learning-introduction) · [Kaggle Learn](https://www.kaggle.com/learn) · [fast.ai](https://course.fast.ai/) · [D2L](https://d2l.ai/) · [ISL](https://www.statlearning.com/)
- Projects: Titanic, house prices, digit classification, a small end-to-end prediction service

**Intermediate (6–24 months)**
- Skills: PyTorch, CNNs/RNNs/Transformers, transfer learning, MLOps basics, cloud, one domain
- Resources: 📚 [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) · [CS231n](https://cs231n.stanford.edu/) · [CS224n](https://web.stanford.edu/class/cs224n/) · [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) · [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
- Projects: image classifier with deployment, fine-tuned LLM, RAG chatbot, Kaggle competition

**Advanced (2–5 years)**
- Skills: LLM training/fine-tuning, evaluation, distributed systems, research reading, system design, leadership
- Resources: 📚 [Hugging Face Agents Course](https://huggingface.co/learn/agents-course) · [CS25: Transformers United](https://web.stanford.edu/class/cs25/) · [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) · [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) · [Chip Huyen: ML Systems](https://huyenchip.com/)
- Conferences: [NeurIPS](https://neurips.cc/) · [ICML](https://icml.cc/) · [ICLR](https://iclr.cc/)
- Projects: production ML platform, LLM fine-tune + serve at scale, research reproduction, open-source contribution

**Expert (5+ years)**
- Skills: novel research, frontier systems, organizational strategy, mentorship, governance
- Impact: publications, patents, products, standards, leading teams
- Resources: 🔗 [Papers with Code](https://paperswithcode.com/) · [Hugging Face Papers](https://huggingface.co/papers) · [The Batch](https://www.deeplearning.ai/the-batch/) · [Latent Space](https://www.latent.space/) · [Import AI](https://importai.substack.com/)

### 11.4 Project Ladder

1. Classical ML end-to-end with clean evaluation
2. CNN or Transformer trained from scratch on a small domain
3. Fine-tune an open LLM (LoRA) on custom data
4. RAG system with evaluation and monitoring
5. Agent with tools, memory, and guardrails
6. Deployed service with CI/CD, drift monitoring, and cost controls
7. Reproduce a paper or publish an experiment

### 11.5 How to Read Papers

- 📄 [How to Read a Paper (Keshav)](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf)
- Three passes: bird's-eye view → details → re-implementation
- Always note: problem, assumptions, method, evidence, limitations
- Keep a paper log; reproduce the smallest possible result

### 11.6 Staying Current

- 🔗 [arXiv cs.LG](https://arxiv.org/list/cs.LG/recent) · [arXiv cs.CL](https://arxiv.org/list/cs.CL/recent) · [arXiv cs.CV](https://arxiv.org/list/cs.CV/recent)
- Newsletters: [The Batch](https://www.deeplearning.ai/the-batch/) · [Import AI](https://importai.substack.com/) · [Latent Space](https://www.latent.space/)
- Communities: [Kaggle](https://www.kaggle.com/) · [Hugging Face](https://huggingface.co/) · [Alignment Forum](https://www.alignmentforum.org/) · [Papers with Code](https://paperswithcode.com/)

---

*Built to be a complete, deduplicated map — from foundations to frontier. Contributions welcome.* 🚀
