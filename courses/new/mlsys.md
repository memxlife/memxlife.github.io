# Machine Learning Systems (ML Sys)

Machine Learning Systems (ML Sys) studies how to build efficient, scalable, and reliable AI systems through the co-design of algorithms, system software, and hardware. Modern AI systems are fundamentally constrained and enabled by computing infrastructure, including GPUs, NPUs, memory hierarchy, communication systems, compilers, runtime systems, and large-scale distributed computing. This course introduces the full stack of modern AI infrastructure, spanning GPU/NPU architecture, CUDA programming, parallel training, large language model (LLM) inference optimization, and hardware-software co-optimization for large-scale AI systems.

Unlike traditional lecture-heavy system courses, ML Sys is designed as a highly hands-on, engineering-driven, in-class hackathons centered around real-world AI infrastructure development. The course emphasizes practical system building, performance analysis, optimization, and iterative debugging on modern GPU and NPU platforms. Students will not only learn how modern AI infrastructure works, but also directly build and optimize core system components used in large-scale AI training and inference systems.

A central theme of the course is the emerging paradigm of Agentic AI-driven system development. With the rise of large language models and multi-agent infrastructures, AI is increasingly becoming an active participant in software and hardware system design. Throughout the course, students will learn how to use large language model-based multi-agent systems to automate profiling, kernel generation, optimization, debugging, experiment scheduling, and infrastructure development for modern AI accelerators.

The course places particularly strong emphasis on AI infrastructure development for both GPUs and NPUs, the two dominant hardware platforms driving modern AI computing. Students will work closely with real-world GPU and NPU software stacks, including CUDA programming, communication kernels, distributed training infrastructure, and large-scale inference systems. Through a sequence of in-class hackathons, students will iteratively develop multi-agent infrastructures capable of automatically generating and optimizing AI system components.

机器学习系统（Machine Learning Systems, ML Sys）研究如何通过算法、系统软件与硬件的协同设计，构建高效、可扩展且可靠的人工智能系统。现代 AI 系统的发展，本质上受到计算基础设施的约束与推动，包括 GPU、NPU、存储层次结构、通信系统、编译器、运行时系统以及大规模分布式计算等。本课程将系统介绍现代 AI 基础设施的完整技术栈，涵盖 GPU/NPU 架构、CUDA 编程、并行训练、大语言模型（LLM）推理优化，以及面向大规模 AI 系统的软硬件协同优化。

不同于传统以理论讲授为主的系统课程，ML Sys 是一门高度强调实践、工程能力与课堂 Hackathon 的课程，核心围绕真实世界 AI 基础设施开发展开。课程重点关注现代 GPU 与 NPU 平台上的系统构建、性能分析、优化与迭代调试。学生不仅将学习现代 AI 基础设施“如何工作”，更将亲手构建与优化大规模 AI 训练与推理系统中的核心系统组件。

本课程的一个核心主题，是 Agentic AI 驱动的系统开发新范式。随着大语言模型与多智能体基础设施的发展，AI 正逐渐从“工具”演变为软硬件系统设计过程中的主动参与者。在课程中，学生将学习如何利用基于大语言模型的多智能体系统，实现 profiling、kernel 自动生成、系统优化、调试、实验调度以及 AI 加速器基础设施开发的自动化。

课程特别强调面向 GPU 与 NPU 的 AI 基础设施开发。GPU 与 NPU 是当前推动现代 AI 计算发展的两类核心硬件平台。学生将在课程中深入接触真实世界 GPU 与 NPU 软件栈，包括 CUDA 编程、通信 Kernel、分布式训练基础设施以及大规模推理系统。通过一系列课堂 Hackathon，学生将逐步构建能够自动生成与优化 AI 系统组件的多智能体基础设施。

## Course Philosophy

> Not everyone can become an AI inference expert, but an expert may come from anyone in 51 hours.

We believe the future of AI infrastructure development should not be limited to a small number of highly specialized engineers inside large organizations. By combining system thinking, engineering capability, and Agentic AI methodologies, individual students can increasingly build and optimize sophisticated AI systems that previously required large expert teams.

The course is therefore structured around a sequence of lectures and hands-on hackathons. The lectures establish the architectural and system foundations behind modern AI infrastructure, while the hackathons focus on solving real-world engineering problems using innovative large language model-based agentic infrastructures. Students will build systems that automatically profile hardware, analyze performance bottlenecks, generate optimized kernels, develop communication runtimes, and optimize distributed AI infrastructure for modern accelerators such as GPUs and Ascend NPUs.

By the end of the course, students are expected to understand not only how AI infrastructure systems are designed today, but also how future intelligent computing systems may increasingly be generated, optimized, and continuously evolved through human-AI collaborative system development.

我们相信，未来的 AI 基础设施开发不应只属于大型组织中少数高度专业化的工程师。通过系统思维、工程能力与 Agentic AI 方法论的结合，越来越多的学生将能够构建与优化过去需要大型专家团队才能完成的复杂 AI 系统。

因此，本课程围绕“课程讲授 + 实践 Hackathon”两部分展开。课程讲授部分将建立现代 AI 基础设施背后的系统与架构基础；而 Hackathon 部分则聚焦于利用基于大语言模型的 Agentic AI 基础设施，解决真实世界中的工程问题。学生将在课程中构建能够自动完成硬件 profiling、性能瓶颈分析、Kernel 自动生成与优化、通信运行时开发，以及分布式 AI 基础设施优化的系统，并面向 GPU 与昇腾 NPU 等现代 AI 加速器开展实践。

课程结束时，学生不仅需要理解今天的 AI 基础设施系统是如何被设计出来的，更需要理解未来的智能计算系统将如何通过“人类 + AI”的协同系统开发方式，被自动生成、持续优化并不断演化。

# Course Structure

The course is organized around two tightly integrated components:

1. **System Foundations Lectures**  
   Students will learn the architectural foundations of modern AI infrastructure, including GPUs, NPUs, CUDA programming, distributed training, communication systems, large-scale inference infrastructure, and hardware–software co-optimization.

2. **Hands-on Agentic AI Hackathons**  
   Students will develop large language model-based multi-agent infrastructures capable of automating profiling, kernel generation, optimization, debugging, runtime orchestration, and distributed AI system optimization on modern GPU and NPU platforms.

A major emphasis of the course is practical engineering capability. Students are expected to solve real-world AI infrastructure problems through iterative system building, experimentation, profiling, and optimization.

The course progressively moves from:
- understanding AI hardware architectures,
- to low-level kernel optimization,
- to distributed training systems,
- to large-scale inference infrastructure,
- and finally toward AI-driven automated infrastructure development.

Throughout the semester, students will work closely with modern GPU and NPU software stacks, including CUDA and Ascend platforms, while developing Agentic AI infrastructures for automated system optimization.

## Course Schedule

| Week | Module | Type | Topic | Homework / Project | Instructor |
|---|---|---|---|---|---|
| 1 | Introduction | Lecture | Introduction: The Intellectual Map of Machine Learning Systems | | SHANG |
| 2 | GPU/NPU Architecture | Lecture | GPU Architecture for Machine Learning Systems | Multi-agent infrastructure for automated GPU performance profiling and modeling | SHANG |
| 3 | GPU/NPU Architecture | Lecture | NPU Architecture for Machine Learning Systems | Multi-agent infrastructure for automated NPU performance profiling and modeling | TIAN |
| 4 | CUDA Programming | Lecture | CUDA Programming Through the Lens of GPU Architecture | | SHANG |
| 5 | CUDA Programming | Lecture | CUDA Programming as Hardware–Software Co-Optimization | | SHANG |
| 6 | GPU Kernel Development | In-class Hackathon | Multi-Agent Infrastructure for Automated CUDA Kernel Development and Optimization on GPUs | Multi-agent infrastructure for automated CUDA kernel generation and optimization | CHEN |
| 7 | NPU Kernel Development | In-class Hackathon | Multi-Agent Infrastructure for Automated Kernel Development and Optimization on NPUs | Multi-agent infrastructure for automated NPU kernel development and optimization | CHEN |
| 8 | NPU Kernel Development | In-class Hackathon | Multi-Agent Infrastructure for Automated FlashAttention Development and Optimization on NPUs | Multi-agent infrastructure for automated FlashAttention optimization | TIAN |
| 9 | LLM Training | Lecture | Data, Tensor, Pipeline, and Expert Parallelism in Large Language Model Training | | XU |
| 10 | LLM Training | Lecture | Advanced Topics in Large-Scale LLM Training Systems | | XU |
| 11 | LLM Training | In-class Hackathon | Multi-Agent Infrastructure for Automated Communication Kernel Development and Optimization | Multi-agent infrastructure for communication kernel optimization | XU |
| 12 | LLM Training | In-class Hackathon | Multi-Agent Infrastructure for Automated Parallel Training Infrastructure Development and Optimization | Multi-agent infrastructure for distributed training infrastructure optimization | TIAN |
| 13 | LLM Inference | Lecture | LLM Inference Systems A–Z | | XU |
| 14 | LLM Inference | Lecture | Advanced Topics in LLM Inference Optimization | | XU |
| 15 | LLM Inference | In-class Hackathon | Multi-Agent Infrastructure for Automated LLM Inference Infrastructure Development and Optimization | Multi-agent infrastructure for LLM inference infrastructure optimization | TIAN |
| 16 | LLM Optimization | Lecture | Advanced Topics in LLM Optimization and System Co-Design | Course Project: “黄大年难题” Challenges | SHANG |
| 17 | AI Infrastructure A–Z | Final Presentation | Course Project Presentation and System Demonstration | Final Project Presentation | ALL |

## Instructors

### Professor CHEN, Chongliang 
### Professor XU, Yuedong 
### Mr. TIAN, Kuyang 
### Professor SHANG, Li
---

