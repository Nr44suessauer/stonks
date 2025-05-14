Title: Deep Dive into Deepseek: Training Process, RL techniques, MoE, ...

DeepSeek is a state-of-the-art language model developed by a Chinese AI startup that has rapidly garnered attention for its breakthrough training methodologies. At the heart of DeepSeek lies a novel training process that combines pure reinforcement learning (RL) with chain-of-thought prompting and an advanced Mixture-of-Experts (MoE) architecture. This hybrid approach not only enables the model to “think” step by step—self-reflecting and refining its reasoning—but also significantly reduces computational costs by activating only task-specific expert modules. In this project, the goal is to dissect DeepSeek’s training process, understand how its RL reward systems (e.g. Group Relative Policy Optimization) drive emergent “aha moments,” and evaluate how its MoE design contributes to efficiency and scalability across diverse tasks like mathematical problem solving, code generation, and general language understanding.

Research Questions <br>
1. How does Deepseeks fine-tuning strategy work?
2. How does DeepSeek’s pure reinforcement learning strategy, integrated with chain-of-thought prompting, improve reasoning accuracy compared to conventional supervised fine-tuning?
3. What role does the Mixture-of-Experts architecture play in reducing training and inference costs while preserving model performance?
4. How effective are the designed reward mechanisms—balancing accuracy and output formatting—in guiding DeepSeek’s self-improvement?
5. Which new GPU optimizations did deepseek implement, to make their model faster?


References <br>
- [Reuters Article] DeepSeek rushes to launch new AI model as China goes all in (2025) <br>
https://www.reuters.com/technology/artificial-intelligence/deepseek-rushes-launch-new-ai-model-china-goes-all-2025-02-25/
- [Paper] “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning”, Guo et al. (2025) <br>
https://arxiv.org/abs/2501.12948
- [Paper] “DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model”, DeepSeek-AI et al. (2024) <br>
https://arxiv.org/abs/2405.04434
- [Youtube Video]  "DeepSeek-R1 Paper Explained - A New RL LLMs Era in AI?" <br>
https://www.youtube.com/watch?v=DCqqCLlsIBU&t=369s
- [Github] "Deepseek's GitHub Organization"
https://github.com/deepseek-ai

Notes from the Initial Briefing - 22.04.2025 <br>
- main: research existing work
- technically & chronically aspects for training
- touch all aspects in a broader spectrum
  - focus on: reinforcement training
  - maybe also: GPU optimization (for example as one aspect to deepen the knowledge)
- comparison with conventional LLMs
  - allowed: use-case during comparison