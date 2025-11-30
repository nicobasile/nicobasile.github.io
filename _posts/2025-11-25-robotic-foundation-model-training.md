---
layout: post
title: Building a Robotics Foundation Model - From Teleoperation Trajectories to GRPO-Optimized Control
categories: [embodied-ai, robotics]
author: Nicolas Basile
---

How far can you push planning in a Vision-Language-Action (VLA) model before you have to confront the messiness of real, high-dimensional control?

In my earlier work on Minecraft-style open-world agents, extracting embodied planning purely via **text** worked surprisingly well. 

A VLA's world model + agentic scaffolding could take a high-level instruction, decompose it into fine-grained sub-tasks, and then call tools inside the game’s API to execute long-horizon objectives. For that setting, representing “actions” as code, tool calls, or scripted macros was enough to achieve impressive behavior.

But that setup is tightly coupled to Minecraft’s hard-coded action space. It exposed two issues that don’t go away once you leave the sandbox:

1. **Autoregressive latency:** If every low-level move is produced via another forward pass and tool call, long-horizon control becomes painfully slow. In a peaceful sandbox game this is fine, but in real world robotics it's a non-starter.
2. **Action fidelity and transfer:** A discrete, engine-specific `action → effect` mapping does not cleanly generalize to continuous, high-dimensional spaces like robot joint torques or end-effector velocities.

This project is my attempt to push that paradigm into robotics.

To cut to the chase, the core idea is to keep the *planning* and world-knowledge benefits of a VLA, while changing how we read out actions:

1. **Use the latent plan, not its text:** Instead of sampling a textual plan, we can use the final embedding state directly. In fact, sampling the latent plan into tokens collapses dimensionality and fidelity purely for human readability; the model's internal state is richer than the text it emits.
2. **Attach and train an action-decoder head:** On top of this plan embedding (plus proprioception and a small window of past states), I add a learned action head that predicts *continuous control* — **end-effector deltas**. The head is trained with standard behavioral cloning losses (e.g., MSE ) against teleoperation trajectories, so it learns to interpret the latent plan as “what to do next” in robot action space. At inference time, a single VLA forward pass produces actions directly through this head, solving the latency problem of the autoregressive text loop.

I trained the model on a blend of real teleoperation (collected on my home-setup) and high-quality synthetic trajectories, then further refined with GRPO reinforcement learning in MuJoCo/MJX.

> This is conceptually similar to how **Pi₀** functions: a vision-language backbone encodes the scene and the task, while a separate “action expert” is trained to map those embeddings to high-frequency continuous actions. Their research was a large inspiration for this project. :contentReference[oaicite:0]{index=0} (https://www.physicalintelligence.company/blog/pi0)

Concretely, I set out to answer:

**Can a single Vision-Language-Action (VLA) model, trained on real and synthetic teleoperation data, learn temporally consistent, instruction-conditioned robotic control comparable to emerging frontier systems like Gemini Robotics and GR-3—without relying on a hand-coded “action → fine control” interface?**

The rest of this post walks through that pipeline end-to-end: data collection, synthetic trajectory generation, VLA fine-tuning, and GRPO-based RL alignment in simulation:

---

*# TODO(nbasile): Finish*

