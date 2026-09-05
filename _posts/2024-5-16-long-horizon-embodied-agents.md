---
layout: post
title: Enhancing Long-Horizon Embodied Agents in Open-World 3D Environments
categories: [embodied-ai, agents]
author: Nicolas Basile
description: "Building on NVIDIA’s Voyager, this project explores how hierarchical memory and a lightweight, RL-inspired policy search implemented with DSPy can improve an embodied agent's behavior. The learned policies improved success rates by ~20% while reducing inference cost."
hook: "Building on <strong>NVIDIA’s Voyager</strong>, this project explores how <strong>hierarchical memory</strong> and a lightweight, RL-inspired policy search implemented with <strong>DSPy</strong> can improve an embodied agent's behavior. The learned policies improved success rates by <strong>~20%</strong> while reducing inference cost."
media_type: video
media_url: /videos/base.mp4
media_alt: "Autonomous embodied agent skill execution in Minecraft"
media_url_2: /videos/cactus.mp4
media_type_2: video
media_alt_2: "Autonomous farm harvesting task"
findings:
  - stat: "~20%"
    label: Success-rate lift
  - stat: ">40%"
    label: Fewer prompt iterations
  - stat: "20"
    label: Episodes per task
hero_caption: 'Credit: <a href="#ref-voyager">NVIDIA Voyager</a>'
hero:
  - url: /videos/base.mp4
    type: video
    label: Build base
    alt: "Autonomous embodied agent skill execution in Minecraft"
  - url: /videos/cactus.mp4
    type: video
    label: Gather cactus
    alt: "Autonomous farm harvesting task"
  - url: /videos/gold.mp4
    type: video
    label: Mine gold
    alt: "Gold collection task"
  - url: /videos/pig.mp4
    type: video
    label: Hunt pig
    alt: "Livestock task"
---

## Introduction

NVIDIA’s Voyager laid the foundation for autonomous curriculum learning and tool‑use, but long-horizon, multi-stage objectives still tend to break down due to **forgetting**, **shallow planning**, and **brittle tool selection**.

My goal in this project was to push those boundaries by:
- optimizing **long-horizon reliability** via memory, planning persistence, and tool robustness.
- and **automating improvement of the agent’s scaffolding** via verifier-driven optimization (DSPy<sup> [[1]](#ref-dspy)</sup>).

On a suite of multi-stage objectives, these changes improved success rates an average of **~20%** while reducing prompting iterations needed by **>40%** in several cases (relative to NVIDIA's baseline Voyager framework <sup>[[9]](#ref-voyager-code)</sup>).

- Evaluation: 5 random seeds × 4 episodes per seed = 20 episodes/task (no initial inventory).
- Prompting iterations: number of planner/tool-selector regeneration attempts, capped at 50 as in Voyager.
- Success criteria: Task dependent - typically an inventory check, or structure built.

## What I built

This project improves upon the Voyager framework with:

### 1) Structured memory

- **Compressed episodic memory:** key events, failures, and environmental discoveries.
- **Task memory:** current high-level objective and sub-tasks, with progress markers.
- **Skill/tool memory:** what tools exist and when they’re useful.

This makes the agent’s behavior more consistent across long rollouts and reduces repeated mistakes. This also introduces an optimization point, allowing DSPy to learn optimal selective memory strategies.

> This mirrors a broader pattern in language-agent research: maintaining an episodic trace, compressing it into higher-level reflections, and retrieving it to steer future plans. <sup>[[2]](#ref-reflexion)</sup>

### 2) Hierarchical planning
The agent plans at multiple time scales:

- **High-level plan:** stages/subgoals (“collect wood → craft pickaxe → collect stone → build house”).
- **Mid-level steps:** concrete actions per stage (tool calls / environment API usage [i.e walk, mine, craft]).
- **Execution loop:** act → observe → update memory → re-plan only when needed.

The key change is **plan persistence**: the agent doesn’t re-derive its intent every step, which reduces thrashing.

### 3) Tool use that actually holds up over long horizons

- explicit tool “contracts” (inputs/outputs + failure handling),
- validation of tool results before moving on,
- recovery strategies when a tool call fails or returns unexpected state.

> Treating actions/tools as first-class outputs with validation aligns with the ReAct view of interleaving reasoning with environment actions to reduce cascading errors. <sup>[[3]](#ref-react)</sup>

---

## Prompt + policy improvement via DSPy

A major part of the gains came from automating improvement upon the *agentic scaffolding* itself.

### VLMs as reward models

For long-horizon tasks, sparse environment rewards are often unusable; instead, I use a *vision-language evaluator* to score progress from observable evidence (images + trace). This mirrors a growing line of work that treats pretrained VLMs as *zero-shot reward models* for language-conditioned goals <sup>[[4]](#ref-vlm-reward-iclr)</sup><sup>[[5]](#ref-vlm-source-rewards)</sup>, turning “is the goal satisfied?” into a learned scoring function over visual states.


Concretely, the evaluator consumes (goal, plan, tool calls, observations) and outputs a scalar score plus failure tags (e.g., missed grasp, wrong inventory precondition). Those scores become the optimization target for improving the agent’s scaffolding.

### DSPy turns scaffolding design into optimization

Rather than hand-tuning prompts, I treat the planner/tool-selector/memory-updater as an *optimizable program* and use DSPy-style compilation/teleprompting to search for module settings that **maximize verifier score**. This is closer to *black-box policy search over scaffolding* than weight-updating RL, useful when you want rapid iteration without finetuning.

> DSPy frames prompt-and-module design as an optimizable program (compile/teleprompt), replacing ad-hoc prompt tinkering with systematic search over module behaviors. 

This turned “prompt engineering” into something closer to **search + optimization**, using an internal (synthetic) reward function that is grounded in environmental evidence. As mentioned earlier, the majority of performance gains came from this iterative, evolution-like approach.

---

## Why this matters
Long-horizon embodied autonomy can’t be solved solely by "a bigger model" and "more data". In these environments, moving beyond impressive demos and into reliable behavior requires:

- memory that persists and stays relevant,
- plans that evolve with the environment,
- tool use that is robust to errors and infinite loops,
- evaluation signals that are aligned with success.

---

### Related work: Odyssey (open-world skills)

Recent work like **Odyssey** <sup>[[6]](#ref-odyssey)</sup><sup>[[7]](#ref-odyssey-code)</sup> extends the Voyager line by **expanding what the agent can do**: it equips Minecraft agents with a large **open-world skill library** (primitive + compositional skills), augments the base model with **domain-specific Minecraft knowledge** via a fine-tuned LLaMA-3 variant trained on a large Minecraft-Wiki Q&A corpus, and introduces a capability-oriented benchmark spanning **long-term planning**, **dynamic-immediate planning**, and **autonomous exploration** tasks.

This project is intentionally **orthogonal**. Rather than primarily increasing skill/action coverage or adding domain knowledge through finetuning, I focus on **making long-horizon behavior reliably hold up** *given* a set of tools, hardening execution and optimizing the agent’s scaffolding for consistency over long horizons.

---

## References

<ol class="ref-list">
  <li id="ref-dspy"><strong>DSPy</strong> — Khattab et al. <a href="https://arxiv.org/abs/2310.03714">arXiv:2310.03714</a></li>
  <li id="ref-reflexion"><strong>Reflexion</strong> — Shinn et al. <a href="https://arxiv.org/abs/2303.11366">arXiv:2303.11366</a></li>
  <li id="ref-react"><strong>ReAct</strong> — Yao et al. <a href="https://arxiv.org/abs/2210.03629">arXiv:2210.03629</a></li>
  <li id="ref-vlm-reward-iclr"><strong>VLMs as Zero-Shot Reward Models</strong> — Rocamonde et al., ICLR 2024. <a href="https://openreview.net/forum?id=N0I2RtD8je">OpenReview</a></li>
  <li id="ref-vlm-source-rewards"><strong>VLMs as a Source of Rewards</strong> — Baumli et al. <a href="https://arxiv.org/abs/2312.09187">arXiv:2312.09187</a></li>
  <li id="ref-odyssey"><strong>Odyssey</strong> — Liu et al. <a href="https://arxiv.org/abs/2407.15325">arXiv:2407.15325</a></li>
  <li id="ref-odyssey-code"><strong>Odyssey (code)</strong> — <a href="https://github.com/zju-vipa/Odyssey">github.com/zju-vipa/Odyssey</a></li>
  <li id="ref-voyager"><strong>Voyager</strong> — Wang et al. <a href="https://arxiv.org/pdf/2305.16291">arXiv:2305.16291</a></li>
  <li id="ref-voyager-code"><strong>Voyager (code)</strong> — <a href="https://github.com/MineDojo/Voyager">github.com/MineDojo/Voyager</a></li>
</ol>
