---
layout: post
title: Enhancing Long-Horizon Embodied Agents in Open-World 3D Environments
categories: [embodied-ai, agents]
author: Nicolas Basile
---


Open‑world 3D environments like Minecraft are a uniquely challenging setting for embodied agents: rewards are sparse, action spaces are enormous, and complex tasks require extended temporal reasoning. 

<div style="display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px;">
  <video autoplay controls playsinline muted loop style="width:100%; border-radius:12px;">
    <source src="/videos/base.mp4" type="video/mp4" />
  </video>

  <video autoplay controls playsinline muted loop style="width:100%; border-radius:12px;">
    <source src="/videos/cactus.mp4" type="video/mp4" />
  </video>

  <video autoplay controls playsinline muted loop style="width:100%; border-radius:12px;">
    <source src="/videos/gold.mp4" type="video/mp4" />
  </video>

  <video autoplay controls playsinline muted loop style="width:100%; border-radius:12px;">
    <source src="/videos/pig.mp4" type="video/mp4" />
  </video>
</div>

<div style="margin-top:8px; color:#9aa0a6; font-size: 0.9em;">
  Credit to <a href="#ref-voyager" rel="noopener noreferrer" style="color:#9aa0a6; text-decoration: underline;">
    NVIDIA Voyager
  </a>
</div>

NVIDIA’s Voyager laid the foundation for autonomous curriculum learning and tool‑use, but long-horizon, multi-stage objectives still tend to break down due to **forgetting**, **shallow planning**, and **brittle tool selection**.

My goal in this project was to push those boundaries by:
- optimizing **long-horizon reliability** via memory, planning persistence, and tool robustness.
- and **automating improvement of the agent’s scaffolding** via verifier-driven optimization (DSPy<sup> [[1]](#ref-dspy)</sup>).

On a suite of multi-stage objectives, these changes improved success rates an average of **~20%** while reducing prompting iterations needed by **>40%** in several cases (relative to NVIDIA's baseline Voyager framework <sup>[[9]](#ref-voyager-code)</sup>).

- Evaluation: 5 random seeds × 4 episodes per seed = 20 episodes/task (no initial inventory).
- Prompting iterations: number of planner/tool-selector regeneration attempts, capped at 50 as in Voyager.
- Success criteria: Task dependent - typically an inventory check, or structure built.

---

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

1. <a id="ref-dspy"></a> **DSPy** — Khattab et al., *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines*. https://arxiv.org/abs/2310.03714

2. <a id="ref-reflexion"></a> **Reflexion** — Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*. https://arxiv.org/abs/2303.11366

3. <a id="ref-react"></a> **ReAct** — Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*. https://arxiv.org/abs/2210.03629

4. <a id="ref-vlm-reward-iclr"></a> **VLMs as Zero-Shot Reward Models** — Rocamonde et al., *Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning* (ICLR 2024). https://openreview.net/forum?id=N0I2RtD8je

5. <a id="ref-vlm-source-rewards"></a> **VLMs as a Source of Rewards** — Baumli et al., *Vision-Language Models as a Source of Rewards*. https://arxiv.org/abs/2312.09187

6. <a id="ref-odyssey"></a> **Odyssey** — Liu et al., *Odyssey: Empowering Minecraft Agents with Open-World Skills*. https://arxiv.org/abs/2407.15325

7. <a id="ref-odyssey-code"></a> **Odyssey (code)** — https://github.com/zju-vipa/Odyssey

8. <a id="ref-voyager"></a> **Voyager** - Wang et al., *Voyager: An Open-Ended Embodied Agent
with Large Language Models*. https://arxiv.org/pdf/2305.16291

9. <a id="ref-voyager-code"></a> **Voyager (code)** — https://github.com/MineDojo/Voyager
