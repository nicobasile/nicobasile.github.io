---
layout: post
title: Building a Robotics Foundation Model - From Simulation To The Real World
categories: [embodied-ai, robotics]
author: Nicolas Basile
---

(WIP hook) How far can you push planning in a Vision-Language-Action (VLA) model before you have to confront the messiness of real, high-dimensional control?
How far can you push the **planning** strengths of a Vision-Language(-Action) model before you collide with the reality of **high-frequency continuous control**?

In my earlier work on Minecraft-style open-world agents, **extracting embodied planning purely via text** worked well, and facilitated observability and multi-agent connectivity.

A vision-language-model's world model had sufficient understanding of Minecraft, that, when combined with agentic scaffolding, it was able to take a high-level instruction, decompose it into fine-grained sub-tasks, and then call tools inside the game’s API to execute long-horizon objectives. For that setting, **representing “actions” as code, tool calls, or scripted macros was enough** to achieve impressive behavior.

**However** - that setup is tightly coupled to Minecraft’s hardcoded action space. It exposes two additional issues that don’t go away once you leave the sandbox:

1. **Autoregressive latency:** If every low-level actuation requires a full autoregressive forward pass, control loops fall below the frequency required for dynamic stability (e.g., <10Hz).
2. **Action fidelity and transfer:** A discrete, simulation-specific `action → effect` mapping (e.g 'move forward 10 blocks') does not cleanly generalize to continuous, high-dimensional spaces (like robot joint torques or end-effector velocities).

- smooth this
This project is my attempt to push that paradigm into robotics.

To cut to the chase, the core idea is to keep the *planning* and world-knowledge benefits of a VLM, while changing how we read out actions.
-- end smooth

<!-- <div style="display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px;">
  <img src="/videos/side-1.gif"  style="width:100%; border-radius:12px;" />

  <img src="/videos/top-2.gif"  style="width:100%; border-radius:12px;" />

  <img src="/videos/top-1.gif"  style="width:100%; border-radius:12px;" />

  <img src="/videos/side-2.gif"  style="width:100%; border-radius:12px;" />
</div> -->

<head>
<style>
  /* The main grid container */
  .gif-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
  }

  /* The black container box for each GIF */
  .gif-wrapper {
    position: relative;
    background-color: #151529; /* Black-blue background for padding */
    border-radius: 12px;       /* Rounded corners move to the container */
    width: 100%;
    /* IMPORTANT: This defines the shape of all boxes. 
       Adjust "4 / 3" to match the aspect ratio of your largest (3rd) GIF 
       if you know it (e.g., "16 / 9" for wide, "1 / 1" for square).
    */
    aspect-ratio: 4 / 3; 
    
    /* Flexbox used to center the image inside the black box */
    display: flex;
    justify-content: center; /* Centers horizontally */
    align-items: center;     /* Centers vertically */
    overflow: hidden;        /* Ensures image corners don't bleed past border-radius */
  }

  /* The images themselves */
  .gif-wrapper img {
    /* This ensures the image fits within the box without stretching */
    max-width: 100%;
    max-height: 100%;
    object-fit: contain; 
    display: block; /* Removes tiny inline spacing gaps */
  }
</style>
</head>

<div class="gif-grid">
  <div class="gif-wrapper">
    <img src="/videos/side-1.gif" alt="GIF 1" />
  </div>

  <div class="gif-wrapper">
     <img src="/videos/top-1.gif" alt="GIF 3" />
  </div>

  <div class="gif-wrapper">
     <img src="/videos/side-2.gif" alt="GIF 4" />
  </div>

  <div class="gif-wrapper">
     <img src="/videos/top-2.gif" alt="GIF 2" />
  </div>
</div>

<div style="margin-top:8px; color:#9aa0a6; font-size: 0.9em;">
  <b>Top:</b> From the final fine-tuned model we see more confident movement (Experiment C.)<br>
  <b>Bottom:</b> Jittery + uncertain movement from a less performant training run (Experiment A.)
</div>

---

## Table of Contents
- [Architecture: Vision-Language-Action Model](#architecture-vision-language-action-model)
- [Action Generation: Receding Horizon Control](#action-generation-receding-horizon-control)
- [Data: Mixing Real and Synthetic Trajectories](#data-mixing-real-and-synthetic-trajectories)
- [Training: The Importance of Batch Ratios](#training-the-importance-of-batch-ratios)
  - [Hypothesis: Synthetic Overfitting](#hypothesis)
  - [Ablation: Real-to-synthetic batch composition](#ablation-real-to-synthetic-batch-composition)
- [Future Research Directions](#future-research-directions)
  - [Data Generation via Simulation](#data-generation-via-simulation)
  - [Cross-Embodiment & End-Effector Control](#cross-embodiment--end-effector-ee-control)
- [References](#references)

---

## Architecture: Vision-Language-Action Model

1. **Use the latent plan directly:** Instead of sampling a textual plan token by token, we can use the final embedding state directly. In fact, sampling the latent plan into tokens collapses dimensionality and fidelity purely for human readability - *the model's internal state is richer than the text it emits.*
2. **Attach and train an action-decoder head:** On top of this plan embedding (plus proprioception and a small window of past states), I add a learned action head that predicts *continuous control* — **end-effector deltas**. The head is trained with standard behavioral cloning losses (e.g., MSE ) against teleoperation trajectories, so it learns to interpret the latent plan as “what to do next” in robot action space. At inference time, a single VLA forward pass produces actions directly through this head, solving the latency problem of the autoregressive text loop.

<div style="display: flex; flex-direction: column; align-items: center; width: 100%;">
  <img src="/images/model_architecture.jpg" style="max-width: 100%; border-radius: 12px; margin-top:8px" alt="Receding-Horizon Control" />
</div>

## Action Generation: Receding Horizon Control 

My robotic arm runs at **30 Hz** (one action every 33.3 ms). A single forward pass from our model predicts an action chunk of length **H = 50**, i.e. **1.67 seconds** of control.

On an NVIDIA 3060 Ti, one forward pass takes ~100 ms. That means the controller can stay ahead of the robot:

- Robot executes queued actions continuously,
- Model asynchronously refreshes the queue,
- **Any remaining queued actions are overwritten** by the newest prediction (receding-horizon control).

<div style="display: flex; flex-direction: column; align-items: center; width: 100%; border: 1px solid gray; border-radius: 3px">
  <img src="/images/action_chunks.jpg" style="max-width: 100%; border-radius: 12px; margin-top:8px" alt="Receding-Horizon Control" />
</div>

---

## Data: Mixing Real and Synthetic Trajectories

To train this system, I use a blend of:

- **Real teleoperation data** collected on my instructor arm setup
- **Synthetic trajectories** generated via:
    - **Segmentation-based augmentation** [x](cite datasocks)
        - Segments robot arm and target objects
        - Applies background replacements while maintaining foreground elements
    - **Kornia visual augments** [X](cite kornia) for color jittering, contrast adjustments, and geometric perturbations, applied consistently across a trajectory

<div style="display: flex; flex-direction: column; align-items: center; width: 100%;">
  <img src="/videos/augmented_synthetic_data.gif" style="max-width: 100%; border-radius: 12px; margin-top:8px" alt="Augmented Synthetic Data" />
  
  <div style="margin-top:8px; color:#9aa0a6; font-size: 0.9em;">
    Achieved a similar effect with my teleop data. Credit to <a href="#ref-datasocks" rel="noopener noreferrer" style="color:#9aa0a6; text-decoration: underline;">
      DataSocks
    </a>
  </div>
</div>

Following established best practices for VLA fine-tuning, the real data collection focused on diversity rather than sheer volume. I collected **N=50** high-quality "anchor" episodes, ensuring varied robot joint configurations by starting and ending trajectories in five distinct zones around the workspace. This diversity is crucial for establishing a robust inverse kinematics prior in the base model.

For each of these 50 real trajectories, I tested a variable number of realistic augmentations, ending on **Y=20**, resulting in a **total dataset of 1,050 trajectories** (50 real, 1000 synthetic).

## Training: The Importance of Batch Ratios

I expected 'more synthetic augmented data = better', but early runs showed the opposite: naive sampling from a 95% synthetic pool made the policy brittle and sometimes worse.

This section is the core experimental story.

(v CUT)
Early experiments demonstrated that standard probabilistic sampling from the imbalanced dataset led to suboptimal performance. I found that explicitly managing the ratio of real-to-synthetic data within each training batch was essential.

I drew inspiration from research into stabilizing generative model training via "Golden Ratio Mixing" [4], which posits that maintaining a high proportion of real data in minibatches—even necessitating heavy oversampling (repetition) of the real examples, is necessary to anchor the model in reality while allowing synthetic data to smooth the decision manifold.
(^ CUT)

(TEMP)

### Hypothesis
When synthetic dominates minibatches, the model overfits synthetic artifacts and “forgets” the real contact/lighting statistics that matter at deployment.

This resembles the *real/synthetic anchoring* arguments explored in Golden Ratio Mixing work on stabilizing training with synthetic data. [4] (Different domain, similar failure mode.)

### Ablation: real-to-synthetic batch composition

I evaluated several mixing strategies on a standard pick-and-place task.
I tested progressive augmentation ratios, finding a distinct performance inflection point requiring the repetition of real data:

- Baseline (zero-shot SmolVLA): **~5%** success due to domain shift in my custom setup.
- Fine-tuned variants:
| Experiment | Dataset Composition | Batch Strategy (Real:Warped) | Notes | Est. Success Rate |
|---|---|---|---|---|
| Exp. A | N=50, Y=5 (Total 250) | 1:5 (Probabilistic) | <span style="font-size:0.7em; display: block;">No repetition of real data needed in batch. Model improved but remained brittle to lighting changes.</span> | 45% |
| Exp. B | N=50, Y=20 (Total 1050) | 1:20 (Probabilistic) | <span style="font-size:0.7em; display: block;">Rare repetition of real data. Performance degraded vs Exp A, likely due to overfitting synthetic artifacts ("catastrophic forgetting" of real physics).</span> | 38% |
| Exp. C (Optimal) | N=50, Y=20 (Total 1050) | 1:2 (Deterministic) | <span style="font-size:0.7em; display: block;">Heavy repetition of real data (cyclical iterator). Real data constitutes 33% of every batch. Online jitter applied to real data repeats.</span> | **82%** |


Experiment	Strategy	Ratio (Real:Sim)	Result (N=10 trials)	Analysis
Exp. A	Probabilistic	1 : 5	45% Success	Model brittle to lighting changes.
Exp. B	Probabilistic	1 : 20	38% Success	**Catastrophic Forgetting.** The model overfit to synthetic artifacts, losing "real" physics grounding.
Exp. C	Deterministic	1 : 2	82% Success	**Optimal.** Heavy oversampling of real data acts as a regularization term, forcing the model to respect real-world dynamics.

** Final Config: ** Even though the data is 95% synthetic, **batch-level oversampling** keeps the model grounded in the 50 real "anchor" demonstrations.

<div style="display: flex; flex-direction: column; align-items: center; width: 100%;">
  <img src="/images/data_sampling.jpg" style="max-width: 100%; border-radius: 12px; margin-top:8px" alt="Receding-Horizon Control" />
</div>

(TODO - confidence intervals (Wilson), so these aren't just point estimates)

(/TEMP)


| Experiment | Dataset Composition | Batch Strategy (Real:Warped) | Notes | Est. Success Rate |
|---|---|---|---|---|
| Exp. A | N=50, Y=5 (Total 250) | 1:5 (Probabilistic) | <span style="font-size:0.7em; display: block;">No repetition of real data needed in batch. Model improved but remained brittle to lighting changes.</span> | 45% |
| Exp. B | N=50, Y=20 (Total 1050) | 1:20 (Probabilistic) | <span style="font-size:0.7em; display: block;">Rare repetition of real data. Performance degraded vs Exp A, likely due to overfitting synthetic artifacts ("catastrophic forgetting" of real physics).</span> | 38% |
| Exp. C (Optimal) | N=50, Y=20 (Total 1050) | 1:2 (Deterministic) | <span style="font-size:0.7em; display: block;">Heavy repetition of real data (cyclical iterator). Real data constitutes 33% of every batch. Online jitter applied to real data repeats.</span> | **82%** |

My final training configuration adopts the strategy in Experiment C, using a zipped dataloader to force a deterministic 1:2 Real-to-Synthetic batch ratio.

> **While the underlying dataset is 95% synthetic, oversampling ensures the model remains grounded in the 50 real "anchor" demonstrations.**

## Future Research Directions

- **Data Generation via Simulation:**
    
    Visual augmentation helps with robustness, but it doesn’t generate *new behaviors*. A future direction would be to explore different ratios and ratio schedules when sythetic data is *behaviorally diverse* rather than *visually diverse*. Next, I want synthetic trajectory generation in simulation via MimicGen [5]:

    - MimicGen reports generating **20k+ demonstrations from ~60 human demos** across tasks, with true novel behavior and domain randomization. The open question is how to allocate training mass between these.
        
        - **Open Research Question:** *When does augmented real data hit diminishing returns, and when does sim start to dominate?*

    - The implication for my setting: Keep a small set of real “anchors,” then scale behavior coverage massively in sim, and transfer back with a grounded batch strategy.

- **Cross-Embodiment & End-Effector (EE) Control**
    
    Currently, my model predicts **Joint Positions** directly. While effective for a specific robot, this locks the model to the SO-101's kinematics.

    The next step is to lift the action space to **End-Effector (EE) Poses**. I have already implemented a differentiable Jacobian-based inverse kinematics (IK) stack to map EE targets to joint angles. Upgrading this to include null-space projection [9] and integrating robust solvers like **Pinocchio** [10] or **MoveIt** [11] would allow the VLA to predict "what the hand should do" rather than "how the motors should move."

    - **Open Research Question:** *Can a VLA learn "Implicit Kinematics" from Cartesian supervision?*

        When we train on joint positions, the model implicitly learns limits (e.g., "I cannot rotate my wrist further"). When we switch to EE prediction, we hide these constraints from the model.

        The question is whether the VLA can learn to output only those EE poses that are kinematically feasible for the specific arm, or if it will constantly predict "valid" 3D coordinates that cause the IK solver to fail (singularities, self-collisions) because it lacks proprioceptive grounding.

<div style="display: flex; flex-direction: column; align-items: center; width: 100%;">
  <img src="/images/ee_vs_joint.jpg" style="max-width: 100%; border-radius: 12px; margin-top:8px" alt="Receding-Horizon Control" />
</div>

---

## References

1. https://www.physicalintelligence.company/blog/pi0

2. github.com/pravsels/augmented_datasocks

3. kornia.readthedocs.io/en/stable

4. https://arxiv.org/abs/2502.18049 Golden Ratio Mixing of Real and Synthetic Data for Stabilizing Generative Model Training (ResearchGate, 2025).

5. https://mimicgen.github.io/

9. https://faculty.sites.iastate.edu/jia/files/inline-files/robot%20control%20III.pdf

10. https://github.com/stack-of-tasks/pinocchio

11. https://moveit.ai/

12. https://github.com/TheRobotStudio/SO-ARM100

---

*# TODO(nbasile): Clean + upload Github repo with my code*


> This is conceptually similar to how **Pi₀** functions: a vision-language backbone encodes the scene and the task, while a separate “action expert” is trained to map those embeddings to high-frequency continuous actions. Their research was a large inspiration for this project. (cite pi0)

Concretely, I set out to answer:

**Can a single Vision-Language-Action (VLA) model, trained on real and synthetic teleoperation data, learn temporally consistent, instruction-conditioned robotic control comparable to emerging frontier systems like Gemini Robotics and GR-3—without relying on a hand-coded “action → fine control” interface?**

The rest of this post walks through that pipeline end-to-end: data collection, synthetic trajectory generation, VLA fine-tuning, and GRPO-based RL alignment in simulation: