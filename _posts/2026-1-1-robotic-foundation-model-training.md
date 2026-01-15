---
layout: post
title: Robotic Foundation Model Training - Anchoring Physics with Golden Ratios
categories: [embodied-ai, robotics]
author: Nicolas Basile
---

<figure class="image">
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
  <figcaption style="margin: 8px 0 8px 0;">
    <div style="color:#9aa0a6; font-size: 0.9em; margin-bottom: 5px;">
      <b>Top:</b> From the final fine-tuned model we see more confident movement (Exp. E)<br>
      <b>Bottom:</b> Jittery + uncertain movement from a less performant training run (Exp. D)
    </div>
    <b>Simulation data is a double-edged sword.</b> I explore the <b>"synthetic amnesia"</b> phenomenon in VLA fine-tuning and the <b>Golden Ratio mixing</b> strategy that anchored my policy in real-world physics, lifting success rates from <b>8% to 84%</b>.
  </figcaption>
</figure>
<style>
  /* The main grid container */
  .gif-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    width: 100%;
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
    margin: 0 !important; /* Override any global image margins */
  }
</style>

## Introduction

In my previous work with Minecraft agents, actions were discrete. If the VLM output `move_to_coords(X, Y, Z)`, the game engine **guaranteed** the physics. The agent never had to calculate torque or friction. However, that "code-as-action" paradigm exposes two critical bottlenecks when applied to real-world robotics:

1. **Autoregressive Latency:** Generating tokens is too slow for dynamic control loops (e.g., <10Hz).

2. **Action Fidelity:** Discrete text commands (e.g., "move forward") do not cleanly generalize to continuous, high-dimensional joint torques.

I attempted to bridge this gap by flooding a VLA model with massive amounts of synthetic data (both visually augmented and MuJoCo-simulated). It backfired. Instead of generalizing, the robot developed "synthetic amnesia," overwriting real-world physics with simulation artifacts. 

This post documents the counter-intuitive reality of VLA fine-tuning: **too much synthetic data will likely degrade performance unless you explicitly anchor the model in reality**.

---

## Table of Contents
- [Introduction](#introduction)
- [Architecture: The SmolVLA Approach](#architecture-the-smolvla-approach)
  - [Model Selection: The Latency Bottleneck](#model-selection-the-latency-bottleneck)
  - [Action Policy: Flow Matching & Receding Horizon Control](#action-policy-flow-matching--receding-horizon-control)
- [The Filament Problem](#the-filament-problem)
- [Data: Mixing Real and Synthetic Trajectories](#data-mixing-real-and-synthetic-trajectories)
- [Training: The Golden Ratio](#training-the-golden-ratio)
  - [Ablation: Real-Augmented-Simulation batch composition](#ablation-real-augmented-simulation-batch-composition)
  - [Fine-Tuning Strategy](#fine-tuning-strategy)
- [Future Research Directions](#future-research-directions)
  - [Cross-Embodiment & End-Effector (EE) Control](#cross-embodiment-end-effector-ee-control)
  - [Training on larger models with more diverse task data](#training-on-larger-models-with-more-diverse-task-data)
- [References](#references)

---

## Architecture: The SmolVLA Approach

To solve the fidelity issues of text-based planning, SmolVLA fundamentally decouples high-level reasoning from low-level control.

1.  **Vision-Language Backbone:** The model uses **SmolVLM** (based on SigLIP and SmolLM2) to process visual observations and textual instructions.
2.  **Layer Skipping:** Crucially, SmolVLA does not wait for the full VLM to finish processing. Instead, it extracts rich semantic features from intermediate layers (typically layer 15 of 30) of the language backbone. This "short-circuit" mechanism significantly reduces computational overhead, as the later layers of an LLM (often responsible for complex verbal reasoning) are redundant for immediate motor control.
3.  **The Action Expert:** These intermediate features are passed to a dedicated action head - a lightweight transformer trained to predict continuous actions.

<div style="display: flex; flex-direction: column; align-items: center; width: 100%;">
  <img src="/images/model_architecture.jpg" style="max-width: 100%; border-radius: 12px; margin-top:8px" alt="SmolVLA Architecture" />
</div>

<div style="margin-top:8px; color:#9aa0a6; font-size: 0.9em;">
  SmolVLA consumes multi-view RGB and language instructions via SigLIP and SmolLM2 encoders, fusing them with a projected state token to predict continuous actions via a flow-matching transformer head.
</div>

### Model Selection: The Latency Bottleneck

Before settling on my final architecture, I benchmarked two heavier contenders: **OpenVLA (7B)** <sup>[[10]](#ref-openvla)</sup> and a quantized version of **Pi-Zero** <sup>[[1]](#ref-pi0)</sup>.

Since my local GPU (8GB of VRAM) struggled to fit & run these larger models at an acceptable fidelity, I initially offloaded inference to an A100 cloud instance on Lambda Labs. I built a custom **ROS 2 Humble** <sup>[[11]](#ref-ros2)</sup> server to handle the communication between the local robot and the cloud brain.

The network round-trip time (serializing multiple images, waiting for inference, and deserializing actions) pushed the control loop below the 10Hz required for stability. The robot moved in jerky, reactive bursts rather than smooth flows.

This constraint led me to **SmolVLA** <sup>[[12]](#ref-smolvla)</sup>. With its ~450m parameters, it runs comfortably on my hardware, eliminating network complexity entirely and allowing for a tight, high-frequency control loop. 

> Although local inference is pragmatic for experimentation & research, I'll certainly be looking at cloud inference again in the future, leveraging some of the methodologies from my inference optimization research at VMware.

### Action Policy: Flow Matching & Receding Horizon Control

To generate smooth actions without the computational cost of traditional diffusion policies, SmolVLA uses **Conditional Flow Matching** <sup>[[13]](#ref-flow-matching)</sup>. 

Unlike diffusion policies, which iteratively remove noise to reveal an action (often requiring 10-100 slow inference steps), flow matching learns a deterministic "velocity field" that transforms random noise into a precise trajectory. This captures the complex distribution of valid movements (handling uncertainty) while allowing for extremely fast sampling (often 10x faster than diffusion).

The combination of the lightweight architecture and flow matching enables **Receding Horizon Control**:

* **Prediction:** The action head predicts a continuous chunk of **H = 50** steps (**1.67s** of control).

* **Asynchronous Inference:** While the SO-101 arm executes the current chunk at **30Hz** (33.3ms/step), the model asynchronously infers the *next* chunk in the background.

* **Latency Masking:** Because of layer skipping, inference takes only ~100ms on my NVIDIA 3060 Ti, well within the 1.67s execution window. This allows the robot to move seamlessly without the "stutter" typical of synchronous VLM policies.

<div style="display: flex; flex-direction: column; align-items: center; width: 100%; border: 1px solid gray; border-radius: 3px">
  <img src="/images/action_chunks.jpg" style="max-width: 100%; border-radius: 12px; margin-top:8px" alt="Receding-Horizon Control" />
</div>

A Note on Action Space: **Simplicity Wins**

Ideally, a robotic foundation model should predict **End-Effector (EE) Poses**. Predicting "where the hand should be" (Cartesian space) generalizes across embodiments better than predicting "how the motors should angle" (Joint space).

However, my early attempts at EE control hit a wall. Mapping EE predictions to joint angles requires an Inverse Kinematics (IK) solver. I experimented with and began building a differentiable Jacobian-based solver, but the VLA lacked proprioceptive grounding, frequently predicting "valid" Cartesian coordinates that were physically impossible for the arm to reach, forcing the IK solver into singularities or self-collisions.

Sholto Douglas noted in his work on orientation control that "Simplicity wins" - using Euler angles often beats elegant but fragile Quaternion math. Similarly, I found that predicting **Joint Positions** directly was significantly more robust for this iteration. While this locks the model to the SO-101's specific kinematics, it guarantees that every predicted action is physically executable. 

Developing/ironing out a robust IK stack (likely integrating Pinocchio <sup>[[7]](#ref-pinocchio)</sup> with null-space projection <sup>[[6]](#ref-robot-control)</sup>) and translating the existing dataset into Cartesian space is a priority for the next iteration. This will allow the model to learn *"Implicit Kinematics"* while maintaining the cross-embodiment benefits of EE control.

## The Filament Problem

Flow matching models learn a vector field that transports noise to a valid action. With only 50 real trajectories, the data represents thin "filaments" floating in the high-dimensional configuration space of the robot. I initially thought that adding augmented synthetic data would fill in the gaps and make the model more robust. It did - but only for visual diversity. The model still struggled to generalize to new object locations and recover from perturbations.

• **The Amnesia:** If the robot is perturbed just millimeters off a real trajectory, the vector field is undefined. The model hallucinates an action, leading to the erratic jitter I observed in early experiments.

• **The Fix:** The simulation data fills the empty space between these filaments. Even if the physics aren't perfect, the simulation provides the topological support necessary to define "smooth motion" across the entire workspace.

However, this volume comes at a cost. When I naively mixed the data, the sheer quantity of simulation data (1000 vs. 50) drowned out the real-world signal. The model learned the *approximate* physics of MuJoCo rather than the *exact* friction and damping of the real arm.

## Data: Mixing Real and Synthetic Trajectories

To address the "synthetic amnesia" problem, I adopted a **Tri-Partitioned Strategy** that treats augmented data as a distinct "third domain" - data which bridges the kinematic ground truth of reality with the visual diversity of simulation.

My final dataset consisted of three distinct buckets:

- **Real teleoperation data (N=50)** collected on my SO-101 instructor arm setup <sup>[[9]](#ref-so-arm101)</sup>.
  - <u>Role:</u> Prevents artifacts and anchors the policy to the true test distribution.
- **Augmented real data (N=150)** generated via:
    - **Segmentation-based augmentation** <sup>[[2]](#ref-datasocks)</sup>
        - Segments robot arm and target objects
        - Applies background replacements while maintaining foreground elements
    - **Kornia visual augments** <sup>[[3]](#ref-kornia)</sup> for color jittering, contrast adjustments, and cropping, applied consistently across a trajectory
    - <u>Role:</u> Forces the vision encoder to ignore framing nuances while preserving the exact kinematic signature of the real robot.
- **Simulation data (N=1000)** generated in MuJoCo using the SO-101 arm URDF and I matched the visual assets (block color, table texture) to the physical lab setup (Digital Twin)
  - <u>Role:</u> Provides the "volume" required for the flow matching head to learn a stable vector field.

<div style="display: flex; flex-direction: column; align-items: center; width: 100%;">
  <img src="/videos/augmented_synthetic_data.gif" style="max-width: 100%; border-radius: 12px; margin-top:8px" alt="Augmented Synthetic Data" />
  
  <div style="margin-top:8px; color:#9aa0a6; font-size: 0.9em;">
    Example of the augmented real data. Credit to <a href="#ref-datasocks" rel="noopener noreferrer" style="color:#9aa0a6; text-decoration: underline;">
      DataSocks.
    </a>
  </div>
</div>

Following established best practices for VLA fine-tuning, the real data collection focused on diversity rather than sheer volume. I collected **N=50** high-quality "anchor" episodes, ensuring varied robot joint configurations by starting and ending trajectories in five distinct zones around the workspace.

## Training: The Golden Ratio

I expected 'more augmented + simulation data = better', but early runs showed the opposite: naive sampling from a 96% synthetic pool made the policy brittle. 

### The Experimental Loop

*   **Hypothesis:** Oversampling real data prevents drift, but excessive synthetic volume without anchoring causes catastrophic forgetting of real-world physics.
*   **Test:** Six variants (A-F) testing the transition from pure real data to heavy synthetic mixing.
*   **Results:**
    *   **Exp A (Pure Real):** 8% success (Filament collapse; zero generalization).
    *   **Exp B (Naive Mix):** 36% success (Synthetic amnesia; model "forgets" real-world friction).
    *   **Exp C (Tri-Partitioned):** 64% success (Stable physics, but brittle to visual shifts).
    *   **Exp D (Golden Ratio):** 84% success (Optimal; anchors physics while leveraging sim volume).
    *   **Exp E (Weak Anchoring):** 72% success (High visual robustness, but physics drift).
    *   **Exp F (Rigid Generalization):** 52% success (Stable physics, but visual fragility).
*   **Conclusion:** Strategy C works mechanistically by forcing the gradient updates to respect the kinematic ground truth of the real arm, even when the visual input is heavily distorted.

### Ablation: Real-Augmented-Simulation batch composition

Early runs confirmed that naively flooding the model with synthetic data degrades the policy. The sheer volume of simulation data drowns out the delicate signals of real-world friction and contact dynamics ("synthetic amnesia").

To fix this, I drew inspiration from "Golden Ratio Mixing" <sup>[[4]](#ref-golden-ratio)</sup>, which posits that maintaining a high proportion of real data in minibatches is necessary to anchor the model in reality (even if it requires heavy oversampling), while allowing synthetic data to smooth the decision manifold.

I conducted an ablation study over several mixing strategies to find the inflection point where simulation supports the policy without overwriting reality. For these benchmarks, **Success** is defined as grasping the object and placing it within 3cm of the target within 15 seconds, averaged over N=25 eval rollouts across held-out **background**, **lighting**, **pick/place coordinate**, and **camera viewpoint** conditions.

- Baseline (zero-shot SmolVLA): **~5%** success due to domain shift in my custom setup.
- Fine-tuned variants:

| <span style="font-size: 0.8em;">Experiment</span> | <span style="font-size: 0.8em;">Data Mix (Real : Aug : Sim)</span> | <span style="font-size: 0.8em;">Batch Sampling Strategy</span> | <span style="font-size: 0.8em;">Success Rate</span> | <span style="font-size: 0.8em;">Diagnosis</span> |
|---|---|---|---|---|
| <span style="font-size: 0.8em;">Exp. A</span> | <span style="font-size: 0.8em;">50 : 0 : 0</span> | <span style="font-size: 0.8em;">**Pure Real** - Only trained on the 50 teleop demos.</span> | <span style="font-size: 0.8em;">8%</span> <br><span style="font-size:0.8em; color:grey;">(CI: 2% - 25%)</span> | <span style="font-size: 0.8em;">**Filament Collapse:** The flow matching vector field was undefined just millimeters off the demos.</span> |
| <span style="font-size: 0.8em;">Exp. B</span> | <span style="font-size: 0.8em;">50 : 150 : 1000</span> | <span style="font-size: 0.8em;">**Naive Mixing** - Sampled uniformly from the total pool</span> | <span style="font-size: 0.8em;">36%</span><br><span style="font-size:0.8em; color:grey;">(CI: 20% - 55%)</span> | <span style="font-size: 0.8em;">**Synthetic Amnesia:** The model learned "MuJoCo physics" (perfect friction), causing it to slip and fail in the real world.</span> |
| <span style="font-size: 0.8em;">Exp. C</span> | <span style="font-size: 0.8em;">50 : 150 : 1000</span> | <span style="font-size: 0.8em;">**Forced Ratio:** 35% Real / 15% Aug / 50% Sim</span> | <span style="font-size: 0.8em;">52%</span><br><span style="font-size:0.8em; color:grey;">(CI: 33% - 70%)</span> | <span style="font-size: 0.8em;">**Rigid Generalization:** Stable physics but failed to generalize to visual noise, similar to a robust Exp. A.</span> |
| <span style="font-size: 0.8em;">Exp. D</span> | <span style="font-size: 0.8em;">50 : 150 : 1000</span> | <span style="font-size: 0.8em;">**Forced Ratio:** 25% Real / 25% Aug / 50% Sim</span> | <span style="font-size: 0.8em;">64%</span><br><span style="font-size:0.8em; color:grey;">(CI: 45% - 80%)</span> | <span style="font-size: 0.8em;">**Visual Overfitting:** Physics were stable, but the model became brittle to lighting changes. The raw pixel data was over-represented.</span> |
| <span style="font-size: 0.8em;">**Exp. E (Optimal)**</span> | <span style="font-size: 0.8em;">50 : 150 : 1000</span> | <span style="font-size: 0.8em;">**Forced Ratio: 15% Real / 35% Aug / 50% Sim**</span> | <span style="font-size: 0.8em;"><b>84%</b></span><br><span style="font-size:0.8em; color:grey;">(CI: 65% - 94%)</span> | <span style="font-size: 0.8em;">**Golden Ratio:** Real data is heavily oversampled to anchor the physics, while sim provides topological support.</span> |
| <span style="font-size: 0.8em;">Exp. F</span> | <span style="font-size: 0.8em;">50 : 150 : 1000</span> | <span style="font-size: 0.8em;">**Forced Ratio:** 10% Real / 40% Aug / 50% Sim</span> | <span style="font-size: 0.8em;">72%</span><br><span style="font-size:0.8em; color:grey;">(CI: 52% - 86%)</span> | <span style="font-size: 0.8em;">**Weak Anchoring:** High visual robustness, but the physics anchor was too weak for complex contact phases.</span> |


<div style="margin-top:8px; color:#9aa0a6; font-size: 0.9em;">
  <b>Table 1:</b> Impact of Batch Composition on Success Rate Evaluated on N=25 trials per model. 95% Confidence Intervals (Wilson Score).
</div>

<div style="display: flex; flex-direction: column; align-items: center; width: 100%;">
  <img src="/images/result_chart.png" style="max-width: 100%; border-radius: 12px; margin-top:8px" alt="Result Chart" />
</div>

### Fine-Tuning Strategy

Training was performed for 20,000 steps (batch size 64) using a rented NVIDIA A100 (80GB) on Lambda Labs. Starting from the `lerobot/smolvla_base` checkpoint and following the default LeRobot recipe <sup>[[12]](#ref-smolvla)</sup>, I froze the VLM backbone and fully fine-tuned the flow-matching Action Expert. This focused the gradients on the robot's control policy rather than overwriting the semantic knowledge of the vision encoder.

### Why the 15 / 35 / 50 Split Works
Although the underlying dataset is 96% synthetic, the tri-partitioned strategy ensures that **50% of every gradient update** is derived from real-world kinematics (Real + Augmented). This prevents the model from overwriting the delicate friction and contact dynamics of the real arm with simulation artifacts.

By combining diverse teleoperation data, visual augmentations, and MuJoCo simulation data via Weighted Stratified Sampling <sup>[[14]](#ref-rt1)</sup>, I fine-tuned SmolVLA to execute smooth, confident motions. This approach successfully eliminated the jitter seen in earlier runs and enabled the model to generalize to new object locations.

## Future Research Directions

### Cross-Embodiment & End-Effector (EE) Control
    
Currently, my model predicts **Joint Positions** directly. While effective for a specific robot, this locks the model to the SO-101's kinematics.

The next step is to lift the action space to **End-Effector (EE) Poses**. I have already implemented a differentiable Jacobian-based inverse kinematics (IK) stack to map EE targets to joint angles. Upgrading this to include null-space projection <sup>[[6]](#ref-robot-control)</sup> and integrating robust solvers like **Pinocchio** <sup>[[7]](#ref-pinocchio)</sup> or **MoveIt** <sup>[[8]](#ref-moveit)</sup> would allow the VLA to predict "what the hand should do" rather than "how the motors should move."

- **Open Research Question:** *Can a VLA learn "Implicit Kinematics" from Cartesian supervision?*

    If we switch to EE prediction, we hide these constraints. Will the VLA learn to output only those poses that are solvable, or will it hallucinate "valid" 3D coordinates that force the IK solver into singularities or self-collisions?

<div style="display: flex; flex-direction: column; align-items: center; width: 100%;">
  <img src="/images/ee_vs_joint.jpg" style="max-width: 100%; border-radius: 12px; margin-top:8px" alt="Receding-Horizon Control" />
</div>

### Training on larger models with more diverse task data

SmolVLA (450M parameters) is impressively efficient, but it likely lacks the world knowledge required for true zero-shot generalization. Scaling to larger models (7B+) and training on more diverse task data (e.g., OpenVLA <sup>[[10]](#ref-openvla)</sup>, Pi-Zero <sup>[[1]](#ref-pi0)</sup>) would be a natural next step.


---

## References

1. <a id="ref-pi0"></a> **Pi-Zero** — Physical Intelligence: A Generalist Robot Policy. https://www.physicalintelligence.company/blog/pi0

2. <a id="ref-datasocks"></a> **Augmented Datasocks** — Pravsels, Augmented Datasocks: Open-Source Teleoperation Hardware. https://github.com/pravsels/augmented_datasocks

3. <a id="ref-kornia"></a> **Kornia** — Riba et al., Kornia: an Open Source Differentiable Computer Vision Library for PyTorch. https://kornia.readthedocs.io/en/stable

4. <a id="ref-golden-ratio"></a> **Golden Ratio Mixing** — He et al., Golden Ratio Weighting Prevents Model Collapse
. https://arxiv.org/abs/2502.18049

5. <a id="ref-mimicgen"></a> **MimicGen** — Mandlekar et al., MimicGen: A Data Generation System for Scalable Robot Learning. https://mimicgen.github.io/

6. <a id="ref-robot-control"></a> **Robot Control III** — Yan-Bin Jia, Robot Control III: Feedforward Control and Computed Torque (Iowa State University). https://faculty.sites.iastate.edu/jia/files/inline-files/robot%20control%20III.pdf

7. <a id="ref-pinocchio"></a> **Pinocchio** — Carpentier et al., Pinocchio: A fast and efficient library for rigid body dynamics algorithms. https://github.com/stack-of-tasks/pinocchio

8. <a id="ref-moveit"></a> **MoveIt** — Coleman et al., MoveIt: The Motion Planning Framework. https://moveit.ai/

9. <a id="ref-so-arm101"></a> **SO-ARM101** — The Robot Studio, SO-ARM101: Open Source 6-Axis Robot Arm. https://github.com/TheRobotStudio/SO-ARM100

10. <a id="ref-openvla"></a> **OpenVLA** — Kim et al., OpenVLA: An Open-Source Vision-Language-Action Model. https://openvla.github.io/

11. <a id="ref-ros2"></a> **ROS 2 Humble** — Open Source Robotics Foundation, Robot Operating System 2 Humble Hawksbill. https://docs.ros.org/en/humble

12. <a id="ref-smolvla"></a> **SmolVLA** — Hugging Face, SmolVLA Base Model. https://huggingface.co/lerobot/smolvla_base, https://huggingface.co/docs/lerobot/en/smolvla#finetune-smolvla-on-your-data

13. <a id="ref-flow-matching"></a> **Conditional Flow Matching** — Lipman et al., Flow Matching for Generative Modeling. https://arxiv.org/abs/2210.02747

14. <a id="ref-rt1"></a> **RT-1** — Brohan et al., RT-1: Robotics Transformer for Real-World Control at Scale. https://arxiv.org/abs/2206.02077


---

*# Soon™: Clean + upload Github repo with my code*
