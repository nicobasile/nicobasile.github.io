---
layout: post
title: Anchoring Simulation Physics To Reality - VLA Fine-Tuning 
categories: [embodied-ai, robotics]
author: Nicolas Basile
description: "When does simulation data stop helping and start erasing real physics? Rebalancing real, augmented, and simulated data when fine-tuning a VLA on my SO-101 raised pick-and-place success from 8% to 84%, without collecting more data. This post covers a six-way ablation of real / aug / sim batch composition."
hook: "<strong>When does simulation data stop helping and start erasing real physics?</strong> Rebalancing real, augmented, and simulated data when fine-tuning a VLA on my SO-101 raised pick-and-place success from <strong>8% to 84%</strong>, without collecting more data. This post covers a six-way ablation of real / aug / sim batch composition."
media_frame: matched
media_type: video
media_url: /videos/side-1-gif-edit.mp4
media_alt: "Exp. E side view: confident pick-and-place after golden-ratio mixing"
media_url_2: /videos/top-1-gif-edit.mp4
media_type_2: video
media_alt_2: "Exp. E top view: confident pick-and-place after golden-ratio mixing"
findings:
  - stat: "8% → 84%"
    label: Pick-and-place success
  - stat: "15 / 35 / 50"
    label: Real / aug / sim mix
  - stat: "N = 25"
    label: Eval rollouts
hero_compare:
  - label: "Optimal mix"
    caption: "<strong>Exp. E:</strong> confident pick-and-place after the 15/35/50 mix."
  - label: "Weaker mix"
    caption: "<strong>Exp. D:</strong> jittery, uncertain motion from a weaker mix."
hero:
  - url: /videos/side-1-gif-edit.mp4
    type: video
    alt: "Exp. E side view: confident pick-and-place after golden-ratio mixing"
  - url: /videos/top-1-gif-edit.mp4
    type: video
    alt: "Exp. E top view: confident pick-and-place after golden-ratio mixing"
  - url: /videos/side-2-gif-edit.mp4
    type: video
    alt: "Exp. D side view: jittery, uncertain motion from a weaker mixing run"
  - url: /videos/top-2-gif-edit.mp4
    type: video
    alt: "Exp. D top view: jittery, uncertain motion from a weaker mixing run"
---

## Introduction

How should real, augmented, and simulated trajectories be mixed when fine-tuning a Vision-Language-Action (VLA) model on a data-scarce manipulator?

In my previous work with discrete-action environments (Minecraft), agents operated under the assumption of "code-as-action" where the environment engine guaranteed physical consistency. That paradigm exposes two critical bottlenecks when applied to high-fidelity, real-world robotics:

1.  **Autoregressive Latency:** Token-by-token generation is computationally prohibitive for high-frequency control loops (e.g., <10Hz).
2.  **Action Fidelity:** Discrete tokens fail to capture the continuous, high-dimensional manifolds of joint-space trajectories.

I attempted to bridge this gap by fine-tuning a VLA on a 1,200-trajectory corpus (50 real, 150 augmented, 1,000 sim). However, instead of generalizing, the policy developed **"Synthetic Amnesia"**, catastrophic forgetting where simulation artifacts overwrite real-world physical priors.

This post documents the counter-intuitive reality of VLA fine-tuning: **unanchored synthetic data scales visual robustness but degrades physical fidelity.** I introduce a **Golden Ratio mixing strategy** that anchors the model in reality, improving success rates from **8% to 84%**.

## Architecture: The SmolVLA Approach

To address the latency and fidelity requirements of high-frequency control, SmolVLA decouples visual-semantic reasoning from motor-control execution:

1.  **Visual-Semantic Backbone:** **SmolVLM** (SigLIP-Vision + SmolLM2-Language) as a frozen feature extractor.
2.  **Intermediate Feature Tapping:** To bypass the autoregressive bottleneck, activations are extracted from intermediate transformer blocks (Layer 15 of 30). This assumes that the later layers, optimized for linguistic syntax and abstract reasoning, are redundant for low-level spatial grounding.
3.  **Flow-Matching Action Head:** These features are projected into a lightweight MLP-Transformer that predicts continuous joint trajectories via **Conditional Flow Matching (CFM)**.

<figure>
  {% include media.html url="/images/model_architecture.jpg" alt="SmolVLA Architecture" %}
  <figcaption>SmolVLA consumes multi-view RGB and language via SigLIP and SmolLM2, fusing them with a projected state token to predict continuous actions via a flow-matching head.</figcaption>
</figure>

### Model Selection: The Latency Bottleneck

Before settling on the final architecture, I benchmarked heavier contenders: **OpenVLA (7B)** <sup>[[10]](#ref-openvla)</sup> and quantized variants of **Pi-Zero** <sup>[[1]](#ref-pi0)</sup>.

Local GPU constraints (8GB of VRAM) initially struggled to fit these larger models at acceptable fidelity, so I offloaded inference to an A100 cloud instance on Lambda Labs and developed a **ROS 2 Humble** <sup>[[11]](#ref-ros2)</sup> server to handle communication between the local robot and the cloud backbone.

The network round-trip time (serialization + inference + deserialization) pushed the control loop below the 10Hz required for stability. The robot moved in "jerky," reactive bursts rather than smooth flows.

This constraint led me to **SmolVLA** <sup>[[12]](#ref-smolvla)</sup>. At ~450M parameters, it runs comfortably on my local hardware, eliminating network complexity entirely and allowing for a tight, high-frequency control loop.

> Although local inference is pragmatic for rapid experimentation, my future work will explore hybrid cloud-edge inference, leveraging methodologies from my previous inference optimization research.

### Action Policy: CFM & Receding Horizon Control (RHC)

To generate smooth actions without the computational cost of traditional diffusion policies, SmolVLA uses **Conditional Flow Matching (CFM)** <sup>[[13]](#ref-flow-matching)</sup>.

Unlike diffusion policies, which iteratively denoise an action over many inference steps, CFM learns a deterministic velocity field that transports random noise to a trajectory in a small number of integration steps. This captures the distribution of valid movements while remaining cheap enough for a high-frequency control loop.

The combination of the lightweight architecture and CFM enables **Receding Horizon Control (RHC)**:

* **Trajectory Prediction:** The action head predicts a continuous chunk of **H = 50** steps (**1.67s** of control).

* **Asynchronous Inference:** While the SO-101 arm executes the current chunk at **30Hz** (33.3ms/step), the model asynchronously infers the *next* chunk in the background.

* **Latency Masking:** Because of layer skipping, inference takes only ~100ms on an NVIDIA 3060 Ti, well within the 1.67s execution window. This allows the robot to move seamlessly without the "stutter" typical of synchronous VLM policies.

<figure>
  {% include media.html url="/images/action_chunks.jpg" alt="Receding-Horizon Control" %}
</figure>

A Note on Action Space: **Joint Space vs. Cartesian Poses**

Ideally, a robotic foundation model should predict **End-Effector (EE) Poses**. Predicting "where the hand should be" (Cartesian space) generalizes across embodiments better than predicting "how the motors should angle" (Joint space).

However, my early attempts at EE control hit a wall. Mapping EE predictions to joint angles requires an Inverse Kinematics (IK) solver. I experimented with and began building a differentiable Jacobian-based solver, but the VLA lacked proprioceptive grounding, frequently predicting "valid" Cartesian coordinates that were physically impossible for the arm to reach, forcing the IK solver into singularities or self-collisions.

Sholto Douglas noted in his work on orientation control that "Simplicity wins" - using Euler angles often beats elegant but fragile Quaternion math. Similarly, I found that predicting **Joint Positions** directly was significantly more robust for this iteration. While this locks the model to the SO-101's specific kinematics, it guarantees that every predicted action is physically executable. 

Developing a robust IK stack (integrating **Pinocchio** <sup>[[7]](#ref-pinocchio)</sup> with **Null-space Projection** <sup>[[6]](#ref-robot-control)</sup>) and translating the existing dataset into Cartesian space is a priority for the next iteration. This will allow the model to learn *"Implicit Kinematics"* while maintaining the cross-embodiment benefits of EE control.

## The Filament Problem: Manifold Support in Action Space

Flow matching models learn a vector field that transports a noise distribution to the target action distribution. With only 50 real-world trajectories, the data represents thin **"filaments"** floating in the high-dimensional action space of the policy.

*   **Extrapolation Failure:** If the robot is perturbed even slightly off these demonstrated filaments, the vector field becomes undefined or highly stochastic. This results in the "jittery" behavior (Exp. A) where the model fails to converge to a stable sink.
*   **The Simulation Solution:** Simulation data provides the necessary **topological support**. By filling the volume between real-world filaments, simulation allows the model to learn a continuous, smooth vector field across the entire workspace, providing the "recovery" behavior needed for robustness.

However, this volume comes at a cost. In naive mixing scenarios, the sheer quantity of simulation data (1000 vs. 50) drowned out the real-world signal. The model learned the *approximate* physics of the sim engines rather than the real robot's dynamics.

## Data: Mixing Real and Synthetic Trajectories

To address the "synthetic amnesia" problem, I adopted a **Tri-Partitioned Strategy** that treats augmented data as a distinct bridge domain.

The final dataset consisted of three distinct buckets:

- **Real Teleoperation Data (N=50):** Collected on the SO-101 instructor arm setup <sup>[[9]](#ref-so-arm101)</sup> to anchor the policy in true physical dynamics.
- **Augmented Real Data (N=150):** Generated via segmentation-based background replacement <sup>[[2]](#ref-datasocks)</sup> and **Kornia** <sup>[[3]](#ref-kornia)</sup> visual transforms. This forces the vision encoder to decouple kinematic signatures from visual framing.

<figure>
  {% include media.html url="/videos/augmented_synthetic_data.mp4" alt="Augmented synthetic training data" controls=true %}
  <figcaption>Example of the augmented real data. Credit: <a href="#ref-datasocks">DataSocks</a>.</figcaption>
</figure>

- **Simulation Data (N=1000):** Generated primarily in ManiSkill <sup>[[15]](#ref-maniskill)</sup> (SAPIEN <sup>[[16]](#ref-sapien)</sup>) with a small fraction from MuJoCo. I matched the visual assets (block color, table texture) to the physical lab setup (Digital Twin). This provides the "volume" required for stable CFM vector fields.

Following established best practices for VLA fine-tuning, the real data collection focused on diversity rather than sheer volume. I collected **N=50** high-quality "anchor" episodes, ensuring varied robot joint configurations by starting and ending trajectories in five distinct zones around the workspace.

> **Technical Note on Simulation:** I initially invested significant time porting the SO-100 to MuJoCo; however, I ultimately pivoted to ManiSkill (SAPIEN) for the bulk of data generation. ManiSkill's GPU-parallelized experts allowed for 10k trajectories in minutes, significantly accelerating the research pipeline.


## Training: The Golden Ratio Strategy

The hypothesis was that oversampling real data prevents drift, but excessive synthetic volume without explicit anchoring would cause catastrophic forgetting of real-world physics. I conducted an ablation study to find the inflection point where simulation supports the policy without overwriting reality.

### Ablation: Batch Composition Analysis

I drew inspiration from **"Golden Ratio Mixing"** <sup>[[4]](#ref-golden-ratio)</sup>, which posits that maintaining a high proportion of real-world data in minibatches is necessary to anchor the model, even when the underlying dataset is predominantly synthetic. I adopted this *principle*; not their literal φ ratio. The 15/35/50 optimum emerged from ablation, not from closed-form derivation.

For these benchmarks, **Success** is defined as grasping the object and placing it within 3cm of the target within 15 seconds, averaged over N=25 eval rollouts across held-out **initial conditions** (background, lighting, and object coordinates) not seen during teleop collection.

- Baseline (zero-shot SmolVLA): **~5%** success due to domain shift in my custom setup.
- Fine-tuned variants:

<div class="table-scroll">
  <table>
    <thead>
      <tr>
        <th>Exp.</th>
        <th>Batch composition<br>(% real / aug / sim)</th>
        <th>Success</th>
        <th>Diagnosis</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>A</td>
        <td><b>100 / 0 / 0</b><br>Pure real</td>
        <td>8%<br><span class="ci">CI 2–25%</span></td>
        <td><b>Filament collapse:</b> undefined vector field off-demonstration.</td>
      </tr>
      <tr>
        <td>B</td>
        <td><b>Uniform</b><br>Naive</td>
        <td>36%<br><span class="ci">CI 20–55%</span></td>
        <td><b>Synthetic amnesia:</b> overfit to sim physics (perfect friction).</td>
      </tr>
      <tr>
        <td>C</td>
        <td><b>35 / 15 / 50</b></td>
        <td>52%<br><span class="ci">CI 33–70%</span></td>
        <td><b>Rigid generalization:</b> stable but brittle to visual noise.</td>
      </tr>
      <tr>
        <td>D</td>
        <td><b>25 / 25 / 50</b></td>
        <td>64%<br><span class="ci">CI 45–80%</span></td>
        <td><b>Visual overfitting:</b> brittle to lighting/viewpoint changes.</td>
      </tr>
      <tr class="is-best">
        <td><b>E</b></td>
        <td><b>15 / 35 / 50</b><br>Optimal</td>
        <td><b>84%</b><br><span class="ci">CI 65–94%</span></td>
        <td><b>Optimal mix:</b> balanced anchoring vs. topological support.</td>
      </tr>
      <tr>
        <td>F</td>
        <td><b>10 / 40 / 50</b></td>
        <td>72%<br><span class="ci">CI 52–86%</span></td>
        <td><b>Weak anchoring:</b> visual robustness with physical drift.</td>
      </tr>
    </tbody>
  </table>
  <p class="table-note"><b>Table 1.</b> Batch composition vs. pick-and-place success. 95% Wilson CIs. N=25 eval rollouts.</p>
</div>

<figure>
  {% include media.html url="/images/result_chart.png" alt="Result Chart" %}
</figure>

### Fine-Tuning Strategy

Training was performed for 20,000 steps using a rented NVIDIA A100 (80GB). Starting from the `lerobot/smolvla_base` checkpoint, I froze the VLM backbone and fully fine-tuned the flow-matching Action Expert. This focused the gradients on the robot's control policy rather than overwriting the semantic knowledge of the vision encoder.

**Hyperparameters:** AdamW optimizer (β=[0.9, 0.95], weight decay=1e-10), learning rate 1e-4 with 1,000-step warmup, batch size 64. Training was stopped at 20k steps (~4 hours) following the recommended fine-tuning schedule.

### Why the 15 / 35 / 50 Split Works
Although the underlying dataset is 96% synthetic, the tri-partitioned strategy ensures that **50% of every gradient update** is derived from real-world kinematics (Real + Augmented). This prevents the model from overwriting the delicate friction and contact dynamics of the real arm with simulation artifacts.

By combining diverse teleoperation data, visual augmentations, and simulation data via Weighted Stratified Sampling <sup>[[14]](#ref-rt1)</sup>, I fine-tuned SmolVLA to execute smooth, confident motions. This approach successfully eliminated the jitter seen in earlier runs and enabled the model to generalize to new object locations.

## Limitations

This study is limited to a single manipulation task (pick-and-place) on a single robot morphology (SO-101). Training used a single seed, and evaluation did not include hardware randomization (camera extrinsics, payload, or friction). I leave further ablations on simulation fidelity, augmentation composition, and the scaling laws of real-world trajectory counts to future work. Extending these findings to multi-task settings and diverse embodiments remains an open research question.

## Future Research Directions

### Cross-Embodiment & End-Effector (EE) Control

The current policy predicts **Joint Positions** (see Architecture for the IK-failure rationale). The next iteration targets **End-Effector (EE) Poses** via Pinocchio <sup>[[7]](#ref-pinocchio)</sup> and null-space IK <sup>[[6]](#ref-robot-control)</sup>, so the VLA predicts "what the hand should do" rather than "how the motors should move."

*   **Open Research Question:** *Can a VLA learn "Implicit Kinematics" from Cartesian supervision?*
    Switching to EE prediction hides joint-limit constraints. Will the VLA learn to output only those poses that are solvable, or will it hallucinate "valid" 3D coordinates that force the IK solver into singularities or self-collisions?

<figure>
  {% include media.html url="/images/ee_vs_joint.jpg" alt="End-effector poses versus joint-space actions" %}
</figure>

### Scalability and Multi-Task Generalization

SmolVLA (450M parameters) is sufficient for 30Hz RHC on a 3060 Ti, but it likely lacks the world-knowledge priors required for zero-shot generalization to novel objects. Scaling to larger backbones (7B+) and training on more heterogeneous task data (e.g., OpenVLA <sup>[[10]](#ref-openvla)</sup>, Pi-Zero <sup>[[1]](#ref-pi0)</sup>) is the natural progression for this research.


---

## References

<ol class="ref-list">
  <li id="ref-pi0"><strong>Pi-Zero</strong> — Physical Intelligence. <a href="https://www.physicalintelligence.company/blog/pi0">physicalintelligence.company</a></li>
  <li id="ref-datasocks"><strong>Augmented Datasocks</strong> — Pravsels. <a href="https://github.com/pravsels/augmented_datasocks">github.com/pravsels/augmented_datasocks</a></li>
  <li id="ref-kornia"><strong>Kornia</strong> — Riba et al. <a href="https://kornia.readthedocs.io/en/stable">kornia.readthedocs.io</a></li>
  <li id="ref-golden-ratio"><strong>Golden Ratio Mixing</strong> — He et al. <a href="https://arxiv.org/abs/2502.18049">arXiv:2502.18049</a></li>
  <li id="ref-mimicgen"><strong>MimicGen</strong> — Mandlekar et al. <a href="https://mimicgen.github.io/">mimicgen.github.io</a></li>
  <li id="ref-robot-control"><strong>Robot Control III</strong> — Yan-Bin Jia. <a href="https://faculty.sites.iastate.edu/jia/files/inline-files/robot%20control%20III.pdf">Iowa State notes (PDF)</a></li>
  <li id="ref-pinocchio"><strong>Pinocchio</strong> — Carpentier et al. <a href="https://github.com/stack-of-tasks/pinocchio">github.com/stack-of-tasks/pinocchio</a></li>
  <li id="ref-moveit"><strong>MoveIt</strong> — Coleman et al. <a href="https://moveit.ai/">moveit.ai</a></li>
  <li id="ref-so-arm101"><strong>SO-ARM101</strong> — The Robot Studio. <a href="https://github.com/TheRobotStudio/SO-ARM100">github.com/TheRobotStudio/SO-ARM100</a></li>
  <li id="ref-openvla"><strong>OpenVLA</strong> — Kim et al. <a href="https://openvla.github.io/">openvla.github.io</a></li>
  <li id="ref-ros2"><strong>ROS 2 Humble</strong> — Open Source Robotics Foundation. <a href="https://docs.ros.org/en/humble">docs.ros.org/humble</a></li>
  <li id="ref-smolvla"><strong>SmolVLA</strong> — Hugging Face. <a href="https://huggingface.co/lerobot/smolvla_base">lerobot/smolvla_base</a>, <a href="https://huggingface.co/docs/lerobot/en/smolvla#finetune-smolvla-on-your-data">fine-tune docs</a></li>
  <li id="ref-flow-matching"><strong>Conditional Flow Matching</strong> — Lipman et al. <a href="https://arxiv.org/abs/2210.02747">arXiv:2210.02747</a></li>
  <li id="ref-rt1"><strong>RT-1</strong> — Brohan et al. <a href="https://arxiv.org/abs/2206.02077">arXiv:2206.02077</a></li>
  <li id="ref-maniskill"><strong>ManiSkill</strong> — Gu et al. <a href="https://github.com/haosulab/ManiSkill">github.com/haosulab/ManiSkill</a></li>
  <li id="ref-sapien"><strong>SAPIEN</strong> — Xiang et al. <a href="https://sapien.ucsd.edu/">sapien.ucsd.edu</a></li>
</ol>
