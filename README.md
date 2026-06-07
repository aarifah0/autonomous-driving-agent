

### Project title
**Autonomous Driving RL Agent**

### Student details
- **Student name:** Aarifah Loonat
- **Course code:** CMP4501
- **Project track:** Highway-driving RL using `highway-env`

### Evolution video

[![Evolution comparison](videos/eval/evolution.gif)](videos/eval/evolution.mp4)

A GIF preview is shown above for inline rendering on GitHub. Click the image to watch or download the full MP4:

- `videos/eval/evolution.mp4`




## 2. Methodology

### 2.1 Reward Function

The custom reward function is defined as:

\[
R_t = w_{speed} \, r_{speed} + w_{collision} \, r_{collision} + w_{lane} \, r_{lane}
\]

Where:

- \(r_{speed} = \text{clip}\left(\frac{v - v_{min}}{v_{max} - v_{min}},\,0,\,1\right)\)
- \(r_{collision} = \begin{cases} -1, & \text{if crash occurs} \\ +1, & \text{if no crash occurs} \end{cases}\)
- \(r_{lane} = 1 - \left(\frac{d}{w / 2}\right)^2\)

With configuration weights in `config.yaml`:

```yaml
wrapper:
  weight_speed: 1.5
  weight_collision: 20.0
  weight_lane: 0.8
```

#### Term explanations
- \(w_{speed} \, r_{speed}\): rewards forward velocity while normalizing speed across the target range.
- \(w_{collision} \, r_{collision}\): gives a strong negative penalty for crashes and a positive incentive for safe behavior.
- \(w_{lane} \, r_{lane}\): encourages the agent to stay centered in its lane and penalizes lateral deviation.

#### Why this reward function is suitable
This reward function balances efficiency and safety:
- the speed term encourages the agent to move quickly,
- the collision term strongly discourages crashes,
- the lane term supports stable lane keeping.

The structure is appropriate for `highway-env` because it uses kinematic state information rather than raw pixel input, and it avoids reward sparsity by providing continuous feedback each timestep.

### 2.2 The Model

This project uses **PPO (Proximal Policy Optimization)** with `stable-baselines3`.

#### Why PPO?
- PPO is stable and commonly used for control tasks.
- It handles discrete and continuous action spaces well.
- It balances sample efficiency and training reliability.

#### Key hyperparameters
- `learning_rate`: `0.0003`
- `gamma`: `0.95`
- `n_steps`: `1024`
- `batch_size`: `64`
- `n_epochs`: `10`
- `ent_coef`: `0.01`
- `total_timesteps`: `500000`
- `checkpoint_freq`: `50000`
- `eval_freq`: `10000`
- `eval_episodes`: `2`

#### Neural network architecture
The agent uses `MlpPolicy` with:
- two hidden layers for the policy network (`pi`): `[256, 256]`
- two hidden layers for the value network (`vf`): `[256, 256]`
- ReLU activations

This architecture is sufficient for the compact kinematic observation space and supports stable function approximation.

### 2.3 States and Actions

#### States
The state is a kinematic observation vector produced by `highway-env`:
- `presence`
- `x`, `y`
- `vx`, `vy`
- `heading`

The environment observes up to 5 nearby vehicles, sorted and normalized. This design keeps the input compact and avoids using expensive pixel-based observations.

#### Actions
The agent uses `DiscreteMetaAction`:
- discrete high-level maneuvers such as accelerate, decelerate, maintain speed, and lane changes
- this simplifies action selection while still allowing meaningful driving behavior

---

## 3. Training Analysis

### 3.1 Reward Graph

![Reward Graph](logs/training_reward.png)

![Episode Length](logs/training_length.png)



### 3.2 Commentary

The training process should demonstrate a clear improvement over time:
- the untrained agent begins with random or unsafe behavior,
- the half-trained agent should show partial control with occasional mistakes,
- the fully trained agent should reduce collisions and maintain better speed.

If training goes well, the reward curve should rise and then stabilize. Plateaus may appear when the agent has learned a stable policy, and noisy sections may indicate exploration or reward scaling issues.

The episode length curve is an additional indicator:
- short episode lengths usually reflect crashes or unsafe decisions,
- longer episode lengths show more stable, successful driving.

Because the reward function combines speed, collision avoidance, and lane discipline, successful training should increase average speed without sacrificing safety.

---

## 4. Challenges and Failures

### Training challenges

- **Policy instability early in training**: the agent initially learned unsafe actions such as unnecessary lane changes and sudden braking. This was mitigated by increasing `n_steps` and using a longer evaluation window to reduce noisy gradient updates.
- **Reward scaling and convergence**: the combined reward signal produced large fluctuations during training. I adjusted the reward weights and normalized speed/lane terms so the model received smoother training feedback.
- **Long training time and checkpointing**: training PPO for 500k timesteps required frequent checkpoints to capture the best model. We added checkpoint frequency and validation evaluation so the final policy is chosen from stable intermediate models.

### Resolution

- improved convergence by tuning PPO hyperparameters (`learning_rate`, `batch_size`, `n_epochs`) and using stable reward scaling
- reduced unsafe exploration with more conservative action sampling and a stronger collision penalty
- made training progress easier to monitor with evaluation episodes and regular checkpoints

---

## 5. Reproducibility

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the agent

```bash
python src/train.py
```

### Generate evaluation video

```bash
python evaluate.py --record --video_dir videos/eval
```

### Generate evolution video

```bash
python evolution_video.py --episodes 1
```

### Generate training plots

```bash
python visualize.py --log_dir logs --window 10
```

---

## Repository Structure

```
agent_driving/
├─ src/
│  ├─ agent.py
│  ├─ train.py
│  ├─ visualize.py
│  └─ visualize_tensorboard.py
├─ config.yaml
├─ environment.py
├─ evaluate.py
├─ evolution_video.py
├─ README.md
├─ .gitignore
├─ logs/
├─ checkpoints/
└─ videos/
```


