# 🚁 UZH-FPV Drone Vision Transformer + IMU Fusion

**Processing RPG Zurich's drone racing dataset for real-world navigation:**

This project builds a multimodal visual-inertial representation model on the UZH-FPV dataset using a transformer, with the goal of reusing that representation to initialize a reinforcement-learning policy for gate-conditioned drone racing control.

The intended RL setup operates under partial observability: observations consist of first-person greyscale images and a short history of IMU measurements, rather than the full simulator state. The multimodal transformer encoder produces a latent representation that will serve as input to the actor and critic.

## Related RL foundation work

To prepare for this project, I built a small set of RL benchmarks (Frozen Lake and Humanoid-v5) to practice discrete and continuous control, PPO training, and reward analysis.

Repository: [RL continuous control](https://github.com/uttara-tech/rl-continuous-control)

## 🚀 Current Progress

### Project status: on hold
This repository is currently a work in progress. Development is paused while I focus on my master’s thesis and may resume in the future.

###	RL implementation status
Reinforcement learning is not implemented in this repository. The actor–critic network architectures (actor and critic heads, multimodal backbone) are defined, but there is no simulation environment available, so no RL training loop or policy optimization is performed.

### Multimodal ViT implementation status
This code implements a multimodal Vision Transformer (image + IMU) trained for future pose prediction on the UZH-FPV dataset. The current setup feeds the full dataset to the model without an explicit train/validation split, so results reflect training error only and do not measure generalization.

✅ Frame extraction + timestamp CSV parsing  
✅ 640X480 -> 256×256 crops -> 16×16 patches = 256 image tokens   
✅ IMU + vision multimodal fusion (MultiModalViT)   
✅ Actor-Critic RL head on top of MultiModalViT encoder  

## 📈 Next Steps Roadmap
```mermaid
graph LR
  A[256×256 Preprocessing] --> B[Train/Val/Test Split]
  B --> C[Transformer Only Training]
  C --> D[Actor-Critic RL Head]
  D --> E[Policy Training in Simulator]

```

### 🧠 RL Architecture

The project now includes an actor–critic reinforcement learning head on top of the MultiModalViT encoder for vision-based drone control:

- `MultiModalViT` 
        
        Provides a 3D position trajectory regression head (x, y, z per step).
- `ActorHead` 

        a. MLP: 256 -> 256 -> 256 -> 4; 
        b. Outputs mean and log-std for 4D continuous actions (thrust, roll rate, pitch rate, yaw rate).
- `CriticHead` 
        
        a. MLP: 256 -> 256 -> 256 -> 1
        b. Outputs state value for PPO-style training.
- `MultiModalViTActorCritic` 

        Wraps encoder + actor + critic into a single RL-ready model

This turns the supervised trajectory prediction model into a reusable backbone for RL-based autonomous drone racing.


## Technical Notes

```markdown
- Input resolution:
    256×256 crops -> 16×16 patches = 256 image tokens
- IMU:
    Short temporal window of 6D accel+gyro -> 16 IMU tokens
- Tokens:
    256 (image) + 16 (IMU) + 1 [CLS] = 273 tokens per frame
- Dataset:
    UZH-FPV (RPG Zurich) -> 3D position trajectory regression (x, y, z)
- RL:
    Actor–critic head trained on top of pretrained encoder in a simulator
- Current status:
    Model architecture and RL heads are implemented; training objective and training loop will be added next
```

