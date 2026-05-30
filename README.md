# 🚁 UZH-FPV Drone Vision Transformer + IMU Fusion

**New:** Added an actor–critic RL head on top of the MultiModalViT backbone to enable end-to-end visuomotor control for UZH-FPV drone racing

**Processing RPG Zurich's drone racing dataset for real-world navigation:**

This project builds a multimodal visual-inertial representation model on the UZH-FPV dataset using a transformer, with the goal of reusing that representation to initialize a reinforcement-learning policy for gate-conditioned drone racing control.

The intended RL setup operates under partial observability: observations consist of first-person greyscale images and a short history of IMU measurements, rather than the full simulator state. The multimodal transformer encoder produces a latent representation that will serve as input to the actor and critic.

## Related RL foundation work

To prepare for this project, I built a small set of RL benchmarks (Frozen Lake and Humanoid-v5) to practice discrete and continuous control, PPO training, and reward analysis.

Repository: [Reinforcement Learning Foundations](https://github.com/uttara-tech/reinforcement-learning-foundations)

## 🚀 Current Progress (Week 1/4)
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

