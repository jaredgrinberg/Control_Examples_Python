# Control Examples in Python with MuJoCo

A comprehensive collection of control algorithms implemented for robotic systems using MuJoCo physics simulation. Includes classical controllers (PID, LQR), advanced optimal control (MPC), and modern reinforcement learning (RL) approaches.

## 🎯 Overview

This project demonstrates and compares control algorithms across two robotic systems:

### Systems and Controllers

**CartPole (balancing):**
- LQR (Linear Quadratic Regulator)
- MPC (Model Predictive Control)
- RL (Reinforcement Learning with PPO)

**2-Link Robot Arm (torque tracking):**
- PID (Proportional-Integral-Derivative)
- LQR (Linear Quadratic Regulator)
- MPC (Model Predictive Control)
- RL (Reinforcement Learning with PPO/SAC)

### Controller Strengths

| Controller | Type | Strengths |
|------------|------|-----------|
| **PID** | Classical Feedback | Simple, effective, widely used |
| **LQR** | Optimal Control | Optimal gains for linear systems |
| **MPC** | Predictive Control | Handles constraints explicitly |
| **RL (PPO/SAC)** | Learning-based | Adapts to disturbances, learns from experience |

## 📦 Installation

### Prerequisites
- Python 3.8+
- Conda (recommended)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/jaredgrinberg/Control_Examples_Python.git
cd Control_Examples_Python
```

2. **Create conda environment:**
```bash
conda env create -f environment.yml
conda activate control
```

3. **Install additional packages:**
```bash
# For MPC (convex optimization)
pip install cvxpy

# For RL (reinforcement learning)
pip install stable-baselines3
pip install gymnasium
```

4. **Verify installation:**
```bash
python -c "import mujoco; print('MuJoCo OK')"
python -c "import cvxpy; print('cvxpy OK')"
python -c "import stable_baselines3; print('sb3 OK')"
```

## 🚀 Usage

### CartPole

Run different controllers on the cart-pole balancing task:

```bash
# LQR controller
python main_cartpole.py lqr

# MPC controller
python main_cartpole.py mpc

# RL controller (PPO)
python main_cartpole.py rl

# Compare all controllers
python main_cartpole.py compare
```

### 2-Link Robot Arm

Run different controllers on the joint torque tracking task:

```bash
# PID controller
python main_arm.py pid

# LQR controller
python main_arm.py lqr

# MPC controller (handles torque constraints)
python main_arm.py mpc

# RL controller with PPO (learns to handle disturbances)
python main_arm.py rl

# RL controller with SAC
python main_arm.py rl-sac
```

### Output

Each run generates:
- **MP4 video**: `simulations/[System]_[Controller]_[timestamp].mp4`
- **Performance plots**: Tracking error, control effort, etc.
- **Console output**: RMS error and control statistics

## 🤖 Reinforcement Learning

### Training vs. Loading

**First run (no model exists):**
```bash
python main_arm.py rl
# -> Trains PPO for 200k steps (~3-5 minutes)
# -> Saves model to trained_models/ppo_arm_torque.zip
# -> Runs simulation with trained policy
```

**Subsequent runs (model exists):**
```bash
python main_arm.py rl
# -> Auto-detects existing model
# -> Loads model instantly (no retraining!)
# -> Runs simulation
```

### Model Locations

**Final trained models:**
- CartPole: `trained_models/ppo_cartpole.zip`
- Arm (PPO): `trained_models/ppo_arm_torque.zip`
- Arm (SAC): `trained_models/sac_arm_torque.zip`

**Training checkpoints (auto-saved every 50k steps):**
- `trained_models/checkpoints/ppo_arm_torque_50000_steps.zip`
- `trained_models/checkpoints/ppo_arm_torque_100000_steps.zip`
- etc.

**Tensorboard logs:**
```bash
tensorboard --logdir trained_models/tensorboard
```

### Resuming from Checkpoint

If training crashes at step 175k, you can resume from the 150k checkpoint:

**Option 1: Copy checkpoint as final model**
```bash
# Copy checkpoint to final model location
cp trained_models/checkpoints/ppo_arm_torque_150000_steps.zip trained_models/ppo_arm_torque.zip

# Run (loads the checkpoint)
python main_arm.py rl
```

**Option 2: Continue training from checkpoint (advanced)**
```python
from controllers import TorqueRL

# Load checkpoint and train 50k more steps (150k + 50k = 200k total)
controller = TorqueRL(
    model_path='trained_models/checkpoints/ppo_arm_torque_150000_steps.zip',
    train_steps=50000,
    algorithm='PPO'
)
```

### Comparing PPO vs SAC

```bash
# Train both algorithms
python main_arm.py rl       # PPO
python main_arm.py rl-sac   # SAC

# Compare results in simulations/ folder
```

## 📊 Controller Comparison

### 2-Link Arm Performance

| Controller | RMS Error | Handles Constraints? | Handles Disturbances? | Training Time |
|------------|-----------|---------------------|----------------------|---------------|
| PID | ~0.02 Nm | Reactive clipping | Fixed gains | N/A |
| LQR | ~0.02 Nm | Reactive clipping | Fixed gains | N/A |
| MPC | ~0.02 Nm | Predictive | Fixed model | N/A |
| RL (PPO) | Variable | Learned | Adaptive | ~3-5 min |
| RL (SAC) | Variable | Learned | Adaptive | ~60-80 min |

**Key Insights:**
- **MPC excels** when desired torque exceeds actuator limits (±20 Nm) - predicts and optimally handles saturation
- **RL excels** when random disturbances are applied (0-10 N forces) - learns robust adaptive compensation
- **PID/LQR** are simple and effective for unconstrained, disturbance-free scenarios

## 📁 Project Structure

```
Control_Examples_Python/
├── controllers/              # All controller implementations
│   ├── CartpoleLQR.py       # CartPole LQR
│   ├── CartpoleMPC.py       # CartPole MPC
│   ├── CartpoleRL.py        # CartPole RL (PPO)
│   ├── JointTorquePID.py    # Arm PID
│   ├── JointTorqueLQR.py    # Arm LQR
│   ├── JointTorqueMPC.py    # Arm MPC
│   └── JointTorqueRL.py     # Arm RL (PPO/SAC)
├── environments/             # MuJoCo models and RL environments
│   ├── two_Link_arm.xml     # Robot arm MJCF model
│   └── arm_torque_env.py    # Custom Gym environment for arm
├── utils/                    # Simulation and plotting utilities
│   ├── simulation_arm.py
│   ├── simulation_cartpole.py
│   └── plotting.py
├── simulations/              # Output videos and plots
├── trained_models/           # RL models and checkpoints
│   ├── ppo_cartpole.zip
│   ├── ppo_arm_torque.zip
│   ├── sac_arm_torque.zip
│   ├── checkpoints/         # Training checkpoints
│   └── tensorboard/         # Training logs
├── main_arm.py              # Arm demo runner
├── main_cartpole.py         # CartPole demo runner
├── environment.yml          # Conda environment
└── README.md               # This file
```

## 🔬 Technical Details

### PID Controller
- Proportional-Integral-Derivative control
- Tuned gains: Kp=0.7, Ki=0.2, Kd=0.0
- Simple, effective baseline
- **Only implemented for arm** (not needed for CartPole)

### LQR Controller
- Linear Quadratic Regulator
- Optimal gains computed via algebraic Riccati equation
- Cost weights: Q (state), R (control)

### MPC Controller
- Model Predictive Control with 20-step horizon
- Quadratic programming via CVXPY (OSQP solver)
- Explicitly handles actuator constraints (±20 Nm torque limits for arm)
- Warm-start optimization for real-time performance

### RL Controller
- **Algorithms**: PPO (Proximal Policy Optimization) and SAC (Soft Actor-Critic)
- **Framework**: Stable-Baselines3
- **CartPole**: Uses built-in Gym environment for balancing
- **Arm**: Custom Gym environment with random disturbances (10% chance/step, 0-10 N forces)
- **Training**: 200k steps with checkpoints every 50k
- **Observation (arm)**: 22-dim state (desired/measured torque, error, integral, joint positions/velocities, history)

## Learning Resources

This project is good for:
- Understanding different control paradigms
- Comparing classical vs. modern control approaches
- Learning MuJoCo simulation
- Implementing custom Gym environments
- Training RL agents for robotics

## License

MIT License

## Contributing

Contributions welcome! Feel free to open issues or submit pull requests.

## Contact

For questions or feedback, please open an issue on GitHub.
