
# Reinforcement Learning Agent

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A framework for developing and experimenting with reinforcement learning agents in various environments, featuring Q-learning implementation.

## Features
- Q-learning algorithm implementation
- Epsilon-greedy policy for exploration-exploitation
- Simple environment simulation
- Modular and extensible design

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import torch
from src.agent import QLearningAgent, Environment

state_size = 10
action_size = 4
agent = QLearningAgent(state_size, action_size)
env = Environment(num_states=state_size, num_actions=action_size)

# Train the agent
for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
    agent.decay_epsilon()

print("Training complete.")
```

## Project Structure

```
. \
├── src\
│   └── agent.py
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
