# customDQN
a personal project to create a customizable DQN (deep Q-learning) algorithm with dynamic environment prediction
built with pytorch and tested on the cartpole environment from gymnasium.

## features(?)
- pretrained models for easy loading and testing
- SO MANY ANNOTATIONS bc im silly and forget easily so this is a way for me to learn and remember!
- configs file for easy editing (most of) the important parameters
- dynamic implementation for model-based planning

## how 2 use
### install dependencies
``` bash
pip install -r requirements.txt
```

### train (optional)
change hyperparams in configs.py (also optional)
```bash
python train.py
```
may take several minutes to > an hour depending on number of episodes

### test
```bash
python test.py
```

outputs a `.mp4` file of the agent in the cartpole environment

## notes
- untested for environments other than cartpole so it may perchance maybe break bc i may have hardcoded some stuff that should be in configs but hopefully will be fixed soon

## future
- implement dynamic fully in training and not just testing

---

README last updated 06/22/2026