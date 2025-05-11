# Quad-Ex

### Install all dependencies
```bash
pip install -r requirements.txt
```
*This will likely not work, but congrads if it did :)*
### Included models
The models availeble for training and visualizing consist of:\
Soft Actor-Critic (SAC),\
Augmented Random Search (ARS),\
Recurrent Proximal Policy Optimization (R-PPO),\
Deep Deterministic Policy Gradient (DDPG),\
Advantage Actor-Critic (A2C)

### Training the agents:
We have created multiple python files with a naming structure in the form: agent{model_type}.py.

There are a lot of arguments with which you can specify the action:\
"--model-path",     type=str, default="models/quad_ex.xml"\
"--total-timesteps",type=int, default=10_000\
"--eval-steps",     type=int, default=1_000\
"--training", type=bool, default=False\
"--use-preset", type=bool, default=False\
"--random-terrain", type=bool, default=False\

To use these, simply add them to your prompt.\
As an example, training the preset Ant-v5 environment from gymnasium is done like this:
```bash
python AGENTS/agent_<model_type>.py --training True --use-preset True
```
### Visualizing the trained models:
To visualize a model, it first needs to be trained.\
After training, simply remove the --training True from your prompt.\
Like this:
```bash
python AGENTS/agent_<model_type>.py --use-preset True
```
