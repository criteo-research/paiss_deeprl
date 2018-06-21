# PAISS Deep Reinforcement Learning Practical Session by Criteo Research

## Environment

    1. Install Miniconda (cross-platform Python distribution): download & run installer from https://conda.io/miniconda.html
    1. Create environment for the session:
```
$ conda create --name paiss_deeprl python=3.6 Keras==2.2.0
```
    1. Activate environment:
```
$ source activate paiss_deeprl
```

    1. Gym RL toolkit install
```
$ git clone https://github.com/openai/gym.git
$ cd gym
$ pip3 install -e .[all]
```

    1. Verify everything is working
```
$ python3 -c "import keras, gym"
```
