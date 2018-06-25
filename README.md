# PAISS Deep Reinforcement Learning Practical Session by Criteo Research

## Environment

1. Install Miniconda (cross-platform Python distribution): 
    - download & run installer from [here](https://conda.io/miniconda.html)
1. Create environment for the session:
    ```
    $ conda create --name paiss_deeprl python=3.6 Keras==2.2.0 tensorflow=1.7.0 matplotlib=2.2.2
    ```
1. Activate environment:
    ```
    $ source activate paiss_deeprl
    ```
1. Install jupyter
    ```
    $ pip install jupyter
    ```
1. Gym RL toolkit dependencies
    - Linux (known to work on Ubuntu 16.04)
        ```
        $ sudo apt-get install cmake swig zlib1g-dev
        ```
    - Mac OSX
        ```
        $ brew install cmake swig
        ```
    - Windows
        - we recommend creating a Linux virtual machine
        - you could try [this answer on StackOverflom](https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows) at your own risk
        
1. Gym RL toolkit install
    ```
    $ git clone https://github.com/openai/gym.git
    $ cd gym
    $ pip install -e .[all]
    ```
1. Verify everything is working
    ```
    $ cd - && python3 -c "import keras, gym"
    ```
