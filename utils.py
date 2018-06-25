from collections import defaultdict, deque

import numpy as np
from matplotlib import pyplot as plt

import gym
import gym.spaces


class RLEnvironment(object):
    """
    Environment in which to play an agent.
    """

    def __init__(self, envname='CartPole', target_perf=190, target_window=100):
        self.env = gym.make("{}-v0".format(envname))
        self.state_size = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.target_perf = target_perf
        self.target_window = target_window

    def run(self, agent, episodes=100, print_delay=10, display_policy=False, seed=None):
        """
        Run the agent.

        Pseudo-code:
        ```
            for i in 1..episodes {

                start new episode

                while episode not finished {

                    ask agent to take action based on current state
                    action is resolved by environment, returning reward and new state

                    <opportunity to feedback agent with (state, reward, new state)>
                    <opportunity to update agent parameters>

                    if new state means agent failed {
                        terminate episode
                    }

                }

                <opportunity to update agent parameters again>

                if last episodes show enough reward {
                    declare task solved
                }

            }

        ```

        :param agent: must implement `get_action(state)`
                      optionally can implement
                        `train_model(action, state, next_state, reward, done)`
                        `update_epsilon()`
                        `update_target_model()`
        :param episodes: nber of episodes to run
        :param print_delay: will print reward every `print_delay` episodes
        :param display_policy: display animation of policy
        :return:
        """
        agent.traces = Trace()
        try:
            last_rewards = deque(maxlen=self.target_window)
            for i in range(1, episodes+1):
                if seed is not None:
                    self.env.seed(seed)
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                total_reward = 0
                while True:
                    if display_policy:
                        plt.imshow(self.env.render('rgb_array'))
                    action = agent.get_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    total_reward += reward
                    if hasattr(agent, 'train_model'):
                        agent.train_model(action, state, next_state, reward, done)
                    if hasattr(agent, 'update_epsilon'):
                        agent.update_epsilon()
                    state = next_state
                    if done:
                        last_rewards.append(total_reward)
                        if hasattr(agent, 'update_target_model'):
                            agent.update_target_model()
                        if (i % print_delay) == 0:
                            print("Episode {}, Total reward {}".format(
                                i, total_reward)
                            )
                        break
                if len(last_rewards) >= self.target_window and np.mean(last_rewards) >= self.target_perf:
                    print("*" * 80)
                    print("CONGRATS !!! YOU JUST SOLVED CARTPOLE !!!")
                    print("*" * 80)
                    print("now you can try with envname='MsPacman-ram' ;)")
                    break
        finally:
            print("Average Total Reward of last {} episodes: {}".format(
                len(last_rewards), np.mean(last_rewards))
            )
            self.env.close()


class Trace:

    def __init__(self):
        self.data = defaultdict(list)

    def __iadd__(self, tup: tuple):
        key, record = tup
        self.data[key] += [record]
        return self

    def __getitem__(self, key: str):
        return self.data[key]


class RLDebugger:

    def __init__(self):
        self.traces = Trace()

    def record(self, action, state, target, target_val, loss, reward):
        self.traces += ('action', action)
        if len(state.shape) > 1 and state.shape[1] > 1:
            state = state[0]
        assert state.shape[0] == 4, state
        self.traces += ('state-x', state[0])
        self.traces += ('state-x-deriv', state[1])
        self.traces += ('state-theta', state[2])
        self.traces += ('state-theta-deriv', state[3])
        try:
            if target is not None:
                if len(target.shape) > 1 and target.shape[1] > 1:
                    target = target[0]
                if len(target_val.shape) > 1 and target_val.shape[1] > 1:
                    target_val = target_val[0]
                assert target.shape[0] == 2, state
                self.traces += ('value_estimation', target[action])
                self.traces += ('value_prediction', target_val[action])
                self.traces += ('bellman_residual', target[action] - target_val[action] - reward)
        except IndexError as IE:
            print(target, action)
            print(target_val, action)
            raise IE
        self.traces += ('model_loss', loss)

    @staticmethod
    def moving_average(iterable, n=10):
        d = deque(maxlen=n)
        for i in iterable:
            d.append(i)
            if len(d) == n:
                yield sum(d)/n

    @staticmethod
    def get_ax(**kwargs):
        fig, ax = plt.subplots(**kwargs)
        return ax

    def _plot(self, metric, color='k', ax=None, ma=True, **kwargs):
        ax = ax if ax is not None else self.get_ax()
        x = self.traces[metric]
        l = len(x)
        if l > 100000:
            resample_ = int(l / 10000)
            x = x[::resample_]
        if ma:
            n_ma = 1+int(len(x)/100)
            x = [_ for _ in self.moving_average(x, n=n_ma)]
        ax.plot(x, color=color, ls='', marker='.', markersize=4, alpha=.5, **kwargs)
        ax.set_title(metric)
        ax.set_xlabel('time')

    def plot_bellman_residual(self, **kwargs):
        self._plot('bellman_residual', color='orange', **kwargs)

    def plot_loss(self, **kwargs):
        self._plot('model_loss', color='red', **kwargs)

    def plot_actions(self, **kwargs):
        self._plot('action', ma=False, **kwargs)

    def plot_state(self, ax=None):
        ax = ax if ax is not None else self.get_ax()
        ax.set_title('state')
        ax.set_xlabel('time')
        ax.set_ylabel('cart position ($x$)')
        ax.plot(self.traces['state-x'], color='blue', label='$x$', ls='', marker='.', markersize=4, alpha=.5)
        ax.yaxis.label.set_color('blue')
        ax.tick_params(axis='y', colors='blue')
        ax.set_ylim(-2.4, 2.4)
        ax2 = ax.twinx()
        ax2.plot(self.traces['state-theta'], color='green', label='$\\theta$', ls='', marker='.', markersize=4, alpha=.5)
        ax2.set_ylim(-.21, .21)
        ax2.set_ylabel('pole angle ($\\theta$)')
        ax2.yaxis.label.set_color('green')
        ax2.tick_params(axis='y', colors='green')

    def plot_diagnostics(self):
        plt.figure(figsize=(10,10))
        ax = plt.subplot(221)
        self.plot_actions(ax=ax)
        ax = plt.subplot(222)
        self.plot_loss(ax=ax)
        ax = plt.subplot(223)
        self.plot_bellman_residual(ax=ax)
        ax = plt.subplot(224)
        self.plot_state(ax=ax)

    def action_counts(self):
        return np.array([
            len([_ for _ in self.traces['action'] if _ == 0]),
            len([_ for _ in self.traces['action'] if _ == 1]),
        ])


