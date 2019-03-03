import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from env import env
from RL_brain import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt

vel = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) * 0.1
# the number of experiments
round = 1000


def run():
    step_of_each_round = []
    for i in range(round):
        print(i)
        observation = env.reset()
        env.plt('start')
        step = 0
        while True:
            observation_of_agent = []
            observation_of_agent_ = []
            for j in range(7):
                observation_of_agent.append(observation[j + 1] - observation[j])
            observation_of_agent.append(observation[0] - observation[7])

            action_list = np.array([[0., 0.], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
            # choose action
            action = []
            for j in range(8):
                action.append(RL[j].choose_action(observation_of_agent[j]))
                if np.linalg.norm(env.E[j]) > 0.1:
                    action_list[j] = vel[action[j]]

            # update environment
            observation_, reward, done = env.step(action_list)
            for j in range(7):
                observation_of_agent_.append(observation_[j + 1] - observation_[j])
            observation_of_agent_.append(observation_[0] - observation_[7])

            if i > 50:
                env.plt('update')

            # restore memory
            for j in range(8):
                RL[j].store_transition(observation_of_agent[j], action[j], reward[j], observation_of_agent_[j])

            if (step > 200) and (step % 5 == 0):
                for j in range(8):
                    RL[j].learn()

            if done:
                # env.plt('finish')
                # RL[1].plot_cost()
                env.plt('clean')
                break

            step = step + 1
        step_of_each_round.append(step)
    plt.ioff()
    for i in range(8):
            RL[i].plot_cost()
    plt.pause(5)
    print(sum(step_of_each_round) / round)
    plt.plot(step_of_each_round)
    plt.pause(0)


if __name__ == "__main__":
    env = env()
    RL = []
    for i in range(8):
        RL.append(DeepQNetwork(n_actions=4, n_features=2,
                               agent_id=i,
                               learning_rate=0.01,
                               reward_decay=0.9,
                               e_greedy=0.9,
                               replace_target_iter=200,
                               memory_size=2000,
                               output_graph=False
                               ))

    run()
