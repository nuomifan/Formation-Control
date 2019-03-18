from env import env
from RL_brain import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv

start = datetime.datetime.now()

acc = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]]) * 0.1
# the number of experiments
round = 50


def run():
    step_of_each_round = []

    for i in range(round):

        t1 = datetime.datetime.now()

        print("round :", i)

        observation = env.reset()

        # env.plt('start')
        step = 0
        while True:
            # choose action
            action = []
            for j in range(8):
                action.append(RL[j].choose_action(observation))

            # update environment
            observation_, reward, done = env.step(action)

            # 训练一段时间后，更新画面
            # if step > 10000:
            #     env.plt('update')

            # restore memory
            for j in range(8):
                RL[j].store_transition(observation, action[j], reward[j], observation_)

            if (step > 200) and (step % 5 == 0):
                for j in range(8):
                    RL[j].learn()

            if done:
                break
            observation = observation_
            step = step + 1
        step_of_each_round.append(step)
        t2 = datetime.datetime.now()
        print(t2 - t1)
    end = datetime.datetime.now()
    print(end - start)

    # output data
    csvFile = open('./data.csv', "a", newline='')
    data = RL[0].layers
    data.append(sum(step_of_each_round) / round)
    data.append(sum(step_of_each_round[-51:-1]) / 50)
    writer = csv.writer(csvFile, dialect='excel')
    writer.writerow(data)
    csvFile.close()

    print('average step: ', sum(step_of_each_round) / round)
    print('average step of latest 50 rounds: ', sum(step_of_each_round[-51:-1]) / 50)
    plt.plot(step_of_each_round)
    plt.pause(0)


if __name__ == "__main__":
    env = env()
    RL = []
    for i in range(8):
        RL.append(DeepQNetwork(n_actions=5, n_features=16,
                               agent_id=i,
                               learning_rate=0.01,
                               reward_decay=0.9,
                               e_greedy=0.9,
                               replace_target_iter=200,
                               memory_size=2000,
                               output_graph=False
                               ))
    run()
