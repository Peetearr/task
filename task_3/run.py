import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
from hopper import Hopper
import mujoco
import mujoco.viewer

device = 'cuda'
agent = Hopper()
model = agent.model
data = agent.data
min_torque = -100
max_torque = 100
action_disc = np.linspace(min_torque, max_torque, 100)

pos = np.linspace(-2,10,100)
vel = np.linspace(-10,10,100)
contact = np.linspace(0, 80, 100)
acc = np.linspace(-50, 50, 100)
forcetorque = np.zeros(6)

# Определение модели
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Класс памяти
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# FrozeLake Deep Q-Learning
class FrozenLakeDQL():
    # Настройка гиперпараметров
    learning_rate_a = 0.001         # alpha
    discount_factor_g = 0.9         # gamma
    network_sync_rate = 100         # кол-во шагов для синхронизации целевой сети и сети политики
    replay_memory_size = 1000       # длина памяти
    mini_batch_size = 100           # размер выборки памяти

    loss_fn = nn.MSELoss()          # функция потерь MSE
    optimizer = None                

    def train(self, episodes):
        
        num_states = 4
        num_actions = 10000
        
        epsilon = 1                 # установка начальной epsilon
        memory = ReplayMemory(self.replay_memory_size)

        # создание целевой сети и сети политики
        policy_dqn = DQN(in_states=num_states, h1_nodes=1024, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=1024, out_actions=num_actions)

        # установка одинаковых начальных значений сетей
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Установка оптимизатора "Adam" 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # инициализация списка наград за эпизод
        rewards_per_episode = np.zeros(episodes)

        epsilon_history = []

        # счетчик шагов для синхронизации сетей
        step_count=0

        
        mujoco.mj_resetData(model, data)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat = np.array([0, 0, 1])
            viewer.cam.distance = 5
            viewer.cam.azimuth = 45
            viewer.cam.elevation = 0
            for i in range(episodes):
                print(i)
                mujoco.mj_resetData(model, data)

                # расчет состояния (скорость, ускорение, позиция по Z, сила контакта)
                for j,c in enumerate(data.contact):
                    mujoco.mj_contactForce(model, data, j, forcetorque)
                state = np.array([np.digitize(data.body('mass').cvel[5], vel),
                            np.digitize(data.body('mass').cacc[5], acc),
                            np.digitize(data.body('mass').xpos[2], pos),
                            np.digitize(forcetorque[0], contact)])
                
                terminated = False      # флаг падения корпуса
                truncated = False       # флаг таймера 20с
                reward = 0 
                count = 0
               
                while(not terminated and not truncated):
                    
                    # выбор действий
                    if random.random() < epsilon:
                        # случайные
                        action_1 = int(100*np.random.random())
                        action_2 = int(100*np.random.random())
                        data.ctrl[0] = action_disc[action_1] # actions: 0=left,1=down,2=right,3=up
                        data.ctrl[1] = action_disc[action_2]
                    else:
                        # на основе обученной политики           
                        with torch.no_grad():
                            tensor = policy_dqn(torch.tensor(state)/1)
                            action_1 =tensor.argmax().item()//100
                            action_2 =tensor.argmax().item()%100
                            data.ctrl[0] = action_disc[action_1]
                            data.ctrl[1] = action_disc[action_2]
                    
                    
                    mujoco.mj_step(model, data)
                    viewer.sync()

                    # расчет нового состояния
                    new_state = np.array([np.digitize(data.body('mass').cvel[5], vel),
                                np.digitize(data.body('mass').cacc[5], acc),
                                np.digitize(data.body('mass').xpos[2], pos),
                                np.digitize(forcetorque[0], contact)])
                    
                    if new_state[2]>22:
                        reward +=  0.001 + 0.1*(new_state[2]-36)*(new_state[2]>36)
                    else:
                        terminated = True

                    # сохранение опыта в память
                    memory.append((state, action_1*100+action_2, new_state, reward, terminated))   
                    
                    state = new_state
                    step_count+=1
                    count+=1
                    if count >20000: truncated=True
                    
                rewards_per_episode[i] = reward
                # проверка длины памяти 
                if len(memory)>self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)        

                    # уменьшение эпсилон
                    epsilon = max(epsilon - 1/episodes, 0)
                    epsilon_history.append(epsilon)

                    # копирование сети политики в целевую сеть при необходимиом количестве шагов
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0
                    
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:


            if terminated: 
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(torch.tensor(state)/1).max()
                    )

            current_q = policy_dqn(torch.tensor(state)/1)
            current_q_list.append(current_q)
            target_q = target_dqn(torch.tensor(state)/1) 
            target_q[action] = target
            target_q_list.append(target_q)
                
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        torch.save(policy_dqn.state_dict(), "task/task_3/net.pt")

        # оптимизация сети
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, episodes, is_slippery=False):
        
        num_states = 4
        num_actions = 10000

        policy_dqn = DQN(in_states=num_states, h1_nodes=1024, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("task/task_3/net.pt"))
        policy_dqn.eval()

        mujoco.mj_resetData(model, data)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat = np.array([0, 0, 1])
            viewer.cam.distance = 5
            viewer.cam.azimuth = 45
            viewer.cam.elevation = 0
            i=0

            for i in range(episodes):
                mujoco.mj_resetData(model, data)
                state = np.array([np.digitize(data.body('mass').cvel[5], vel),
                            np.digitize(data.body('mass').cacc[5], acc),
                            np.digitize(data.body('mass').xpos[2], pos),
                            np.digitize(forcetorque[0], contact)])
                terminated = False
                truncated = False          
                count=0
                while(not terminated and not truncated):  
                    with torch.no_grad():
                        tensor = policy_dqn(torch.tensor(state)/1)
                        action_1 =tensor.argmax().item()//100
                        action_2 =tensor.argmax().item()%100
                        data.ctrl[0] = action_disc[action_1]
                        data.ctrl[1] = action_disc[action_2]
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    state = np.array([np.digitize(data.body('mass').cvel[5], vel),
                                np.digitize(data.body('mass').cacc[5], acc),
                                np.digitize(data.body('mass').xpos[2], pos),
                                np.digitize(forcetorque[0], contact)])
                    if state[2]<22:
                        terminated = True
                    count+=1
                    truncated= count>20000
                i+=1


if __name__ == '__main__':
    frozen_lake = FrozenLakeDQL()
    is_slippery = False
    frozen_lake.train(100)
    frozen_lake.test(50)
    plt.plot(list(range(100)), frozen_lake.GLOBAL, label="$rw$")
    plt.xlabel('Time', loc='right')
    plt.ylabel('q', loc='top').set_rotation(np.pi/2)
    plt.legend()
    plt.grid()
    plt.show()