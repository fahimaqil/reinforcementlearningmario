#################### Reference ################################################################################################################################################
"""""
### Super Mario Bros Gym Ai####
@misc{gym-super-mario-bros,
  author = {Christian Kauten},
  howpublished = {GitHub},
  title = {{S}uper {M}ario {B}ros for {O}pen{AI} {G}ym},
  URL = {https://github.com/Kautenja/gym-super-mario-bros},
  year = {2018},
}

###PPO Implementation####
@misc{ 
  author = {Colin Skow},
  howpublished = {GitHub},
  title = Coding Demos from the School of AI's Move37 Course,
  URL = {https://github.com/colinskow/move37.git},
  year = {2018},
}
'"""""
################################################################################################################################################


import os
from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
from torch.nn import init
import numpy as np
import math
from torch.nn import init
from multiprocessing_env import *
from model import *
from gym.wrappers import FrameStack
import copy 
from preprocessing import *

NUM_ENVS            = 8 
LEARNING_RATE       = 0.0005
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
PPO_EPSILON         = 0.2
CRITIC_DISCOUNT     = 0.01
ENTROPY_BETA        = 0.01
PPO_STEPS           = 256
MINI_BATCH_SIZE     = 64
PPO_EPOCHS          = 3
RB_ALPHA=0.5
LOAD_FILE=False
TESTING=False
TRAINING=False

####### https://github.com/colinskow/move37.git ###############
def make_env():
  def _thunk():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env=RewardScalar(env)
    env=WarpFrame(env)
    env=StochasticFrameSkip(env,4,0.5)
    env=FrameStack(env,4)
    env=ScaledFloatFrame(env)
    return env
  return _thunk
def state_converter(state):
   state=np.squeeze(state, axis = 4)
   state = torch.FloatTensor(state).to(device)
   state = state.float()
   return state

####### https://github.com/colinskow/move37.git ###############
def compute_gae(next_value, rewards, masks, values, gamma=GAMMA, lam=GAE_LAMBDA):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        # prepend to get correct order back
        returns.insert(0, gae + values[step])
    return returns

####### https://github.com/colinskow/move37.git ###############
def test_env(env, model, device, deterministic=True):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env=RewardScalar(env)
    env=WarpFrame(env)
    env=FrameStack(env,4)
    env=StochasticFrameSkip(env,4,0.5)
    env=ScaledFloatFrame(env)
   # env=gym.wrappers.Monitor(env, 'recording/PPORB5/{}'.format(str(num)), video_callable=lambda episode_id: True, force=True)
    state=env.reset()
    done = False
    total_reward = 0
    distance=[]
    print("yes")
    for i in range(2000):
        state = torch.FloatTensor(state).to(device)
        state = state.float()
        state=state.permute(3,0,1,2)
        dist, _ = model(state)
        policy=dist
        policy=Categorical(F.softmax(policy, dim=-1).data.cpu())
        actionLog=policy.sample()
        action=actionLog.numpy()
        next_state, reward, done, info = env.step(action[0])
        distance.append(info['x_pos'])
        state = next_state
        total_reward += reward
        env.render()

    print(total_reward)
    print(max(distance))

def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x

####### https://github.com/colinskow/move37.git ###############
def ppo_iter(states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    actions=actions.view(2048, 1)
    log_probs=(log_probs.view(2048,1))
    for _ in range(batch_size // MINI_BATCH_SIZE):
        rand_ids = np.random.randint(0, batch_size, 8)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]  

####### https://github.com/colinskow/move37.git ###############
def ppo_update(states, actions, log_probs, returns, advantages,rB, clip_param=PPO_EPSILON):
    count_steps = 0
    sum_loss_actor = 0.0
    sum_loss_critic = 0.0
    sum_loss_total = 0.0

    # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
    for _ in range(PPO_EPOCHS):
        # grabs random mini-batches several times until we have covered all data
        for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            policy=dist
            policy=Categorical(F.softmax(policy, dim=-1))
            entropy = policy.entropy().mean()
            actionLog=policy.sample()
            new_log_probs = policy.log_prob(actionLog)
            ratio = (new_log_probs - old_log_probs.cuda()).exp()
            surr1 = ratio * advantage
            if rB:
              surr2 = (ratio<(1.0-clip_param))*(-RB_ALPHA*ratio+(1+RB_ALPHA)*(1-clip_param)) + (ratio>(1.0+clip_param))*(-RB_ALPHA*ratio+(1+RB_ALPHA)*(1+clip_param)) + (ratio<=(1.0+clip_param))*(ratio>=(1.0-clip_param))*ratio
            else:
              surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * entropy
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 0.5)
            optimizer.step()

            # track statistics
            sum_loss_actor += actor_loss
            sum_loss_critic += critic_loss
            sum_loss_total += loss
            
            count_steps += 1 
    return sum_loss_critic.item(),sum_loss_total.item()

####### https://github.com/colinskow/move37.git ###############
if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)
    envs = [make_env() for i in range(8)]
    envs = SubprocVecEnv(envs)
    state = envs.reset()
    state_size=envs.observation_space.shape
    action_size=envs.action_space.n
    model = CnnActorCriticNetwork(state_size,action_size,True).to(device)

    if LOAD_FILE:
      model.load_state_dict(torch.load('modelPPORBFinal2.pth',map_location='cpu'))

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(model)
    frame_idx  =  0
    frameArr=[]
    rewardArr=[]
    train_epoch = 0
    XArrayMax=[]
    XArrayMean=[]
    actor_Array=[]
    critic_Array=[]
    loss_Array=[]
    step_Array=[]

    if TRAINING:
      for i in trange(10000):
          log_probs = []
          values    = []
          states    = []
          actions   = []
          rewards   = []
          masks     = []
          rewardsArray=[]
          totalRewards=[]
          XArray=[]
          currentXArray=[]
          f=0
          for i in range(PPO_STEPS):
              state=np.squeeze(state, axis = 4)
              state = torch.FloatTensor(state).to(device)
              state = state.float()
              dist, value = model(state)
              policy=dist
              policy=Categorical(F.softmax(policy, dim=-1).data.cpu())
              entropy = policy.entropy().mean()
              actionLog=policy.sample()
              action=actionLog.numpy()
              if f==0 or f==4:
                print(action)
              next_state, reward, done, info = envs.step(action)
              rewardsArray.append(reward)   
              currentXArray.append(info[0]['x_pos'])
              states.append(state)
              for i in done:
                if i==True:
                  totalRewards.append(sum(rewardsArray))
                  rewardsArray = []
                  XArray.append(currentXArray)
                  currentXArray = []
                  stop=True
                  break
              state=next_state
              masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
              log_prob = policy.log_prob(actionLog)
              log_probs.append(log_prob)
              values.append(value)
              rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
              actions.append(actionLog)
              frame_idx += 1
              f+=1
          if len(XArray)>0:
              test_reward = np.mean(np.array(totalRewards))
              rewardArr.append(test_reward)
              XArray=np.array(XArray)
              frameArr.append(frame_idx)
              print('Frame %s. reward: %s' % (frame_idx, test_reward))
              XArrayMean.append(np.hstack(XArray).mean())
              XArrayMax.append(np.hstack(XArray).max())            
              print("MAX Distance: %f" % np.hstack(XArray).max())            
          next_state = state_converter(next_state)
          _, next_value = model(next_state)
          returns = compute_gae(next_value, rewards, masks, values)
          returns   = torch.cat(returns).detach()
          log_probs = torch.cat(log_probs).detach()
          values    = torch.cat(values).detach()
          states    = torch.cat(states)
          actions   = torch.cat(actions)
          advantage = returns - values
          advantage = normalize(advantage)
          criticLoss,actorLoss=ppo_update( states, actions, log_probs, returns, advantage,True)
          step_Array.append(i)
          actor_Array.append(actorLoss)
          critic_Array.append(criticLoss)
          train_epoch += 1
          torch.save(model.state_dict(), 'modelPPORBFinal3.pth')
          f = open("modelPPORBFinal4.csv", "w")
          f.write("{},{}\n".format("Frame", "Reward","Max Xposition","Mean X"))
          for x in zip(frameArr, rewardArr,XArrayMax,XArrayMean):
            f.write("{},{},{},{}\n".format(x[0], x[1], x[2], x[3]))
          f.close()
          f = open("trainPPORBLoss4.csv", "w")
          f.write("{},{},{}\n".format("Step", "Critic","Actor"))
          for x in zip(frameArr,critic_Array,actor_Array):
            f.write("{},{},{}\n".format(x[0], x[1],x[2]))
          f.close()
    else:
        model5 = CnnActorCriticNetwork(state_size,action_size,True).to(device)
        optimizer5 = optim.Adam(model5.parameters(), lr=LEARNING_RATE)
        #### Agent after 12 hours ##########
        model5.load_state_dict(torch.load('modelPPORBFinal2.pth',map_location='cpu'))
        model5.eval()
        totalReward1=0
        totalReward2=0
        totalReward3=0
        totalReward4=0
        totalReward5=0

        disArr=0
        maxArr=[]
        for i in trange (1):
            dis,maxDis,reward=test_env(i,model5,device,"Sample")
            totalReward1+=reward
            disArr+=dis
            maxArr.append(maxDis)
            maxNum=max(maxArr)
            maxArr=[]
            print("################### 1 ####################")
            print(totalReward1/10)
            print(disArr/10)
            print("#######################################")
            disArr=0
            
                