import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
from arguments import parse_args
from replay_buffer import ReplayBuffer
from model import openai_actor, openai_critic
from SparkMAENV import SparkMAENV


def get_trainers(env, obs_shape_n, action_shape_n, arglist):
    """
    init the trainers or load the old model
    """
    actors_cur = [None for _ in range(env.n)]
    critics_cur = [None for _ in range(env.n)]
    actors_tar = [None for _ in range(env.n)]
    critics_tar = [None for _ in range(env.n)]
    optimizers_c = [None for _ in range(env.n)]
    optimizers_a = [None for _ in range(env.n)]
    input_size_global = sum(obs_shape_n) + sum(action_shape_n)

    if arglist.restore == True: # restore the model
        for idx in arglist.restore_idxs:
            trainers_cur[idx] = torch.load(arglist.old_model_name+'c_{}'.format(agent_idx))
            trainers_tar[idx] = torch.load(arglist.old_model_name+'t_{}'.format(agent_idx))

    # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
    for i in range(env.n):
        actors_cur[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        critics_cur[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
        actors_tar[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        critics_tar[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
        optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)
    actors_tar = update_trainers(actors_cur, actors_tar, 1.0) # update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0) # update the target par using the cur
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c

def update_trainers(agents_cur, agents_tar, tao):
    """
    update the trainers_tar par using the trainers_cur
    This way is not the same as copy_, but the result is the same
    out:
    |agents_tar: the agents with new par updated towards agents_current
    """
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key]*tao + \
                                (1-tao)*state_dict_t[key]
        agent_t.load_state_dict(state_dict_t)
    return agents_tar


def save_memory(memory):
    for i in range(memory._next_idx):
        arr0 = memory._storage[i][0]
        arr1 = memory._storage[i][1]
        arr2 = memory._storage[i][2]
        arr3 = memory._storage[i][3]
        memory._memory2txt[i] = np.hstack((arr0, arr1, arr2, arr3))
    np.savetxt('memory-maddpg.txt', memory._memory2txt, fmt='%f', delimiter=' ')
    print('memory saved...')

# storage:s,a,r,s_四个数组     memory.txt:拉成一行

def load_memory(memory):
    s_dim=11
    a_dim=24
    memory._memory2txt=np.loadtxt('memory-maddpg.txt',delimiter=' ')
    #    s=11 a=24
    for i in range(1000):
        s=memory._memory2txt[i][0:s_dim]
        a=memory._memory2txt[i][s_dim:s_dim+a_dim]
        r=memory._memory2txt[i][s_dim+a_dim:s_dim+a_dim+3]
        s_=memory._memory2txt[i][-s_dim:]
        s=np.array(s)
        a=np.array(a)
        r=np.array(r)
        s_=np.array(s_)
        memory._storage[i].append(s)
        memory._storage[i].append(a)
        memory._storage[i].append(r)
        memory._storage[i].append(s_)




def agents_train(arglist, memory, obs_size, action_size, \
                 actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c):
        # update every agent in different memory batch
    for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            # 3个agent对应3组网络，zip合并，enumerate打包迭代返回。一共三组（ac,at,cc,ct,oa,oc）
        if opt_c == None: continue # jump to the next model update

        _obs_n_o, _action_n, _rew_n, _obs_n_n = memory.sample(arglist.batch_size, agent_idx) # Note_The func is not the same as others
            # s,a,r,s',done     agent[idx]只有r和done是自己的，其他都是全体的
            # --use the date to update the CRITIC
        rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float) # set the rew to gpu
            # agent i的r
        # done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device) # set the rew to gpu
            # agent i的done
        action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
            # 所有的a
        obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
            # 所有的s
        obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)
            # 所有的s'
        action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                                    for idx, a_t in enumerate(actors_tar)], dim=1)
            # 所有的target a  拼接起来了
        q = critic_c(obs_n_o, action_cur_o).reshape(-1)       # q

            # 所有的s和所有a   reshape(-1)拉成一行
        q_ = critic_t(obs_n_n, action_tar).reshape(-1)      # q_
            # 所有的s'和所有a_target
        tar_value = q_*arglist.gamma + rew       # q_*gamma*done + reward

        loss_c = torch.nn.MSELoss()(q, tar_value) # bellman equation
        opt_c.zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
        opt_c.step()
            # 全局critic更新：所有东西都是全局，
            # --use the data to update the ACTOR
            # There is no need to cal other agent's action
        action_i_new = actor_c(obs_n_o[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]])
            # 自己的obs，自己的action
            # print('model_out,policy_c_new=',model_out,policy_c_new)
            # update the action of this agent
        action_cur_o[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = action_i_new
            # 把整体的a中agent[idx]的动作改一下
            # print('action_cur_o=',action_cur_o)
            # loss_pse = torch.mean(torch.pow(model_out, 2))
        loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))
            # 所有obs，所有action注意这里的action是别人的action加上自己的新action
        opt_a.zero_grad()
        loss_a.backward()
        nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
        opt_a.step()
        # update the tar par
    actors_tar = update_trainers(actors_cur, actors_tar, arglist.tao)
    critics_tar = update_trainers(critics_cur, critics_tar, arglist.tao)
    return actors_cur, actors_tar, critics_cur, critics_tar
#             rew= tensor([ 0.1554,  1.1252, -0.0573,  ...,  0.5463, -2.2780, -0.2678],
#                         device='cuda:0')
# done= tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')
# action_cur_o tensor([[4.3892e-01, 2.3939e-01, 1.4247e-02,  ..., 7.0067e-02, 7.4438e-02,
#                       4.6561e-02],
#                      [3.9591e-02, 8.9632e-02, 2.9476e-02,  ..., 5.1187e-01, 4.7294e-01,
#                       4.4096e-03],
#                      [2.4553e-01, 2.0556e-01, 3.8692e-02,  ..., 1.2092e-01, 3.1200e-02,
#                       3.2440e-01],
#                      ...,
#                      [5.0153e-01, 4.4634e-02, 3.3658e-02,  ..., 2.9266e-01, 1.1316e-01,
#                       7.9084e-02],
#                      [9.5490e-01, 4.5099e-02, 9.3256e-08,  ..., 2.7824e-01, 4.6557e-08,
#                       7.2174e-01],
#                      [4.7597e-01, 2.8424e-01, 1.3000e-01,  ..., 9.5890e-01, 1.6057e-02,
#                       1.1779e-03]], device='cuda:0')
# obs_n_o= tensor([[-0.3151, -0.5406, -0.0787,  ...,  0.3431,  1.3296,  0.9999],
#                  [-0.0579,  0.9396, -1.5271,  ..., -2.4721, -0.9187, -2.9631],
#                  [-0.5763,  1.2076, -2.3673,  ..., -1.4763,  0.5721, -2.4646],
#                  ...,
#                  [-0.3259, -0.2798, -0.2037,  ..., -0.6867, -0.5348, -0.6340],
#                  [-1.4845, -0.0915, -0.4810,  ...,  2.8178,  4.7899,  2.4012],
#                  [-0.9468,  0.7261, -0.8621,  ...,  0.0132, -0.4606, -0.2509]],
#                 device='cuda:0')
# obs_n_n= tensor([[-0.3354, -0.5400, -0.0990,  ...,  0.3197,  1.3323,  0.9623],
#                  [-0.0715,  1.0128, -1.5407,  ..., -2.5754, -0.9107, -3.0170],
#                  [-0.6175,  1.2473, -2.4085,  ..., -1.5180,  0.5324, -2.4948],
#                  ...,
#                  [-0.4002, -0.2666, -0.2780,  ..., -0.6932, -0.5207, -0.6612],
#                  [-1.4890, -0.0915, -0.4854,  ...,  2.9678,  5.0389,  2.5513],
#                  [-0.9743,  0.7197, -0.8896,  ...,  0.0271, -0.4356, -0.2627]],
#                 device='cuda:0')
# action_tar= tensor([[0.6098, 0.0767, 0.0510,  ..., 0.5980, 0.0563, 0.2605],
#                     [0.2563, 0.2915, 0.0038,  ..., 0.9609, 0.0131, 0.0083],
#                     [0.0085, 0.0201, 0.0053,  ..., 0.0535, 0.1113, 0.0045],
#                     ...,
#                     [0.0745, 0.4014, 0.0485,  ..., 0.0244, 0.8238, 0.0379],
#                     [0.0304, 0.9353, 0.0036,  ..., 0.7360, 0.0240, 0.0251],
#                     [0.1029, 0.3623, 0.0382,  ..., 0.8295, 0.0390, 0.0239]],
#                    device='cuda:0')
# q= tensor([-0.9292,  3.1620,  2.7752,  ...,  1.0828, -4.3375, -1.0423],
#           device='cuda:0', grad_fn=<ViewBackward>)
# q_= tensor([ 0.7258,  1.8031,  1.6498,  ...,  0.5155,  1.7434, -1.2757],
#            device='cuda:0', grad_fn=<ViewBackward>)
# model_out,policy_c_new= tensor([[ -3.2957,   8.1088,  -5.8539,   0.8280,  -1.1158],
#                                 [ -4.6910,  12.6698,  -4.0634,   0.6726,  -1.7807],
#                                 [ -4.2145,  13.1004,  -2.2763,  -2.4282,  -3.2974],
#                                 ...,
#                                 [ -1.5191,   3.1699,  -2.0989,   1.2840,  -0.2735],
#                                 [ -8.9939,  24.8753, -12.0271,  -4.9068,  -2.9353],
#                                 [ -2.8474,   8.5742,  -1.9785,  -1.5457,  -2.1726]], device='cuda:0',
#                                grad_fn=<AddmmBackward>)
#                         tensor([[5.3203e-06, 9.9883e-01, 2.2336e-06, 1.0205e-03, 1.4150e-04],
#                                  [4.5581e-09, 1.0000e+00, 3.1291e-08, 3.4651e-06, 2.0376e-07],
#                                  [4.1057e-09, 1.0000e+00, 1.9039e-06, 6.9175e-08, 5.0346e-09],
#                                  ...,
#                                  [3.6134e-03, 8.4441e-01, 4.1738e-03, 6.0214e-02, 8.7590e-02],
#                                  [1.5094e-16, 1.0000e+00, 2.2496e-18, 2.1915e-15, 5.4556e-14],
#                                  [1.0422e-06, 9.9981e-01, 4.9878e-05, 1.3149e-04, 5.7458e-06]],
#                                 device='cuda:0', grad_fn=<SoftmaxBackward>)
# action_cur_o= tensor([[4.3892e-01, 2.3939e-01, 1.4247e-02,  ..., 7.0067e-02, 7.4438e-02,
#                        4.6561e-02],
#                       [3.9591e-02, 8.9632e-02, 2.9476e-02,  ..., 5.1187e-01, 4.7294e-01,
#                        4.4096e-03],
#                       [2.4553e-01, 2.0556e-01, 3.8692e-02,  ..., 1.2092e-01, 3.1200e-02,
#                        3.2440e-01],
#                       ...,
#                       [5.0153e-01, 4.4634e-02, 3.3658e-02,  ..., 2.9266e-01, 1.1316e-01,
#                        7.9084e-02],
#                       [9.5490e-01, 4.5099e-02, 9.3256e-08,  ..., 2.7824e-01, 4.6557e-08,
#                        7.2174e-01],
#                       [4.7597e-01, 2.8424e-01, 1.3000e-01,  ..., 9.5890e-01, 1.6057e-02,
#                        1.1779e-03]], device='cuda:0', grad_fn=<CopySlices>)

def train(arglist):
    """step1: create the environment """
    env = SparkMAENV()
    """step2: create agents"""
    # obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    # action_shape_n = [env.action_space[i].n for i in range(env.n)] # no need for stop bit
    obs_shape_n= [7, 2, 2]
    action_shape_n= [17, 3, 4]
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers(env, obs_shape_n, action_shape_n, arglist)
    #memory = Memory(num_adversaries, arglist)
    memory = ReplayBuffer(arglist.memory_size)

    print('=2 The {} agents are inited ...'.format(env.n))
    print('=============================')

    """step3: init the pars """
    alltime=[]
    obs_size = []
    action_size = []
    count=0
    update_cnt = 0
    t_start = time.time()
    rew_n_old = [0.0 for _ in range(env.n)] # set the init reward
    agent_info = [[[]]] # placeholder for benchmarking info
    episode_rewards = [0.0] # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)] # individual agent reward   [0,0,0]
    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    # 数组的界限
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a
    # print('obs-size=',obs_size)
    # print('action-size=',action_size)

# obs-size= [(0, 8), (8, 18), (18, 28)]
# action-size= [(0, 5), (5, 10), (10, 15)]
    print('=3 starting iterations ...')
    print('=============================')
    obs_n = env.reset()
    var = 0.5
    # print('reset obs n=',obs_n)
# reset obs n= [array([-0.62287349, -1.51335813,  0.13172575, -0.15969423,  0.62638958,-1.03822719, -0.25517576,  0.06323998]),
#               array([-1.24926307, -0.47513094, -1.24926307, -0.47513094, -0.49466383,0.87853295, -0.62638958,  1.03822719, -0.88156534,  1.10146717]),
#               array([-0.36769774, -1.57659811, -0.36769774, -1.57659811,  0.38690151,-0.22293422,  0.25517576, -0.06323998,  0.88156534, -1.10146717])]
    for i in range(arglist.max_episode):
        for j in range(arglist.per_episode_max_len):
            # get action
            action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                        for agent, obs in zip(actors_cur, obs_n)]
            # print('action_n=',action_n)
            for i1 in range(3):
                for i2 in range(len(action_n[i1])):
                    action_n[i1][i2]=0.5+action_n[i1][i2]/2
            print('action_n=',action_n)
            action_n = [np.clip(np.random.normal(action_n[i], var), 0, 1) for i in range(3)]
            print('noise action_n=', action_n)
            # add noise for 3 action
            # print('action_n=',action_n)
            # action_n= [array([0.17200199, 0.58447105, 0.00387306, 0.18657362, 0.0530803 ],dtype=float32),
            #            array([0.1254732 , 0.0842263 , 0.33557272, 0.36403796, 0.09068981],dtype=float32),
            #            array([0.70688844, 0.08467325, 0.04576227, 0.04084975, 0.12182634],dtype=float32)]
            # interact with env
            new_obs_n, rew_n, dur = env.step(action_n,i,j)
            alltime.append(dur)

            # print('new_s_n=',new_obs_n)
            # print('r_n=',rew_n)
            # print('done_n=',done_n)
            # print('info_n=',info_n)
# new_s_n= [array([ 0.65506065, -3.26873333,  1.55775681, -2.38931705,  0.64016171, -2.91375116,  1.85534001, -2.29744744]),
#           array([ 0.9175951 ,  0.52443411,  0.01489895, -0.35498217,  0.9175951 ,0.52443411, -0.64016171,  2.91375116,  1.2151783 ,  0.61630372]),
#           array([-0.2975832 , -0.09186961, -1.20027935, -0.97128589, -0.2975832 ,-0.09186961, -1.85534001,  2.29744744, -1.2151783 , -0.61630372])]
# r_n= [-8.135442234704628, 2.5408282004788054, 2.5408282004788054]
# done_n= [False, False, False]
# info_n= {'n': [{}, {}, {}]}
            # save the experience
            # memory.add(obs_n, np.concatenate(action_n), rew_n , new_obs_n)       yuanlai


            memory.add(np.concatenate(obs_n), np.concatenate(action_n), rew_n , np.concatenate(new_obs_n))
            # print(memory._storage)
            # np.concatenate拼接数组，action_n是3个一维数组，直接拼接成一行
            # update the obs_n
            count += 1
            if memory._next_idx>10:
                var *= .99  # decay the action randomness
                actors_cur, actors_tar, critics_cur, critics_tar = agents_train( \
                    arglist, memory, obs_size, action_size, \
                    actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)
                print('var=', var)


            if count > arglist.start_save_model and count % arglist.fre4save_model == 0:
                time_now = time.strftime('%y%m_%d%H%M')
                print('=time:{} step:{}        save'.format(time_now, count))
                model_file_dir = os.path.join(arglist.save_dir, '{}'.format(time_now))
                if not os.path.exists(model_file_dir):  # make the path
                    os.mkdir(model_file_dir)
                for agent_idx, (a_c, a_t, c_c, c_t) in \
                        enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                    torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                    torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                    torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                    torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

            obs_n = new_obs_n   # error
        # evert episode save memory
        save_memory(memory)

        file = open('test.txt', 'w')
        file.write(str(alltime))
        file.close()

            # if ep_reward > -300:RENDER = True


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)

  # 1-33 20s