import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        self._memory2txt = np.zeros((1000, 11 * 2 + 24 + 3), dtype=np.float32)

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1):
        data = (obs_t, action, reward, obs_tp1)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes, agent_idx):
        obses_t, actions, rewards, obses_tp1 = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            # print('all data=',data)
            # memory一条数据：
       # agent1 s 8   ([array([ 2.03130159,  0.99343303,  1.02605718,  2.42564815,  1.06909806,-0.08956268,  1.74261421,  1.72357365]),
       # agent2 s 10      array([-0.04304088,  2.51521083,  0.96220353,  1.0829957 , -0.04304088, 2.51521083, -1.06909806,  0.08956268,  0.67351615,  1.81313633]),
       # agent3 s 10     array([-0.71655703,  0.70207449,  0.28868738, -0.73014063, -0.71655703, 0.70207449, -1.74261421, -1.72357365, -0.67351615, -1.81313633])],
       # a 3*5=15    array([8.6060867e-02, 1.8589406e-01, 4.3036819e-02, 2.7548291e-02,6.5745997e-01, 7.6665112e-04, 5.0076656e-04, 3.2225009e-03,2.0509800e-03, 9.9345917e-01, 2.5560968e-03, 6.6452981e-03,9.4728643e-01, 3.9572250e-02, 3.9399415e-03], dtype=float32),
       # r           [-7.47192834842456, 1.772856723227769, 1.772856723227769],
       # agent1 s' 8           [array([ 2.04681398,  1.09504845,  1.04156957,  2.52726357,  1.07299832,-0.1119808 ,  1.72357833,  1.850753  ]),
       # agent2 s' 10           array([-0.03142875,  2.63924437,  0.97381567,  1.20702925, -0.03142875,2.63924437, -1.07299832,  0.1119808 ,  0.65058001,  1.9627338 ]),
       # agent3 s' 10            array([-0.68200876,  0.67651058,  0.32323566, -0.75570455, -0.68200876,0.67651058, -1.72357833, -1.850753  , -0.65058001, -1.9627338 ])],
       # done            [False, False, False])
            obs_t, action, reward, obs_tp1 = data
            # obses_t.append(np.concatenate(obs_t[:]))   # s拉到一起        yuanlai
            obses_t.append(obs_t)   # s拉到一起
            actions.append(action)
            rewards.append(reward[agent_idx])         # agent[i]的r
            # obses_tp1.append(np.concatenate(obs_tp1[:]))  # s’拉到一起         yuanlai
            obses_tp1.append(obs_tp1)  # s’拉到一起
                       # agent[i]的done
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1)
    # 随机选batch_size个index 范围0-storage
    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size, agent_idx):
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes, agent_idx)

    def collect(self):
        return self.sample(-1)

