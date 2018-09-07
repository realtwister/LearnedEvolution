from learnedevolution.agents.batch import Batch,OIARTV;
from collections import deque;
import numpy as np;

def fill_batch(batch, T):
    observations = [];
    internals = [];
    actions = [];
    rewards = [];
    terminals =[];
    values =[];
    for i in range(T):
        observations += [i];
        internals += [i];
        actions += [i];
        rewards += [np.random.rand()];
        terminals += [i==T-1];
        values += [np.random.rand()];
        batch.append(
            observations[-1],
            internals[-1],
            actions[-1],
            rewards[-1],
            terminals[-1],
            values[-1]
            );
    return observations, internals, actions, rewards, terminals, values;


def test_init():
    batch = Batch();
    assert batch._gamma == 0.95;
    assert isinstance(batch._memory, deque);
    assert len(batch._memory)==0;
    assert isinstance(batch._buffer, deque);
    assert len(batch._memory)==0;
    assert batch._memory.maxlen == 1000;

    batch = Batch(reward_discount=0.3, maxlen=100);
    assert batch._gamma == 0.3;
    assert isinstance(batch._memory, deque);
    assert len(batch._memory)==0;
    assert isinstance(batch._buffer, deque);
    assert len(batch._buffer) == 0;
    assert batch._memory.maxlen == 100;

def test_appendwithoutterminal():
    batch = Batch();
    for i in range(2):
        state,internals, action,reward,terminal,value = i,i,i,i,False,i;
        batch.append(state,internals, action,reward,terminal,value);
    assert len(batch._memory) == 0;
    assert len(batch._buffer) == 2;
    assert batch._buffer[0] == OIARTV(0,0,0,0,terminal,0)

def test_calculate_returns():
    # The return is defined as
    # R_t = sum_{k=0}^{T-t} gamma^k r[t+k] + gamma^{T-t+1} V(T+1)

    g = 0.95
    batch = Batch(reward_discount = g);
    r = [1,2,3];
    v = 10;
    returns = [
        r[0]+g*r[1]+g**2*r[2]+g**3*v,
        r[1]+g*r[2]+g**2*v,
        r[2]+g*v
        ];
    assert np.all(np.array(returns) == np.array(batch.calculate_returns(r,v)))

    # General setting
    T = 1000;
    r = np.random.rand(T);
    v = np.random.rand()*100;
    rets = batch.calculate_returns(r,v);
    for t in range(T):
        ret = g**(T-t)*v;
        for k in range(T-t):
            delta = g**k*r[t+k]
            ret += delta;

        assert np.abs(ret -rets[t])<1e-10, (t,delta)

def test_calculate_advantage():
    # The advantage is defined as
    # A_t = r_t+v_{t+1}-v_t

    g = 0.95
    batch = Batch(reward_discount = g);
    r = [1,2,3];
    v = [100,200,300];
    final_value = 400;
    true_advantages = [
        1+200-100,
        2+300-200,
        3+400-300,
    ]
    assert np.all(np.array(true_advantages) == np.array(batch.calculate_advantages(r,v, final_value)))

    # General setting
    T = 1000;
    r = np.random.rand(T);
    v = np.random.rand(T);
    final_value = np.random.rand();
    advs = batch.calculate_advantages(r,v,final_value);
    for t in range(T):
        if t == T-1:
            adv = r[t]+final_value - v[t];
        else:
            adv = r[t]+v[t+1]-v[t];

        assert np.abs(adv -advs[t])<1e-10, (t,delta)


def test_process_buffer():
    batch = Batch();
    T=100;
    observations, internals, actions, rewards, terminals, values = fill_batch(batch, T);
    assert len(batch._buffer) == 0;
    assert len(batch._memory) == T;
    advantages = batch.calculate_advantages(rewards,values);
    returns = batch.calculate_returns(rewards);
    for i,(observation, internal, action, ret, adv) in enumerate(batch._memory):
        assert observation == observations[i];
        assert internal == internals[i];
        assert action == actions[i];
        assert ret == returns[i];
        assert adv == advantages[i];
    for _ in range(100):
        fill_batch(batch, T);
    assert len(batch._memory) == batch._memory.maxlen;

def test_iter():
    batch = Batch();
    fill_batch(batch, 300);
    N = 3;
    batch_size = 160;
    n=0;
    for b in batch.iter(N, batch_size):
        n+=1;
        assert len(b) == batch_size;
    assert n == N;
