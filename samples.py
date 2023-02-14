import numpy as np
import torch
from torch.distributions import Categorical
from lib.utils.distance import is_similar

class MbStack:
    """
    A stack structure to store samples? f is the oracle
    """
    def __init__(self, f):
        self.stack = []
        self.f = f

    def push(self, x, i):
        self.stack.append((x, i))

    def pop_all(self):
        if not len(self.stack):
            return []
        with torch.no_grad():
            ys = self.f([i[0] for i in self.stack]) # eos_tok == 2 in gfp
        idxs = [i[1] for i in self.stack]
        self.stack = []
        return zip(ys, idxs)


def filter_len(x, y, max_len):
    """
    Remove sequences with length longer than or equal to max_len
    """
    res = ([], [])
    for i in range(len(x)):
        if len(x[i]) < max_len:
            res[0].append(x[i])
            res[1].append(y[i])
    return res


class RolloutWorker:
    def __init__(self, task, oracle, tokenizer, max_len=8, device='cpu',
                 gen_reward_exp=2, gen_sampling_temperature=2., gen_reward_min=0.,
                 gen_reward_exp_ramping=3., gen_data_sample_per_step=16,  # End of task specific params
                 gen_episodes_per_step=16, gen_random_action_prob=0.001, gen_output_coef=10., gen_balanced_loss=1,
                 gen_reward_norm=1., gen_loss_eps=1e-5, gen_leaf_coef=25.,):
        self.oracle = oracle
        self.max_len = max_len
        self.episodes_per_step = gen_episodes_per_step
        self.random_action_prob = gen_random_action_prob
        self.reward_exp = gen_reward_exp
        self.sampling_temperature = gen_sampling_temperature
        self.out_coef = gen_output_coef

        self.balanced_loss = gen_balanced_loss == 1
        self.reward_norm = gen_reward_norm
        self.reward_min = torch.tensor(float(gen_reward_min))
        self.loss_eps = torch.tensor(float(gen_loss_eps)).to(device)
        self.leaf_coef = gen_leaf_coef
        self.exp_ramping_factor = gen_reward_exp_ramping
        self.gen_data_sample_per_step = gen_data_sample_per_step

        self.tokenizer = tokenizer
        # Build function to map likelihood to rewards?
        if self.exp_ramping_factor > 0:
            self.l2r = lambda x, t=0: (x) ** (1 + (self.reward_exp - 1) * (1 - 1 / (1 + t / self.exp_ramping_factor)))
        else:
            self.l2r = lambda x, t=0: (x) ** self.reward_exp
        self.device = device
        self.workers = MbStack(oracle)
        self.task = task
        if self.task == 'amp':
            # self.max_len = gen_max_len - 2
            self.eos_tok = -1
            self.eos_char = tokenizer.eos_token
            self.pad_tok = 22

    def rollout(self, model, episodes, use_rand_policy=True):
        visited = []
        lists = lambda n: [list() for i in range(n)]
        states = [[] for i in range(episodes)]
        traj_states = [[[]] for i in range(episodes)]
        if self.task == 'amp':
            states = [''] * episodes
            traj_states = [[''] for i in range(episodes)]
        traj_actions = lists(episodes)
        traj_rewards = lists(episodes)
        traj_dones = lists(episodes)

        for t in (range(self.max_len) if episodes > 0 else []):
            x = self.tokenizer.process(states).to(self.device)
            if self.task == 'amp':
                active_indices = np.int32([i for i in range(episodes) if not states[i].endswith(self.eos_char)])
                x = self.tokenizer.process([states[i] for i in active_indices]).to(self.device)
                lens = torch.tensor([len(i) for i in states if not i.endswith(self.eos_char)]).long().to(self.device)
            lens = torch.tensor([len(i) for i in states]).long().to(self.device)
            with torch.no_grad():
                if self.task == 'amp':
                    logits = model(x, lens, coef=self.out_coef, pad=self.pad_tok)
                else:
                    logits = model(x, None, coef=self.out_coef)
            if t == 0:
                logits[:, 0] = -1000  # Prevent model from stopping
                # without having output anything
            try:
                cat = Categorical(logits=logits / self.sampling_temperature)
            except:
                print(states)
                print(x)
                print(logits)
                print(list(model.model.parameters()))
            actions = cat.sample()
            if use_rand_policy and self.random_action_prob > 0:
                for i in range(actions.shape[0]):
                    if np.random.uniform(0, 1) < self.random_action_prob:
                        actions[i] = torch.tensor(np.random.randint(t == 0, logits.shape[1])).to(self.device)
            if self.task == 'amp':
                chars = [self.tokenizer.vocab.itos[i.item()] for i in actions]

                for i, c, a in zip(active_indices, chars, actions):
                    if c == self.eos_char or t == self.max_len - 1:
                        self.workers.push(states[i] + (c if c != self.eos_char else ''), i)
                        r = 0
                        d = 1
                    else:
                        r = 0
                        d = 0
                    traj_states[i].append(states[i] + c)
                    traj_actions[i].append(a)
                    traj_rewards[i].append(r)
                    traj_dones[i].append(d)
                    states[i] += c
                if all(i.endswith(self.eos_char) for i in states):
                    break

            else:
                # Append predicted characters for active trajectories
                for i, a in enumerate(actions):
                    if t == self.max_len - 1:
                        self.workers.push(states[i] + [a.item()], i)
                        r = 0
                        d = 1
                    else:
                        r = 0
                        d = 0
                    traj_states[i].append(states[i] + [a.item()])
                    traj_actions[i].append(a)
                    traj_rewards[i].append(r)
                    traj_dones[i].append(d)
                    states[i] += [a.item()]
        return visited, states, traj_states, traj_actions, traj_rewards, traj_dones

    def execute_train_episode_batch(self, model, it=0, dataset=None, use_rand_policy=True):
        # run an episode
        lists = lambda n: [list() for i in range(n)]  # create a list of n lists
        visited, states, traj_states, traj_actions, traj_rewards, traj_dones = self.rollout(model,
                                                            self.episodes_per_step, use_rand_policy=use_rand_policy)
        lens = np.mean([len(i) for i in traj_rewards])
        bulk_trajs = []
        rq = []
        for (r, mbidx) in self.workers.pop_all():
            traj_rewards[mbidx][-1] = torch.tensor(self.l2r(r, it))
            rq.append(r.item())
            s = states[mbidx]
            if self.task == 'amp':
                s = s + (self.eos_char if not s.endswith(self.eos_char) else '')
            visited.append((s, traj_rewards[mbidx][-1].item(), r.item()))
            bulk_trajs.append((s, traj_rewards[mbidx][-1].item()))
        if self.gen_data_sample_per_step > 0 and dataset is not None:
            n = self.gen_data_sample_per_step
            m = len(traj_states)
            if self.task == 'amp':
                x, y = dataset.sample(n)
                x, y = filter_len(x, y, self.max_len)
            else:
                x, y = dataset.sample(n)  # sample(n, 0.5)
            n = len(x)
            traj_states += lists(n)
            traj_actions += lists(n)
            traj_rewards += lists(n)
            traj_dones += lists(n)
            if self.task == 'amp':
                bulk_trajs += list(zip([i + self.eos_char for i in x], [self.l2r(torch.tensor(i), it) for i in y]))

                for i in range(len(x)):
                    traj_states[i + m].append('')
                    for c, a in zip(x[i] + self.eos_char, self.tokenizer.process([x[i] + self.eos_char])[0] - 2):
                        traj_states[i + m].append(traj_states[i + m][-1] + c)
                        traj_actions[i + m].append(a)
                        traj_rewards[i + m].append(0 if c != self.eos_char else self.l2r(torch.tensor(y[i]), it))
                        traj_dones[i + m].append(float(c == self.eos_char))
            else:
                bulk_trajs += list(zip([i for i in x], [self.l2r(torch.tensor(i), it) for i in y]))
                for i in range(len(x)):
                    traj_states[i + m].append([])
                    for c, a in zip(x[i], self.tokenizer.process([x[i]]).reshape(-1)):
                        traj_states[i + m].append(traj_states[i + m][-1] + [c])
                        traj_actions[i + m].append(a)
                        traj_rewards[i + m].append(0 if len(traj_actions[i + m]) != self.max_len else self.l2r(torch.tensor(y[i]), it))
                        traj_dones[i + m].append(float(len(traj_rewards[i + m]) == self.max_len))
        return {
            "visited": visited,
            "trajectories": {
                "traj_states": traj_states,
                "traj_actions": traj_actions,
                "traj_rewards": traj_rewards,
                "traj_dones": traj_dones,
                "states": states,
                "bulk_trajs": bulk_trajs
            }
        }


def filter_samples(samples, reference_set, filter_distance_type="edit", filter_threshold=0.1):
    filtered_samples = []
    for sample in samples:
        similar = False
        for example in reference_set:
            if is_similar(sample, example, filter_distance_type, filter_threshold):
                similar = True
                break
        if not similar:
            filtered_samples.append(sample)
    return filtered_samples


def sample_batch(task, rollout_worker, generator, oracle, num_sampled_per_round=128, num_round=100):
    print("Generating samples")
    samples = ([], [])
    scores = []
    while len(samples[0]) < num_sampled_per_round * num_round:
        rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it=0, use_rand_policy=False)
        states = rollout_artifacts["trajectories"]["states"]
        if task == 'amp':
            samples[0].extend(states)
            scores.extend([rews[-1].cpu().item() for rews in rollout_artifacts["trajectories"]["traj_rewards"]])
        else:
            vals = oracle(states).reshape(-1)
            samples[0].extend(states)
            samples[1].extend(vals)
            scores.extend(torch.tensor(rollout_artifacts["trajectories"]["traj_rewards"])[:, -1].numpy().tolist())

    idx_pick = np.argsort(scores)[::-1][:num_sampled_per_round]

    picked_states = np.array(samples[0])[idx_pick].tolist()
    if task == 'amp':
        return picked_states, np.array(oracle(picked_states)).tolist()
    else:
        return picked_states, np.array(samples[1])[idx_pick].tolist()



