import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from lib.acquisition_fn import get_acq_fn
from lib.dataset import get_dataset
from lib.generator import get_generator
from lib.logging import get_logger
from lib.oracle_wrapper import get_oracle
from lib.proxy import get_proxy_model
from lib.utils.env import get_tokenizer

from metrics import eval_metrics
from samples import sample_batch, RolloutWorker

parser = argparse.ArgumentParser()

# General params
parser.add_argument("--name", default='test_mlp')
parser.add_argument("--device", default='cuda')
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--enable_tensorboard", action="store_true")

# Task/dataset params
parser.add_argument("--task", default="tfbind", type=str)
parser.add_argument("--vocab_size", default=4, type=int)
parser.add_argument("--max_len", default=8, type=int)
# parser.add_argument("--gen_max_len", default=8, type=int)   # why two max_len here?

# Proxy params
parser.add_argument("--save_path", default='results/tfbind')
parser.add_argument("--proxy_num_iterations", default=3000, type=int)
parser.add_argument("--proxy_num_hid", default=64, type=int)
parser.add_argument("--proxy_learning_rate", default=1e-4, type=float)
parser.add_argument("--proxy_L2", default=1e-4, type=float)
parser.add_argument("--gen_reward_min", default=0, type=float)

# Acquisition function params
parser.add_argument("--acq_fn", default="none", type=str)

# Sampler params
parser.add_argument("--num_rounds", default=1, type=int)
parser.add_argument("--num_sampled_per_round", default=128, type=int)
parser.add_argument("--n_sample_round", default=5, type=int)

# Generator params
parser.add_argument("--gen_do_explicit_Z", default=0, type=int)
parser.add_argument("--gen_learning_rate", default=5e-4, type=float)
parser.add_argument("--gen_Z_learning_rate", default=5e-3, type=float)

# Train generator
parser.add_argument("--gen_reward_exp", default=2, type=float)
parser.add_argument("--gen_sampling_temperature", default=2., type=float)
parser.add_argument("--gen_reward_exp_ramping", default=3, type=float)
parser.add_argument("--gen_data_sample_per_step", default=16, type=int)
parser.add_argument("--gen_num_iterations", default=10000, type=int)


def train_generator(logger, task, generator, oracle, tokenizer, dataset, max_len=8, device='cpu',
                    gen_reward_exp=2, gen_sampling_temperature=2., gen_reward_min=0.,
                    gen_reward_exp_ramping=3., gen_data_sample_per_step=16, gen_num_iterations=1000):
    print("Training generator")
    visited = []
    rollout_worker = RolloutWorker(task, oracle, tokenizer, max_len=max_len, device=device,
                                   gen_reward_exp=gen_reward_exp, gen_sampling_temperature=gen_sampling_temperature,
                                   gen_reward_min=gen_reward_min, gen_reward_exp_ramping=gen_reward_exp_ramping,
                                   gen_data_sample_per_step=gen_data_sample_per_step)
    for it in tqdm(range(gen_num_iterations + 1)):
        rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it, dataset)
        visited.extend(rollout_artifacts["visited"])

        loss, loss_info = generator.train_step(rollout_artifacts["trajectories"])        
        logger.add_scalar("generator_total_loss", loss.item())
        for key, val in loss_info.items():
            logger.add_scalar(f"generator_{key}", val.item())
        if it % 100 == 0:
            rs = torch.tensor([i[-1] for i in rollout_artifacts["trajectories"]["traj_rewards"]]).mean()
            logger.add_scalar("gen_reward", rs.item())
    return rollout_worker, None


def construct_proxy(tokenizer, save_path, logger, dataset=None,
                    task='tfbind', vocab_size=4, max_len=8, device='cpu',
                    proxy_num_iterations=3000, proxy_num_hid=64, proxy_learning_rate=1e-4, proxy_L2=1e-4,
                    gen_reward_min=0., gen_reward_norm=1.,
                    acq_fn_type='none'):
    proxy = get_proxy_model(tokenizer, save_path, logger,
                            task, vocab_size, max_len, device='cpu',
                            proxy_num_iterations=proxy_num_iterations, proxy_num_hid=proxy_num_hid,
                            proxy_learning_rate=proxy_learning_rate, proxy_L2=proxy_L2)
    l2r = lambda x: x.clamp(min=gen_reward_min) / gen_reward_norm
    reward_exp_min = max(l2r(torch.tensor(gen_reward_min)), 1e-32)
    acq_fn = get_acq_fn(acq_fn_type)
    return acq_fn(proxy, l2r, dataset, device), reward_exp_min


def train(task, oracle, dataset, logger, save_path, vocab_size=4, max_len=8, device='cpu',
          proxy_num_iterations=3000, proxy_num_hid=64, proxy_learning_rate=1e-4, proxy_L2=1e-4,
          gen_reward_min=0., acq_fn='none',
          gen_do_explicit_Z=0, gen_learning_rate=5e-4, gen_Z_learning_rate=5e-3,
          gen_reward_exp=2, gen_sampling_temperature=2.,
          gen_reward_exp_ramping=3., gen_data_sample_per_step=16, gen_num_iterations=1000,
          num_sampled_per_round=128, n_sample_round=100, num_rounds=1,
          ):
    tokenizer = get_tokenizer(task)

    print("Initializing proxy:")
    logger.set_context("iter_0")
    proxy, reward_exp_min = construct_proxy(tokenizer, save_path, logger, dataset=dataset,
                            task=task, vocab_size=vocab_size, max_len=max_len, device=device,
                            proxy_num_iterations=proxy_num_iterations, proxy_num_hid=proxy_num_hid,
                            proxy_learning_rate=proxy_learning_rate, proxy_L2=proxy_L2,
                            gen_reward_min=gen_reward_min, acq_fn_type=acq_fn)
    eval_metrics(task, dataset)
    proxy.update(dataset)
    print(f"Will run for {num_rounds} rounds...")

    for rond in range(num_rounds):
        print(f"Training round {rond+1}:")
        logger.set_context(f"iter_{rond+1}")
        generator = get_generator(tokenizer, reward_exp_min, task=task, vocab_size=vocab_size, max_len=max_len, device=device,
                                  gen_do_explicit_Z=gen_do_explicit_Z, gen_learning_rate=gen_learning_rate,
                                  gen_Z_learning_rate=gen_Z_learning_rate,)

        rollout_worker, losses = train_generator(logger, task, generator, oracle, tokenizer, dataset,
                                                 max_len=max_len, device=device,
                                                 gen_reward_exp=gen_reward_exp,
                                                 gen_sampling_temperature=gen_sampling_temperature,
                                                 gen_reward_min=gen_reward_min,
                                                 gen_reward_exp_ramping=gen_reward_exp_ramping,
                                                 gen_data_sample_per_step=gen_data_sample_per_step,
                                                 gen_num_iterations=gen_num_iterations)
        batch = sample_batch(task, rollout_worker, generator, oracle,
                             num_sampled_per_round=num_sampled_per_round, num_round=n_sample_round)
        logger.add_object("collected_seqs", batch[0])
        logger.add_object("collected_seqs_scores", batch[1])
        dataset.add(batch)
        eval_metrics(task, dataset, collected=True)
        if task == 'amp' or rond != num_rounds - 1:
            proxy.update(dataset)


def main(args):
    save_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger = get_logger(args)

    print(f"task: {args.task}, device: {args.device}")

    oracle = get_oracle(task=args.task, device=args.device)
    dataset = get_dataset(task=args.task, oracle=oracle)

    train(args.task, oracle, dataset, logger, args.save_path,
          vocab_size=args.vocab_size, max_len=args.max_len, device=args.device,
          proxy_num_iterations=args.proxy_num_iterations, proxy_num_hid=args.proxy_num_hid,
          proxy_learning_rate=args.proxy_learning_rate, proxy_L2=args.proxy_L2,
          gen_reward_min=args.gen_reward_min, acq_fn=args.acq_fn,
          gen_do_explicit_Z=args.gen_do_explicit_Z, gen_learning_rate=args.gen_learning_rate,
          gen_Z_learning_rate=args.gen_Z_learning_rate,
          gen_reward_exp=args.gen_reward_exp, gen_sampling_temperature=args.gen_sampling_temperature,
          gen_reward_exp_ramping=args.gen_reward_exp_ramping, gen_data_sample_per_step=args.gen_data_sample_per_step,
          gen_num_iterations=args.gen_num_iterations,
          num_sampled_per_round=args.num_sampled_per_round, n_sample_round=args.n_sample_round, num_rounds=args.num_rounds,
          )

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)