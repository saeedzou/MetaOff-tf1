import json
import argparse
import os
import tensorflow as tf
import numpy as np
import time
from utils import logger

class Trainer():
    def __init__(self,algo,
                env,
                sampler,
                sample_processor,
                policy,
                n_itr,
                batch_size=500,
                start_itr=0):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size

    def train(self):
        """
        Implement the repilte algorithm for ppo reinforcement learning
        """
        start_time = time.time()
        avg_ret = []
        avg_pg_loss = []
        avg_vf_loss = []

        avg_latencies = []
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")

            paths = self.sampler.obtain_samples(log=True, log_prefix='')

            """ ----------------- Processing Samples ---------------------"""
            logger.log("Processing samples...")
            samples_data = self.sampler_processor.process_samples(paths, log='all', log_prefix='')

            """ ------------------- Inner Policy Update --------------------"""
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.batch_size)

            #print("task losses: ", losses)
            print("average policy losses: ", np.mean(policy_losses))
            avg_pg_loss.append(np.mean(policy_losses))

            print("average value losses: ", np.mean(value_losses))
            avg_vf_loss.append(np.mean(value_losses))

            """ ------------------- Logging Stuff --------------------------"""

            ret = np.sum(samples_data['rewards'], axis=-1)
            avg_reward = np.mean(ret)

            latency = samples_data['finish_time']
            avg_latency = np.mean(latency)

            avg_latencies.append(avg_latency)


            logger.logkv('Itr', itr)
            logger.logkv('Average reward', avg_reward)
            logger.logkv('Average latency', avg_latency)
            logger.dumpkvs()
            avg_ret.append(avg_reward)

        return avg_ret, avg_pg_loss,avg_vf_loss, avg_latencies

if __name__ == "__main__":
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from policies.meta_seq2seq_policy import  Seq2SeqPolicy
    from samplers.seq2seq_sampler import Seq2SeqSampler
    from samplers.seq2seq_sampler_process import Seq2SeSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.ppo_offloading import PPO
    from utils import utils, logger

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='./val_config.json', type=str, 
                        help='configuration file path')
    args = parser.parse_args()

    with open(args.config) as f:
        data = json.load(f)

    class Config:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                setattr(self, key, value)
    
    c = Config(data)
    save_path = f'{c.save_path}/h_{c.encoder_units}_isAtt_{c.is_attention}_olr_{c.outer_lr}_ilr_{c.inner_lr}_mbs_{c.meta_batch_size}_obs_{c.obs_dim}_as_{c.adaptation_steps}_ib_{c.inner_batch_size}_ml_{c.max_path_length}_dp_{c.dropout}_nl_{c.num_layers}_nrl_{c.num_residual_layers}_isb_{c.is_bidencoder}_mec_{c.mec_process_capable}_mob_{c.mobile_process_capable}_bwu_{c.bandwidth_up}_bwd_{c.bandwidth_down}_ut_{c.unit_type}_sp_{c.start_iter}'
    logger.configure(dir=save_path+f"/val_logs_{c.iter}/", format_strs=['stdout', 'log', 'csv'])
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    resource_cluster = Resources(mec_process_capable=(c.mec_process_capable * 1024 * 1024),
                                 mobile_process_capable=(c.mobile_process_capable * 1024 * 1024),
                                 bandwidth_up=c.bandwidth_up, 
                                 bandwidth_dl=c.bandwidth_down)

    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=c.graph_number,
                                graph_number=c.graph_number,
                                graph_file_paths=c.graph_file_paths,
                                time_major=False,
                                encoding=c.encoding)

    print("calculate baseline solution======")

    env.set_task(0)
    action, finish_time = env.greedy_solution()
    target_batch, task_finish_time_batch = env.get_reward_batch_step_by_step(action[env.task_id],
                                          env.task_graphs_batchs[env.task_id],
                                          env.max_running_time_batchs[env.task_id],
                                          env.min_running_time_batchs[env.task_id])
    discounted_reward = []
    for reward_path in target_batch:
        discounted_reward.append(utils.discount_cumsum(reward_path, 1.0)[0])

    print("avg greedy solution: ", np.mean(discounted_reward))
    print("avg greedy solution: ", np.mean(task_finish_time_batch))
    print("avg greedy solution: ", np.mean(finish_time))

    print()
    finish_time = env.get_all_mec_execute_time()
    print("avg all remote solution: ", np.mean(finish_time))
    print()
    finish_time = env.get_all_locally_execute_time()
    print("avg all local solution: ", np.mean(finish_time))

    hparams = tf.contrib.training.HParams(
            unit_type=c.unit_type,
            encoder_units=c.encoder_units,
            decoder_units=c.decoder_units,
            n_features=c.action_dim,
            time_major=c.time_major,
            is_attention=c.is_attention,
            forget_bias=c.forget_bias,
            dropout=c.dropout,
            num_gpus=1,
            num_layers=c.num_layers,
            num_residual_layers=c.num_residual_layers,
            start_token=c.start_token,
            end_token=c.end_token,
            is_bidencoder=c.is_bidencoder
        )
    
    policy = Seq2SeqPolicy(obs_dim=c.obs_dim,
                           vocab_size=c.action_dim,
                           name="core_policy",
                           hparams=hparams)

    sampler = Seq2SeqSampler(env,
                             policy,
                             rollouts_per_meta_task=1,
                             max_path_length=c.max_path_length,
                             envs_per_task=None,
                             parallel=False)

    baseline = ValueFunctionBaseline()

    sample_processor = Seq2SeSamplerProcessor(baseline=baseline,
                                              discount=c.gamma,
                                              gae_lambda=c.tau,
                                              normalize_adv=c.normalize_adv,
                                              positive_adv=c.positive_adv)
    algo = PPO(policy=policy,
               meta_sampler=sampler,
               meta_sampler_process=sample_processor,
               lr=c.inner_lr,
               num_inner_grad_steps=c.adaptation_steps,
               clip_value=c.clip_eps,
               max_grad_norm=c.max_grad_norm)

    # define the trainer of ppo to evaluate the performance of the trained meta policy for new tasks.
    trainer = Trainer(algo=algo,
                      env=env,
                      sampler=sampler,
                      sample_processor=sample_processor,
                      policy=policy,
                      n_itr=c.num_iterations,
                      start_itr=c.start_iter,
                      batch_size=c.inner_batch_size)

    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        if c.load:
            policy.load_variables(load_path=c.load_path)
        avg_ret, avg_pg_loss, avg_vf_loss, avg_latencies = trainer.train()


