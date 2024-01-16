import json
import argparse
import os
import tensorflow as tf
import numpy as np
import time
from utils import logger

class Trainer(object):
    def __init__(self,algo,
                env,
                sampler,
                sample_processor,
                policy,
                n_itr,
                greedy_finish_time,
                save_path,
                start_itr=0,
                inner_batch_size = 500,
                save_interval = 100):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.inner_batch_size = inner_batch_size
        self.greedy_finish_time = greedy_finish_time
        self.save_interval = save_interval
        self.save_path = save_path

    def train(self):
        """
        Implement the MRLCO training process for task offloading problem
        """

        start_time = time.time()
        avg_ret = []
        avg_loss = []
        avg_latencies = []
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")

            task_ids = self.sampler.update_tasks()
            paths = self.sampler.obtain_samples(log=False, log_prefix='')

            greedy_run_time = [self.greedy_finish_time[x] for x in task_ids]
            logger.logkv('Average greedy latency', np.mean(greedy_run_time))

            """ ----------------- Processing Samples ---------------------"""
            logger.log("Processing samples...")
            samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')

            """ ------------------- Inner Policy Update --------------------"""
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.inner_batch_size )

            print("average task losses: ", np.mean(policy_losses))
            avg_loss.append(np.mean(policy_losses))

            print("average value losses: ", np.mean(value_losses))

            """ ------------------ Resample from updated sub-task policy ------------"""
            print("Evaluate the one-step update for sub-task policy")
            new_paths = self.sampler.obtain_samples(log=True, log_prefix='')
            new_samples_data = self.sampler_processor.process_samples(new_paths, log="all", log_prefix='')

            """ ------------------ Outer Policy Update ---------------------"""
            logger.log("Optimizing policy...")
            self.algo.UpdateMetaPolicy()

            """ ------------------- Logging Stuff --------------------------"""

            ret = np.array([])
            for i in range(META_BATCH_SIZE):
                ret = np.concatenate((ret, np.sum(new_samples_data[i]['rewards'], axis=-1)), axis=-1)

            avg_reward = np.mean(ret)

            latency = np.array([])
            for i in range(META_BATCH_SIZE):
                latency = np.concatenate((latency, new_samples_data[i]['finish_time']), axis=-1)

            avg_latency = np.mean(latency)
            avg_latencies.append(avg_latency)


            logger.logkv('Itr', itr)
            logger.logkv('Average reward', avg_reward)
            logger.logkv('Average latency', avg_latency)

            logger.dumpkvs()
            avg_ret.append(avg_reward)

            if itr % self.save_interval == 0:
                self.policy.core_policy.save_variables(save_path=self.save_path+"/meta_model_"+str(itr)+".ckpt")

        self.policy.core_policy.save_variables(save_path=self.save_path+"/meta_model_final.ckpt")

        return avg_ret, avg_loss, avg_latencies


if __name__ == "__main__":
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.MRLCO import MRLCO

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='./train_config.json', type=str, 
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
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir=save_path+"/train_logs/", format_strs=['stdout', 'log', 'csv'])

    META_BATCH_SIZE = c.meta_batch_size

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

    action, greedy_finish_time = env.greedy_solution()
    print("avg greedy solution: ", np.mean(greedy_finish_time))
    print()
    finish_time = env.get_all_mec_execute_time()
    print("avg all remote solution: ", np.mean(finish_time))
    print()
    finish_time = env.get_all_locally_execute_time()
    print("avg all local solution: ", np.mean(finish_time))
    print()

    baseline = ValueFunctionBaseline()

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

    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=META_BATCH_SIZE, 
                                    obs_dim=c.obs_dim,
                                    vocab_size=c.action_dim,
                                    hparams=hparams)

    sampler = Seq2SeqMetaSampler(
        env=env,
        policy=meta_policy,
        rollouts_per_meta_task=1,  # This batch_size is confusing
        meta_batch_size=META_BATCH_SIZE,
        max_path_length=c.max_path_length,
        parallel=False,
    )

    sample_processor = Seq2SeqMetaSamplerProcessor(baseline=baseline,
                                                   discount=c.gamma,
                                                   gae_lambda=c.tau,
                                                   normalize_adv=c.normalize_adv,
                                                   positive_adv=c.positive_adv)
    algo = MRLCO(policy=meta_policy,
                 meta_sampler=sampler,
                 meta_sampler_process=sample_processor,
                 inner_lr=c.inner_lr,
                 outer_lr=c.outer_lr,
                 meta_batch_size=META_BATCH_SIZE,
                 num_inner_grad_steps=c.adaptation_steps,
                 clip_value = c.clip_eps,
                 max_grad_norm=c.max_grad_norm)

    trainer = Trainer(algo=algo,
                      env=env,
                      sampler=sampler,
                      sample_processor=sample_processor,
                      policy=meta_policy,
                      n_itr=c.num_iterations,
                      greedy_finish_time= greedy_finish_time,
                      start_itr=c.start_iter,
                      inner_batch_size=c.inner_batch_size,
                      save_interval=c.save_every,
                      save_path=save_path)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if c.load:
            meta_policy.core_policy.load_variables(c.load_path)
        avg_ret, avg_loss, avg_latencies = trainer.train()


