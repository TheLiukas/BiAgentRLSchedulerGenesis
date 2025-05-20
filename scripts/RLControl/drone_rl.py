from  datetime import datetime
import json
import torch
import tianshou as ts
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.nn.modules import Tanh
from tianshou.utils.logger.wandb import WandbLogger
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.policy import PPOPolicy
from torch.utils.tensorboard import SummaryWriter
import logging as logLevel
import numpy as np
from env_drone import DroneEnv
import os
import argparse

from torch.distributions import Independent, Normal
from earlystoppingclass import EarlyStoppingCallback


def save_params_to_log(logdir,log_par):
    with open(logdir+"/params.json","x+") as file:
        json.dump(log_par,file)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='DronePose'+str(datetime.now()))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--buffer-size', type=int, default=4096) # PPO typically uses steps_per_collect
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=30000)
    
    # PPO specific arguments
    parser.add_argument('--repeat-per-collect', type=int, default=10) # PPO updates per data collection
    parser.add_argument('--batch-size', type=int, default=4) # Mini-batch size for PPO updates
    parser.add_argument('--step-per-collect', type=int, default=2046) # Steps collected before update
    parser.add_argument('--vf-coef', type=float, default=0.5) # Value function loss coefficient
    parser.add_argument('--ent-coef', type=float, default=0.01) # Entropy coefficient
    parser.add_argument('--eps-clip', type=float, default=0.2) # PPO clipping epsilon
    parser.add_argument('--gae-lambda', type=float, default=0.95) # Generalized Advantage Estimation lambda
    parser.add_argument('--rew-norm', type=int, default=0) # Reward normalization
    parser.add_argument('--bound-action-method', type=str, default="clip") # Action bounding
    parser.add_argument('--max-grad-norm', type=float, default=0.5)

    # Trainer

    parser.add_argument('--training-num', type=int, default=10) # Number of parallel envs for training
    parser.add_argument('--test-num', type=int, default=100) # Number of parallel envs for testing
    parser.add_argument('--test-episodes', type=int, default=100*5)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0) # Not used in this setup directly
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Genesis
    parser.add_argument("-v", "--verbose", action="store_true", default=True)

    args = parser.parse_args()
    return args

def get_train_cfg(exp_name, max_iterations,seed=42):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "tanh",
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 100,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": seed,
    }

    return train_cfg_dict



def get_cfgs():
    env_cfg = {
        "num_actions": 4,
        # termination
        "termination_if_roll_greater_than": 180,  # degree
        "termination_if_pitch_greater_than": 180,
        "termination_if_close_to_ground": 0.1,
        "termination_if_x_greater_than": 4.0,
        "termination_if_y_greater_than": 4.0,
        "termination_if_z_greater_than": 2.0,
        # base pose
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 60.0,
        "at_target_threshold": 0.1,
        "clip_actions": 1.0,
        # visualization
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
    }
    obs_cfg = {
        "num_obs": 17,
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    reward_cfg = {
        "yaw_lambda": -10.0,
        "reward_scales": {
            "target": 10.0,
            "smooth": -1e-2,
            "yaw": 0.1,
            "angular": -2e-4,
            "crash": -15.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range": [-1.0, 1.0],
        "pos_y_range": [-1.0, 1.0],
        "pos_z_range": [1.0, 1.0],
        "hold_time" : 0
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def train_drone_ppo(args=get_args()):


    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    # --- Logger ---
    log_path = os.path.join(args.logdir, args.task, 'ppo')
    writer = SummaryWriter(log_path)
    #logger = ts.utils.WandbLogger(log_dir=log_path)
    logger=ts.utils.TensorboardLogger(writer)
     # -- Logging parameters --
    params_dict=env_cfg | obs_cfg | reward_cfg | command_cfg | args.__dict__
    save_params_to_log(logdir=log_path,log_par=params_dict)

    genesis_logging=logLevel.INFO if args.verbose else logLevel.ERROR

    train_envs = DroneEnv(  num_envs=args.training_num,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        device="cuda",seed=args.seed,
        n_render_env=[0],
        log_level=genesis_logging
    )
    test_envs= DroneEnv(  num_envs=args.test_num,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        device="cuda",seed=args.seed,
        n_render_env=[0],
        log_level=genesis_logging
    )


    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # --- Network Setup ---
     # Get env specs from a single instance
    args.state_shape = train_envs.observation_space.shape or train_envs.observation_space.n
    args.action_shape = train_envs.action_space.shape or train_envs.action_space.n
    args.max_action = train_envs.action_space.high[0] # Assumes symmetric [-max, max] action space after scaling

    # Model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device,activation=Tanh)
    actor = ActorProb(
        net_a, args.action_shape, unbounded=True, device=args.device
    ).to(args.device)

    net_c = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device,activation=Tanh)
    critic = Critic(net_c, device=args.device).to(args.device)


    # Initialize parameters correctly (orthogonal init often helps PPO)
    torch.nn.init.constant_(actor.sigma_param, -0.5) # Initialize log_std
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    # Optimizer
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)

    # --- PPO Policy Setup ---
    # Define the action distribution (Gaussian for continuous actions)
    def dist(*logits):
        # logits is (mu, log_sigma)
        mu, log_sigma = logits[0]
        sigma = torch.exp(log_sigma)
        # Use Independent to treat dimensions as independent samples from Normal
        return Independent(Normal(mu, sigma), 1)

    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=True, # Tianshou handles action scaling to env bounds if True
        action_bound_method=args.bound_action_method, # 'clip' or 'tanh'
        eps_clip=args.eps_clip,
        value_clip=False, # Typically False for PPO
        action_space=train_envs.action_space,
        # Note: Tianshou's PPO handles GAE calculation internally
    ).to(args.device)

    # --- Collector Setup ---
    # Use VectorReplayBuffer for on-policy algorithms like PPO
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(len(train_envs)*500, len(train_envs)), # Total buffer size across envs
        # exploration_noise=True # Handled by policy's stochastic nature
    )
    test_collector = Collector(policy, test_envs)

 
    
    
    # --- Callback functions ---
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    early_stopper = EarlyStoppingCallback(
        patience=20,
        min_delta=0.1,
        min_pct_delta=0.05,
        warmup=30,
        verbose=True
    )


    # --- Trainer ---
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        save_best_fn=save_best_fn,
        update_per_step=1/10,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.test_episodes, # Run test_num episodes for testing
        batch_size=args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=early_stopper,
        logger=logger
    )#.run()

    early_stopper.setTrainer(trainer)
    
 



    # --- Run Training ---
    print(f"Starting PPO training on {args.device}")
    try:
        trainer.run()
    except :
        print("Stopped training")
        save_best_fn(trainer.policy)
   
 
    #trainer.pprint_asdict()

    print(f"Finished training! Best policy saved at {os.path.join(log_path, 'policy.pth')}")

    # --- Close environments ---
    train_envs.close()
    test_envs.close()

    # --- Example: Evaluate the trained policy ---
    print("Evaluating trained policy...")
    policy.load_state_dict(torch.load(os.path.join(log_path, 'policy.pth'), map_location=args.device))
    policy.eval() # Set policy to evaluation mode (deterministic actions if needed, but PPO usually uses stochastic)
    env_cfg["visualize_target"]=True
    eval_env = DroneEnv(num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        device="cuda",seed=args.seed) # Use human rendering for eval
    eval_collector = Collector(policy, eval_env)
    result = eval_collector.collect(n_episode=5,reset_before_collect=True) # Render at 30 FPS
    print(f"Evaluation Result: {result}")
    eval_env.close()

if __name__ == '__main__':
    train_drone_ppo()


