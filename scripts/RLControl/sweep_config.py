import wandb
wandb.login()


# define the sweep configuration
# define the sweep configuration
sweep_config = {
    'method': 'bayes',
    'name': 'Drone-PPO',
    'description': 'Tianshou PPO, ActorCritic 2 Hidden Layers [128,128]',
    'metric': {
        'name': 'info/best_reward',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate':{'values': [3e-4, 5e-4, 1e-3, 1e-4]},
        'repeat_per_collect': {'values': [1, 2, 5, 10, 15]},
        'step_per_epoch': {'values': [1000,2000,5000,10000,20000,50000]},
        'step_per_collect': {'values': [100,200,500,1000,2000,5000]},
        'batch_size': {'values': [4,16,64,128,256,1024]},
    }
}

sweep_id = wandb.sweep(
  sweep=sweep_config,
  entity="mdianaRLSched",
  project="drone"
  )

print(sweep_id)