import numpy as np
from collections import deque
import logging

# Optional: Set up logging to see when stopping occurs
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class EarlyStoppingCallback:
    """
    A callback to stop training if the mean reward doesn't improve significantly
    over a specified number of evaluation periods (epochs).

    :param patience: How many epochs to wait for improvement before stopping.
    :param min_delta: Minimum change in the monitored quantity to qualify as an improvement.
    :param min_pct_delta: Minimum percentage change in the monitored quantity to qualify as an improvement.
    :param warmup: Number of initial epochs to ignore before starting to monitor.
    :param verbose: If True, prints messages when checking and stopping.
    """
    def __init__(self,  patience: int = 10, min_delta: float = 0.1, min_pct_delta: float = 0.1, warmup: int = 5, verbose: bool = True,):
        if patience <= 0:
            raise ValueError("Patience must be a positive integer.")
        if min_delta < 0:
             raise ValueError("min_delta must be non-negative.")
        if min_pct_delta < 0:
             raise ValueError("min_pct_delta must be non-negative.")     
        if warmup < 0:
            raise ValueError("warmup must be non-negative.")

        
        self.patience = patience
        self.min_delta = min_delta
        self.min_pct_delta=min_pct_delta
        self.warmup = warmup
        self.verbose = verbose

        # Use deque to automatically keep track of the last 'patience' rewards
        self.reward_history:deque = deque(maxlen=self.patience)
        self.epoch_count = 0
        self.best_reward = -np.inf # Keep track of the best reward seen *after* warmup

    def setTrainer(self,trainer):
        self.trainer=trainer
    
    def __call__(self, mean_rewards: float) -> bool:
        """
        Call method executed by the Tianshou trainer.

        :param mean_rewards: The average reward obtained in the latest test phase.
        :return: True if training should stop, False otherwise.
        """
        if self.trainer==None:
            self.epoch_count += 1
        else :
            if self.epoch_count==self.trainer.epoch:
                self.best_reward = max(self.best_reward, mean_rewards)
                return False
            self.epoch_count=self.trainer.epoch
            #log.info(f"[EarlyStopping] Warmup epoch {self.epoch_count}/{self.warmup}. Current reward: {mean_rewards:.4f}")
        # --- Warmup Phase ---
        if self.epoch_count <= self.warmup:
            # During warmup, just record the reward and potentially update best_reward
            # We don't add to history yet as we don't check for stagnation
            self.best_reward = max(self.best_reward, mean_rewards)
            if self.verbose:
                log.info(f"[EarlyStopping] Warmup epoch {self.epoch_count}/{self.warmup}. Current reward: {mean_rewards:.4f}")
            return False # Don't stop during warmup

        # --- Monitoring Phase ---
        # Add current reward to history (deque handles the size limit)
        
        

        # Check if we have enough history to make a decision
        if len(self.reward_history) < self.patience:
            # Not enough history yet, update best_reward and continue
            
            self.best_reward = max(self.best_reward, mean_rewards)
            if self.verbose:
                log.info(f"[EarlyStopping] Collecting history ({len(self.reward_history)}/{self.patience}). Current reward: {mean_rewards:.4f}")
                self.reward_history.append(mean_rewards)
            return False

        reward_delta=mean_rewards-self.best_reward 
        reward_delta_pct=reward_delta/np.abs(self.best_reward)
        # Check 1: Has the best reward overall improved recently?
        if reward_delta > self.min_delta or reward_delta_pct > self.min_pct_delta:
             # Significant improvement found, update best reward and reset patience implicitly
             if self.verbose:
                  log.info(f"[EarlyStopping] Improvement detected! Best reward updated from {self.best_reward:.4f} to {mean_rewards:.4f}")
             self.best_reward = max(self.best_reward, mean_rewards)     
             self.reward_history.clear()
             self.reward_history.append(mean_rewards)
             # Even though deque is fixed size, finding improvement means we shouldn't stop
             return False # Continue training

        # Check 2: If no significant improvement overall, check if we've waited long enough
        # At this point, we know:
        # 1. We are past the warmup phase.
        # 2. We have 'patience' entries in our history.
        # 3. The best reward in the last 'patience' epochs is NOT significantly better
        #    than the best reward seen before this window. -> Implies stagnation.
        if self.verbose:
            log.warning(f"[EarlyStopping] No significant improvement for {self.patience} epochs (current: {mean_rewards:.4f}, overall_best: {self.best_reward:.4f}). Stopping.")
        return True # Stop training