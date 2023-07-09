import hydra
import numpy as np
import shutil
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from src.utils import save_onnx
from numpy.core.fromnumeric import mean
import pytorch_lightning as pl
from src.noise import OUNoise
from collections import defaultdict
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def save_model(agent, epoch, loss, dest_dir):
    actor = (agent.actor, str(Path(dest_dir) / f"actor_{epoch}.pt"))
    critic = (agent.critic, str(Path(dest_dir) / f"critic_{epoch}.pt"))

    for net, path in (actor, critic):
        torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'loss': loss,
                }, path)

def load_model(agent, src_dir, epoch="4180"):
    device = torch.device('cuda:1')

    actor_path = str(Path(src_dir) / f"actor_{epoch}.pt")
    agent.actor.load_state_dict(torch.load(actor_path)["model_state_dict"])
    agent.actor = agent.actor.to(device)

    critic_path = str(Path(src_dir) / f"critic_{epoch}.pt")
    agent.critic.load_state_dict(torch.load(critic_path)["model_state_dict"])
    agent.critic = agent.critic.to(device)

    agent.hard_update_target_networks()

def report2tensorboard(training_step_outputs, idx):
    buffer = defaultdict(list)
    
    for out in training_step_outputs:
        for key, val in out.items():
            buffer[key].append(val)

    avg_buffer = dict()

    for key, list_val in buffer.items():
        avg_value = mean(list_val)
        avg_buffer[key] = avg_value

        writer.add_scalar(key, avg_value, idx)

    return avg_buffer

@hydra.main(version_base='1.2', config_path='config', config_name='config')
def main(cfg):
    pl.seed_everything(cfg.seed)
    if not cfg.only_get_onnx:
        logger = pl.loggers.TensorBoardLogger(cfg.logdir)
   
    env = hydra.utils.instantiate(cfg.env)

    act_noise = cfg.act_noise

    agent = hydra.utils.instantiate(
        cfg.agent,
        model_conf=cfg.model,
        memory_conf=cfg.memory,
        optimizer_conf=cfg.optimizer,
        scheduler_conf=cfg.scheduler,
        process_state_conf=cfg.process_state,
        target_noise=cfg.target_noise,
        noise_clip=cfg.noise_clip,
        _recursive_=False,
        )
    
    load_model(agent, src_dir="outputs/2023-06-14/10-19-30", epoch=1500)
    # load_model(agent, src_dir="outputs/2023-06-22/08-40-06", epoch=2000)
    # load_model(agent, src_dir="outputs/2023-06-24/16-01-25", epoch=1000)

    try:
        if not cfg.only_get_onnx:
        
            noise = OUNoise(cfg.action_size)
            for epoch in range(cfg.max_epochs):
                training_step_outputs = []
                done = False

                episode_reward = 0

                state = env.reset_random_init_pos()
                step = 0

                while not done:
                    action = agent.get_action(state)
                    env_action = np.reshape(action, (-1,2))

                    noise = np.random.normal(0,1, size=env_action[0].shape)*act_noise
                    env_action[0] = noise + env_action[0]
                    env_action[0] = np.clip(env_action[0], -1,1)
                    env_action[0] = env_action[0]*np.array([env.max_v, env.max_w])

                    new_state, reward, done = env.step(env_action)
                    
                    agent.save_experience(state, action, reward, done, new_state)

                    if len(agent.memory) > cfg.batch_size:
                        out = agent.train_networks(cfg.batch_size, step, reward)
                        training_step_outputs.append(out)

                    state = new_state
                    step += 1
                    episode_reward += reward

                avg_values = report2tensorboard(training_step_outputs, epoch)
                agent.step_schedulers(actor_loss=avg_values["actor_loss"], critic_loss=avg_values["critic_loss"])

                writer.add_scalar("reward", episode_reward, epoch)
                writer.add_scalar("epoch", epoch, epoch)
                writer.add_scalar("max_dist", env.max_dist, epoch)

                if epoch and not epoch % cfg.sync_save_onnx and not cfg.is_debug:
                    save_onnx(agent.actor, logger.save_dir, cfg.process_state.state_size)
                    
                if env.max_dist < 1.50:
                    env.max_dist += 0.10

                if epoch and not epoch % cfg.sync_save_pt:
                    save_model(agent=agent, 
                            epoch=epoch, 
                            loss=avg_values["actor_loss"]+avg_values["critic_loss"],
                            dest_dir=logger.save_dir,
                            )

    except KeyboardInterrupt:
        print('KeyboardInterrupt raised.')
    finally:
        if not cfg.is_debug:
            if cfg.only_get_onnx:
                save_onnx(agent.actor, "onnx", cfg.process_state.state_size)
            else:
                save_onnx(agent.actor, logger.save_dir, cfg.process_state.state_size)
        
        writer.flush()
        writer.close()
        
if __name__ == '__main__':
    main()