import os
import torch


def save_optim_state(output_path,
                     optimizer: torch.optim.optimizer.Optimizer,
                     lr_scheduler: torch.optim.lr_scheduler):
    optim_path = os.path.join(output_path, "optim.bin")
    lr_sched_path = os.path.join(output_path, "lr.bin")
    torch.save(optimizer.state_dict(), optim_path)
    torch.save(lr_scheduler.state_dict(), lr_sched_path)
    print(f"Optimizer & LR Scheduler saved at {output_path}")
