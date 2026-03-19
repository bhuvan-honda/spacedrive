# ------------------------------------------------------------------------
# SpaceDrive
# Copyright (c) 2026 Zhenghao Zhang. All Rights Reserved.
# ------------------------------------------------------------------------

from mmcv.runner.hooks import WandbLoggerHook
import wandb
import os

from mmcv.runner.hooks import HOOKS
from mmcv.runner import master_only

@HOOKS.register_module()
class CustomWandbLoggerHook(WandbLoggerHook):  
    def __init__(self, print_interval: int = 10, run_name = None,by_epoch=False) -> None:
        """
        Args:
            print_interval (int): iteration interval to print the log. Default: 10.
        """
        super().__init__(interval=print_interval, by_epoch=by_epoch)
        # init wandb
        self.init_kwargs = {
            'project': os.environ["PROJECT_NAME"],
            'name': os.environ["RUN_NAME"] if "RUN_NAME" in os.environ else run_name,
            'settings': wandb.Settings(
                code_dir=os.path.dirname(os.path.realpath(__file__))
            )
        }

