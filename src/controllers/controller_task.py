"""Task lifecycle status controller."""

import time
from typing import override
from controllers.abstract_controller import AbstractController
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.log import OrcaLog
import enum

orca_logger = OrcaLog.get_instance()

class TaskStatus(enum.Enum):
    NOT_STARTED = 0
    RUNNING = 1
    END = 2

class TaskStatusController(AbstractController):
    def __init__(self, env: OrcaGymLocalEnv,
                 base_body: str,
                 is_controller: bool = True
                 ):
        super().__init__(env, [], {}, base_body)
        self.current_status = TaskStatus.NOT_STARTED
        self.current_time = time.time()
        self.is_controller = is_controller

    @override
    def run_controller(self)-> TaskStatus:
        return self.current_status

    def update_task_status(self, next_status: bool):
        # Apply a debounce interval to controller-driven status transitions.
        if self.is_controller:
            current_time = time.time()
            if current_time - self.current_time < 0.2:
                return
            self.current_time = current_time
        if next_status:
            if self.current_status == TaskStatus.NOT_STARTED:
                self.current_status = TaskStatus.RUNNING
                orca_logger.info("Task status: RUNNING")
            elif self.current_status == TaskStatus.RUNNING:
                self.current_status = TaskStatus.END
                orca_logger.info("Task status: END")
            elif self.current_status == TaskStatus.END:
                self.current_status = TaskStatus.NOT_STARTED
                orca_logger.info("Task status: NOT_STARTED")
        
    def reset(self):
        self.current_status = TaskStatus.NOT_STARTED