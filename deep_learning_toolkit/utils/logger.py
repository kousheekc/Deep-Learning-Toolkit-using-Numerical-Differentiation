import os
import csv
import shutil
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, dataset, optimizer):
        cur_dir = os.getcwd()
        self.optimizer = optimizer

        self.log_dir = os.path.join(cur_dir, 'logs', 'loss', dataset, optimizer)
        self.model_dir = os.path.join(cur_dir, 'logs', 'models', dataset, optimizer)

        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)

        os.makedirs(self.log_dir)
        os.makedirs(self.model_dir)

        self.num_of_logs = 0
        self.header = True

        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log(self, loss):
        file = os.path.join(self.log_dir, 'loss.csv')

        with open(file, 'a') as f:
            writer = csv.writer(f)
            if self.header:
                writer.writerow(['loss'])
                self.header = False

            writer.writerow([loss])

        self.writer.add_scalar('loss/epoch', loss, self.num_of_logs)
        self.num_of_logs += 1

    def log_model(self, parameters):
        np.save(os.path.join(self.model_dir, 'parameters'), parameters)
