import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, dataset, optimizer=None):
        self.dataset = dataset
        self.optimizer = optimizer

        sns.set()
        self.cur_dir = os.getcwd()
        self.plt_dir = os.path.join(self.cur_dir, 'logs', 'plots', self.dataset)

        if not os.path.exists(self.plt_dir):
            os.makedirs(self.plt_dir)

    def plot_individual(self, display=False):
        log_file = os.path.join(self.cur_dir, 'logs', 'loss', self.dataset, self.optimizer, 'loss.csv')
        loss = self.get_loss_from_file(log_file)

        ax = sns.lineplot(x=list(range(len(loss))), y=loss)
        ax.axes.set_title(f'{self.dataset} {self.optimizer}', fontsize=20)
        ax.set_xlabel('epoch', fontsize=15)
        ax.set_ylabel('loss', fontsize=15)

        plot_file_name = os.path.join(self.plt_dir, f'{self.dataset}_{self.optimizer}_plot.png')
        plt.savefig(plot_file_name)

        if display:
            plt.show()

    def plot_combined(self, display=True):
        legends = []

        for optimizer_directory_name in os.listdir(os.path.join(self.cur_dir, 'logs', 'loss', self.dataset)):
            optimizer_log_file = os.path.join(self.cur_dir, 'logs', 'loss', self.dataset, optimizer_directory_name, 'loss.csv')
            optimizer_loss = self.get_loss_from_file(optimizer_log_file)

            ax = sns.lineplot(x=list(range(len(optimizer_loss))), y=optimizer_loss)

            legends.append(optimizer_directory_name)
        
        ax.axes.set_title(f'{self.dataset} all optimizers', fontsize=20)
        ax.set_xlabel('epoch', fontsize=15)
        ax.set_ylabel('loss', fontsize=15)
        plt.legend(labels=legends)

        plot_file_name = os.path.join(self.plt_dir, f'{self.dataset}_all_plot.png')
        plt.savefig(plot_file_name)

        if display:
            plt.show()

    def get_loss_from_file(self, log_file):
        loss = []
        loss_logs = pd.read_csv(log_file)

        for i in range(len(loss_logs)):
            loss.append(loss_logs.loc[i, 'loss'])

        return loss