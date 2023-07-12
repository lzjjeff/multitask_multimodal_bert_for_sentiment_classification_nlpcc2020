import os
from matplotlib import pyplot as plt


def update_lr(optimizer, learning_anneal):
    for g in optimizer.param_groups:
        g['lr'] = g['lr'] / learning_anneal


def my_plot(needed_types, x, y, x_tag="x", y_tag="y", output_dir=""):
    for data_type in needed_types:
        if len(x[data_type]) > len(y[data_type]):
            x[data_type] = x[data_type][:len(y[data_type])]
        elif len(x[data_type]) < len(y[data_type]):
            y[data_type] = y[data_type][:len(x[data_type])]
        plt.plot(x[data_type], y[data_type])
        plt.title("%s-%s curve of %s task" % (y_tag, x_tag, data_type))
        plt.xlabel(x_tag)
        plt.ylabel(y_tag)
        plt.savefig(os.path.join(output_dir, "%s_curve_of_%s.jpg" % (y_tag, data_type)))
        plt.show()
        plt.close()


def lr_plot(needed_types, num_idx, lrs, x_tag="num", output_dir=""):
    """
    绘制学习率-样本数曲线

    :param needed_types: 需要绘制学习率曲线的任务类型
    :param num_idx: 已处理的样本数
    :param lrs: 所有step的学习率
    :return: None
    """
    my_plot(needed_types, num_idx, lrs, x_tag, y_tag="lr", output_dir=output_dir)


def loss_plot(needed_types, num_idx, losses, x_tag="num", output_dir=""):
    """
        绘制损失值-样本数曲线

        :param needed_types: 需要绘制损失值曲线的任务类型
        :param num_idx: 已处理的样本数
        :param losses: 所有step的损失值
        :return: None
        """
    my_plot(needed_types, num_idx, losses, x_tag=x_tag, y_tag="loss", output_dir=output_dir)


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
