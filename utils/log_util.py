from torch.utils import tensorboard


class TbLogger:

    def __init__(self, log_dir):
        self.writer = tensorboard.SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_image(self, tag, image, step):
        self.writer.add_image(tag, image, step)

    def log_everything(self, scalars, step):
        for tag, value in scalars.items():
            self.log_scalar(tag, value, step)
