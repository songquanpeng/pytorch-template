from tensorboardX import SummaryWriter


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, image, step):
        """
        Add image summary.
        The image should be passed as a 3-dimension tensor of size [3, H, W].
        """
        self.writer.add_image(tag, image, step)
