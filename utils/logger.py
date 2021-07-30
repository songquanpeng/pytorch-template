import tensorflow as tf

is_tensorflow_v1 = tf.__version__.startswith('1.')


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        if is_tensorflow_v1:
            self.writer = tf.summary.FileWriter(log_dir)
        else:
            self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        if is_tensorflow_v1:
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.writer.add_summary(summary, step)
        else:
            with self.writer.as_default():
                tf.summary.scalar(tag, value, step=step)
                self.writer.flush()
