import numpy as np
import tensorflow as tf
from pathlib import Path

from CMRSegment.segmentor import Segmentor


class TF1Segmentor(Segmentor):
    def __init__(self, model_path: Path, overwrite: bool = False):
        tf.compat.v1.reset_default_graph()
        self._sess = tf.compat.v1.Session()
        super().__init__(model_path, overwrite)

    def __enter__(self):
        self._sess.__enter__()
        self._sess.run(tf.compat.v1.global_variables_initializer())
        # Import the computation graph and restore the variable values
        saver = tf.compat.v1.train.import_meta_graph('{0}.meta'.format(self.model_path))
        saver.restore(self._sess, '{0}'.format(self.model_path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sess.close()
        self._sess.__exit__(exc_type, exc_val, exc_tb)

    def execute(self, phase_path: Path, output_dir: Path):
        """Segment a 3D volume cardiac phase from phase_path, save to output_dir"""
        raise NotImplementedError("Must be implemented by subclasses.")

    def run(self, image: np.ndarray) -> np.ndarray:
        """Call sess.run()"""
        raise NotImplementedError("Must be implemented by subclasses.")
