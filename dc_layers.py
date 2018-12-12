import tensorflow as tf
from deepchem.models.tensorgraph.layers import Layer

# Layer to add a constant to all input elements.
class AddConstant(Layer):
    def __init__(self, offset, **kwargs):
        super(AddConstant, self).__init__(**kwargs)
        self._offset = offset
        try:
            self._shape = self.in_layers[0].shape
        except:
            pass

    def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
        inputs = self._get_input_tensors(in_layers)
        if len(inputs) != 1:
            raise ValueError("Must have one parent")
        parent_tensor = inputs[0]
        offset = tf.constant(self._offset, dtype=parent_tensor.dtype)
        out_tensor = tf.add(parent_tensor, offset)
        if set_tensors:
            self.out_tensor = out_tensor
        return out_tensor

# Layer to add batch indices in front of input indices.
# The resulting tensor can then be used as per-batch indices in a gather_nd
# operation (Gather layer).
# Input tensor shape: (batch_size, n_indices)
# Output tensor shape: (batch_size, n_indices + 1)
class InsertBatchIndex(Layer):
  def __init__(self, **kwargs):
    super(InsertBatchIndex, self).__init__(**kwargs)
    try:
      s = list(self.in_layers[0].shape)
      s[-1] += 1
      self._shape = tuple(s)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Must have one parent")
    parent_tensor = inputs[0]
    bi = tf.range(tf.shape(parent_tensor)[0], dtype=parent_tensor.dtype)
    out_tensor = tf.concat([tf.expand_dims(bi, 1), parent_tensor], axis=1)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor
