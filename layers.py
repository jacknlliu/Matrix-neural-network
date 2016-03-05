import tensorflow as tf
import re

class Layer():
  # layer's input params:
  # input: a tensor input to this layer
  # layer_name: a string represent layer name
  # layer_attr: a dict that like this {"type":"matrix_layer", "params":[64, 64]}
  # reproduce: normally we multiply matrix Ai of every channel of input by the same method, B_i * A_i * C_i
  #            but this param is the numbers of B_i, C_i multiply the same channel A_i
  # combine: we can linear add the results of result of every channel transformation.

  TOWER_NAME = 'tower'
  @staticmethod
  def _activation_summary(x):
    """Helper to create summaries for activations.
  
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
  
    Args:
      x: Tensor
    Returns:
      nothing
    """

    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % Layer.TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
  
  @staticmethod
  def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
  
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
  
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer)
    return var
  
  @staticmethod
  def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
  
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
  
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
  
    Returns:
      Variable Tensor
    """
    var = Layer._variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd:
      weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)
    return var

  @staticmethod
  def repeat(x, l):
    '''Repeat a 2D tensor:
    if x has shape (m, n),
    the output will have shape (l, m, n)
    '''
    
    tensors = [x] * l
    stacked = tf.pack(tensors)
    
    return stacked
  
  @staticmethod
  def threee_tensor_mul(A, B, C, res):
    # for example 
    # A = tf.ones([4, 3, 2], tf.int32)
    # B = tf.ones([4, 2, 5, 3], tf.int32)
    # C = tf.ones([4, 5, 6], tf.int32)
    # return: (4, 3, 6) which combine 3 channel of matrix multiplication
    c = B.get_shape().as_list()[-1]
    res += tf.batch_matmul(tf.batch_matmul(A, B), C)
    
    return res
    
  @staticmethod
  def matrix_layer(input, layer_name, layer_attr, reproduce = None, combine = False):
    b, w, h, c = input.get_shape().as_list()
    dimL, dimR = layer_attr["params"]
    
    res = tf.zeros_initializer((b, dimL, dimR), dtype=tf.float32)
    with tf.variable_scope(layer_name) as scope:
      for i in range(c):
        scope_name = layer_name + "_" + str(i)
        with tf.variable_scope(scope_name) as scope:
          matrixL = Layer._variable_with_weight_decay('matrixL', shape=[dimL, h], stddev=1e-4, wd=0.0)
          matrixR = Layer._variable_with_weight_decay('matrixR', shape=[w, dimR], stddev=1e-4, wd=0.0)
          stackedL = Layer.repeat(matrixL, b)
          stackedR = Layer.repeat(matrixR, b)
          
          res = Layer.threee_tensor_mul(stackedL, input[:, :, :, i], stackedR, res)
          
      biases = Layer._variable_on_cpu('biases', [dimL, dimR], tf.constant_initializer(0.0))
      bias = res + biases
      act1 = tf.nn.relu(bias, name=scope.name) 
      Layer._activation_summary(act1)
    
    return act1
  
  @staticmethod
  def conv_relu(self, input, layer_name, kernel_shape, bias_shape):
    with tf.variable_scope(layer_name) as scope:
      # Create variable named "weights".
      weights = tf.get_variable("weights", kernel_shape,
          initializer=tf.random_normal_initializer())
      # Create variable named "biases".
      biases = tf.get_variable("biases", bias_shape,
          initializer=tf.constant_initializer(0.0))
      conv = tf.nn.conv2d(input, weights,
          strides=[1, 1, 1, 1], padding='SAME')
      return tf.nn.relu(conv + biases)
  
  @staticmethod
  def pool_layer(self):
    pass








