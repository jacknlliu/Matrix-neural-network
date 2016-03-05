import tensorflow as tf
import gzip
import os
import re
import sys
import tarfile

import tensorflow.python.platform

class InputData():

  def __init__(self, data_set_name, batch_size, data_dir):
    self.name = data_set_name
    self.batch_size = batch_size
    self.data_dir =data_dir
    
  def get_data(self):
    if self.name == "cifar10":
      return self.get_cifar10_batch_aug()
    elif self.name == "VOC":
      pass
      
  def get_cifar10_batch_aug(self):
    from tensorflow.models.image.cifar10 import cifar10_input
    images, labels = cifar10_input.distorted_inputs(self.data_dir, self.batch_size)
    
    return (images, labels)
    
  def get_VOC(self):
    pass

if __name__ == '__main__':
  get_data = InputData()
  get_data.get_data





