"""
  Multi-objective ebnn
  -- kandasamy@cs.cmu.edu
"""

# You should have one of the following two in this file:
#   1. objectives: A list of functions, each of which corresponds to an objective.
#   2. A function called compute_objectives which returns the value of all objectives when
#      called as a list, and an integer called num_objectives which indicates the number
#      of objectives.

# pylint: disable=invalid-name
import numpy as np
import os
import time
import CNN_VGG_CIFAR        
import math
#from translate import translate
import energy_latency_vgg

tu = 3 # Mbps throughput
technology = 'wifi' #Wifi and GPU and fingers crossed


def _get_coords(x):
  
  #return translate(x) #int data
  return (x)  #Discrete numeric data

def energy(x):
  return evaluate(_get_coords(x), 1)

def latency(x):
  return evaluate(_get_coords(x), 2)

def error(x):
  return error_fn(_get_coords(x))

#def memory(x):
#  return memory_fn(_get_coords(x))

def evaluate(x, obj):

  global index_latency
  global index_energy
  global values_latency
  global values_energy

  if obj == 2:
    return values_latency[0]

  index_latency, index_energy, values_latency, values_energy = energy_latency_vgg.evaluate(x, tu, technology)
  print(index_energy)
  print(values_energy)
  print(index_latency)
  print(values_latency)

  return values_energy[0]


def error_fn(x): #OD: remember there is an input argument here x

  #acc1 = CNN_alex.get_values(x) #if using alexnet search space
  acc1 = CNN_VGG_CIFAR.get_values(x) #if using vgg search space

  error1 = 100 - float(acc1)
  #print(error1)

  '''error1 = 1000*(error1 - 5.0)/(13.5 - 5.0)
  if error1 < 0:
    error1 = 0'''

  return error1



 
objectives = [error, energy, latency]


