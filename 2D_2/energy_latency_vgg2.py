import warnings
warnings.filterwarnings("ignore")
import numpy as np
import time, os
from Predict_CNN import *

arch_param = {}
alpha = {"wifi":283.17, "LTE":438.39, "3G":868.98}
beta = {"wifi":132.86, "LTE":1288.04, "3G":817.88}

Size = []		#same length as the following two arrays
Energy_dev = []		#0, (conv1, act1)*l1, pool1*bool, (conv2, act2)*l2, pool2*bool, (conv3, act3)*l3, pool3*bool, (conv4, act4)*l4, pool4*bool, (conv5, act5)*l5, pool5*bool, (fc6, act6), (fc7, act7), softmax
Latency_dev = []	# Same, all of length 14 for the vgg search space

index_list = []
corr_factor = 1.6


def get_sizes():
	#Fill sizes array
	temp_width = 224
	temp_filters = 3	#initial dimensions of the input image
	classes = 10		#number of neurons in last dense layer

	Size.append(temp_width*temp_width*temp_filters) # in bytes
	i = 0
	while i<5:
		output_size = temp_width*temp_width*arch_param['f%s' %(i+1)]*4 	#4 because now it's floating point
		Size.append(output_size)
		if arch_param['p'] == (i+1):
			Size.append(output_size)
		else:
			temp_width = temp_width/2
			output_size = temp_width*temp_width*arch_param['f%s' %(i+1)]*4 	#4 because now it's floating point
			Size.append(output_size)
		i = i+1
	if arch_param['fc1'] == 0:
		Size.append(output_size)
	else:
		output_size = arch_param['fc1']*4
		Size.append(output_size)
	if arch_param['fc2'] == 0:
		Size.append(output_size)
	else:
		output_size = arch_param['fc2']*4
		Size.append(output_size)
	'''if arch_param['fc3'] == 0:	#if using fc3
		Size.append(output_size)
	else:
		output_size = arch_param['fc3']*4
		Size.append(output_size)'''
	Size.append(classes*4)				#[input, conv, pool, conv, pool, conv, pool, conv, pool, conv, pool, fc, fc, #fc, output]
	#print(len(Size))


def evaluations_dev():	#Call Berken's models
	#On-device energy and latency for each conv block
	Energy_dev.append(0)
	Latency_dev.append(0)
	#first conv block is treated singularly because of the different size consideration of the input image (first element of Size[])
	power_conv = (Call_CNN_Power([Size[0], (arch_param['k1']**2)*arch_param['f1']]) + Call_Norm_Power(Size[1]/4) + Call_Relu_Power(Size[1]/4))/3	#avg power across 3 layers
	latency_conv = (Call_CNN_Latency([Size[0], (arch_param['k1']**2)*arch_param['f1']]) + Call_Norm_Latency(Size[1]/4) + Call_Relu_Latency(Size[1]/4))/corr_factor #latency for 3 layers conv, batch, activation
	Latency_dev.append(arch_param['l1']*latency_conv) #mseconds
	Energy_dev.append(arch_param['l1']*latency_conv*power_conv) #mJ
	#first conv. pool
	if arch_param['p'] != 1:
		power_pool = Call_Pool_Power([Size[1]/4, Size[2]/4])
		latency_pool = (Call_Pool_Latency([Size[1]/4, Size[2]/4]))/corr_factor
	else:
		power_pool = 0
		latency_pool = 0
	Latency_dev.append(latency_pool) #mseconds
	Energy_dev.append(latency_pool*power_pool) #mJ
	i = 1
	while i<5:
		#On-device power and latency of next conv. block
		power_conv = (Call_CNN_Power([(Size[2*i]/4), (arch_param['k%s' %(i+1)]**2)*arch_param['f%s' %(i+1)]]) + Call_Norm_Power(Size[2*i + 1]/4) + Call_Relu_Power(Size[2*i + 1]/4))/3
		latency_conv = (Call_CNN_Latency([(Size[2*i]/4), (arch_param['k%s' %(i+1)]**2)*arch_param['f%s' %(i+1)]]) + Call_Norm_Latency(Size[2*i + 1]/4) + Call_Relu_Latency(Size[2*i + 1]/4))/corr_factor
		Latency_dev.append(arch_param['l%s' %(i+1)]*latency_conv) #mseconds
		Energy_dev.append(arch_param['l%s' %(i+1)]*latency_conv*power_conv) #mJ
		#On-device power and latency of next pool
		if arch_param['p'] != (i+1):
			power_pool = Call_Pool_Power([Size[2*i + 1]/4, Size[2*i + 2]/4])
			latency_pool = (Call_Pool_Latency([Size[2*i + 1]/4, Size[2*i + 2]/4]))/corr_factor
		else:
			power_pool = 0
			latency_pool = 0
		Latency_dev.append(latency_pool) #mseconds
		Energy_dev.append(latency_pool*power_pool) #mJ
		i = i+1
	#On-device energy and latency for fc1
	if arch_param['fc1'] != 0:
		power_fc = (Call_Fc_Power([(Size[10]/4), (Size[11]/4)]) + Call_Relu_Power(Size[11]/4))/2
		latency_fc = Call_Fc_Latency([(Size[10]/4), (Size[11]/4)]) + Call_Relu_Latency(Size[11]/4)
	else:
		power_fc = 0
		latency_fc = 0
	Latency_dev.append(latency_fc)
	Energy_dev.append(latency_fc*power_fc)
	#On-device energy and latency for fc2
	if arch_param['fc2'] != 0:
		power_fc = (Call_Fc_Power([(Size[11]/4), (Size[12]/4)]) + Call_Relu_Power(Size[12]/4))/2
		latency_fc = Call_Fc_Latency([(Size[11]/4), (Size[12]/4)]) + Call_Relu_Latency(Size[12]/4)
	else:
		power_fc = 0
		latency_fc = 0
	Latency_dev.append(latency_fc)
	Energy_dev.append(latency_fc*power_fc)
	#On-device energy and latency for fc3
	'''if arch_param['fc3'] != 0:
		power_fc = (Call_Fc_Power([(Size[12]/4), (Size[13]/4)]) + Call_Relu_Power(Size[13]/4))/2
		latency_fc = Call_Fc_Latency([(Size[12]/4), (Size[13]/4)]) + Call_Relu_Latency(Size[13]/4)
	else:
		power_fc = 0
		latency_fc = 0
	Latency_dev.append(latency_fc)
	Energy_dev.append(latency_fc*power_fc)'''
	#On-device energy and latency for Prob
	power_fc = (Call_Fc_Power([(Size[12]/4), (Size[13]/4)]) + Call_Relu_Power(Size[13]/4))/2		#IMPORTANT, ADD 1 to size index if fc3 exists and vice versa
	latency_fc = Call_Fc_Latency([(Size[12]/4), (Size[13]/4)]) + Call_Relu_Latency(Size[13]/4)
	Latency_dev.append(latency_fc)
	Energy_dev.append(latency_fc*power_fc)

	#print(len(Energy_dev))
	#print(len(Latency_dev))


def identify_splits():
	index_list.append(0)		#The all-cloud solution
	
	i = 1
	while i<len(Size)-1:		#Because we already include the all-edge solution separately
		if Size[0]<Size[i]:		#Append only layers who offer size gains over the all-cloud solution
			i = i+1
			continue
		else:
			index_list.append(i)
			i = i+1
	index_list.append(13)		#The all-edge solution		IMPORTANT:EDIT ACCORDING TO WHETHER FC3 exists (14) or not (13)


def arrange_splits(tu, ALPHA, BETA):
	#print(index_list)
	Cumulative_latency = []
	Cumulative_energy = []
	for index in index_list:
		#add the computation latency and energy
		comm_latency = (Size[index]*8)/(tu*1000) 		#msecs
		comm_power = ALPHA*tu + BETA 					#mW
		comm_energy = comm_latency*comm_power/1000 		#mJ
		dev_latency = sum(Latency_dev[:index+1]) 		#msecs
		dev_energy = sum(Energy_dev[:index+1]) 			#mJ
		Cumulative_latency.append(dev_latency+comm_latency)
		Cumulative_energy.append(dev_energy+comm_energy)
		#print(index)
		#print(comm_latency)
		#print(dev_latency)
	edge_latency = Cumulative_latency[-1]
	edge_energy = Cumulative_energy[-1]
	#print(Cumulative_energy)
	#print(Cumulative_latency)
	#Sort according to best latency and energy
	zipped_latency = zip(index_list, Cumulative_latency)
	zipped_energy = zip(index_list, Cumulative_energy)
	zipped_latency = sorted(zipped_latency, key = lambda x:x[1])
	zipped_energy = sorted(zipped_energy, key = lambda x:x[1])
	index_list_sorted_latency = [x for x, y in zipped_latency]
	index_list_sorted_energy = [x for x, y in zipped_energy]

	#print(Size)
	#print(Latency_dev)
	#print(Cumulative_latency)
	#print(Energy_dev) 
	#print(Cumulative_energy)


	return edge_latency, edge_energy, index_list_sorted_latency, index_list_sorted_energy, sorted(Cumulative_latency), sorted(Cumulative_energy)


def evaluate(x, tu, technology):

	Size.clear()
	index_list.clear()
	Energy_dev.clear()
	Latency_dev.clear()

	i = 0
	while i<5:
		arch_param['f%s' %(i+1)] = int(x[i*3])
		arch_param['k%s' %(i+1)] = int(x[i*3 + 1])
		arch_param['l%s' %(i+1)] = int(x[i*3 + 2])
		i = i+1
	arch_param['p'] = int(x[15])
	arch_param['fc1'] = int(x[16])
	arch_param['fc2'] = int(x[17])
	#arch_param['fc3'] = int(x[18]) 

	if technology == 'wifi':
		ALPHA = alpha['wifi']
		BETA = beta['wifi']
	elif technology == 'LTE':
		ALPHA = alpha['LTE']
		BETA = beta['LTE']
	if technology == '3G':
		ALPHA = alpha['3G']
		BETA = beta['3G']

	get_sizes()			#sizes of feature maps in the network
	evaluations_dev()	#on-device evaluations to obtain latency and energy	
	

	assert len(Size) > 0, "Size does not contain any values!"
	assert len(Size) == len(Energy_dev), "Energy_dev and Size not same length!"
	assert len(Energy_dev) == len(Latency_dev), "Energy_dev and Latency_dev not same length!"

	identify_splits()
	edge_latency, edge_energy, index_latency, index_energy, values_latency, values_energy = arrange_splits(tu, ALPHA, BETA)

	return edge_latency, edge_energy, index_latency, index_energy, values_latency, values_energy
	#return 0,0,0,0









