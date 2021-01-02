from Predict_CNN import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

Latency_dev = []
Energy_dev = []

width = 224

power_conv = (Call_CNN_Power([width*width*3, 49*128]) + Call_Norm_Power(width*width*128) + Call_Relu_Power(width*width*128))/3	#avg power across 3 layers
latency_conv = Call_CNN_Latency([width*width*3, 49*128]) + Call_Norm_Latency(width*width*128) + Call_Relu_Latency(width*width*128) #latency for 3 layers conv, batch, activation
Latency_dev.append(2*latency_conv) #mseconds
Energy_dev.append(2*latency_conv*power_conv) #mJ
#first conv. pool

power_conv = (Call_CNN_Power([width*width*128, 9*384]) + Call_Norm_Power(width*width*384) + Call_Relu_Power(width*width*384))/3	#avg power across 3 layers
latency_conv = Call_CNN_Latency([width*width*128, 9*384]) + Call_Norm_Latency(width*width*384) + Call_Relu_Latency(width*width*384) #latency for 3 layers conv, batch, activation
Latency_dev.append(latency_conv) #mseconds
Energy_dev.append(latency_conv*power_conv) #mJ
#first conv. pool
power_pool = Call_Pool_Power([width*width*384, width*width*384/2])
latency_pool = Call_Pool_Latency([width*width*384, width*width*384/2])
width = width/2
Latency_dev.append(latency_pool) #mseconds
Energy_dev.append(latency_pool*power_pool) #mJ

power_conv = (Call_CNN_Power([width*width*384, 25*384]) + Call_Norm_Power(width*width*384) + Call_Relu_Power(width*width*384))/3	#avg power across 3 layers
latency_conv = Call_CNN_Latency([width*width*384, 25*384]) + Call_Norm_Latency(width*width*384) + Call_Relu_Latency(width*width*384) #latency for 3 layers conv, batch, activation
Latency_dev.append(latency_conv) #mseconds
Energy_dev.append(latency_conv*power_conv) #mJ

power_pool = Call_Pool_Power([width*width*384, width*width*384/2])
latency_pool = Call_Pool_Latency([width*width*384, width*width*384/2])
width = width/2
Latency_dev.append(latency_pool) #mseconds
Energy_dev.append(latency_pool*power_pool) #mJ

power_conv = (Call_CNN_Power([width*width*384, 9*512]) + Call_Norm_Power(width*width*512) + Call_Relu_Power(width*width*512))/3	#avg power across 3 layers
latency_conv = Call_CNN_Latency([width*width*384, 9*512]) + Call_Norm_Latency(width*width*512) + Call_Relu_Latency(width*width*512) #latency for 3 layers conv, batch, activation
Latency_dev.append(latency_conv) #mseconds
Energy_dev.append(latency_conv*power_conv) #mJ

power_conv = (Call_CNN_Power([width*width*512, 9*128]) + Call_Norm_Power(width*width*128) + Call_Relu_Power(width*width*128))/3	#avg power across 3 layers
latency_conv = Call_CNN_Latency([width*width*512, 9*128]) + Call_Norm_Latency(width*width*128) + Call_Relu_Latency(width*width*128) #latency for 3 layers conv, batch, activation
Latency_dev.append(latency_conv) #mseconds
Energy_dev.append(latency_conv*power_conv) #mJ


#On-device energy and latency for fc1
power_fc = (Call_Fc_Power([width*width*128, 512]) + Call_Relu_Power(512))/2
latency_fc = Call_Fc_Latency([width*width*128, 512]) + Call_Relu_Latency(512)
Latency_dev.append(latency_fc)
Energy_dev.append(latency_fc*power_fc)
#On-device energy and latency for fc2
power_fc = (Call_Fc_Power([512, 64]) + Call_Relu_Power(64))/2
latency_fc = Call_Fc_Latency([512, 64]) + Call_Relu_Latency(64)
Latency_dev.append(latency_fc)
Energy_dev.append(latency_fc*power_fc)
#On-device energy and latency for Prob
power_fc = (Call_Fc_Power([64, 10]) + Call_Relu_Power(10))/2
latency_fc = Call_Fc_Latency([64, 10]) + Call_Relu_Latency(10)
Latency_dev.append(latency_fc)
Energy_dev.append(latency_fc*power_fc)

print(Energy_dev)
print(Latency_dev)


