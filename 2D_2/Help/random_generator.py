
import pandas as pd
import os
import mobo
from Predict_CNN import *
import random 

#Choice possibilities 

f = [24, 36, 64, 96, 128, 256]
k = [3, 5, 7]
l = [1, 2, 3]
p =[1, 2, 3, 4, 5, 6]
fc1 = [256, 512, 1024, 2048, 4096, 8192]
fc2 = [0, 256, 512, 1024, 2048, 4096, 8192]

total_points = []
total_Error = []
total_Latency = []
total_Energy = []
for i in range(0,2):	#make (0, 300) for the full test
	sample = []
	for i in range(0, 5):
		sample.append(random.choice(f))
		sample.append(random.choice(k))
		sample.append(random.choice(l))
	sample.append(random.choice(p))
	sample.append(random.choice(fc1))
	sample.append(random.choice(fc2))
	
	print(sample)
	error = mobo.error(sample)	#error = 1
	energy = mobo.energy(sample)
	latency = mobo.latency(sample)

	total_points.append(sample)
	total_Error.append(error)
	total_Latency.append(latency)
	total_Energy.append(energy)

df = pd.DataFrame({'Point': total_points,
					'Error': total_Error,
					'Latency': total_Latency,
					'Energy': total_Energy})

writer = pd.ExcelWriter('random_measurements.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1', index = False)
writer.save() 

