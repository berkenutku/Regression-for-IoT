#import mobo
import modela_energy_latency_vgg as energy_latency_vgg
import CNN_VGG_CIFAR 

#x = [36, 5, 1, 128, 7, 2, 1, 64, 5, 1, 0, 256, 100]		#ALEXNET
#x = [36, 5, 2, 1, 96, 5, 1, 0, 384, 11, 3, 1, 128, 5, 1, 1, 128, 7, 3, 1, 4096, 1024]	#71.83%, 70.38%(dropout(0.2)), 76.34%(data augmentation), 79.88%(batch normalization), 71.38%(all)
#x = [36, 5, 1, 24, 3, 1, 128, 5, 1, 24, 5, 2, 24, 7, 2, 5, 512, 256]	#72.84%, 80.27%(data augmentation + batch normalization)

Modela = [24, 3, 2, 36, 3, 1, 128, 7, 1, 24, 5, 1, 96, 5, 1, 5, 512, 256]

acc1 = CNN_VGG_CIFAR.get_values(Modela)
print(acc1)


'''
tu = 10

index_latency, index_energy, values_latency, values_energy = energy_latency_vgg.evaluate(Modela, tu, 'wifi')
print(index_energy)
print(values_energy)
print(index_latency)
print(values_latency)
'''
