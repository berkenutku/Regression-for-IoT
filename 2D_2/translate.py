def translate(x):
    #f [1,9] --> {24, 36, 48, 64, 96, 128, 256, 384, 512}
    #k [1,4]  --> {3, 5, 7, 11} 
    #l [1, 3]  --> same
    #p --> bool
    #fc [0-4] --> {0, 0.25, 0.5, 0.75, 1}
    filter_dict = {1: 36, 2: 64, 3: 96, 4: 128, 5: 256, 6: 384, 7: 512}
    filter_dict = {1: 24, 2: 36, 3: 48, 4: 64, 5: 96, 6: 128, 7: 256, 8: 384, 9: 512}
    kernel_dict = {1: 3, 2: 5, 3: 7, 4: 11}

    #fc_neurons_ratio = {0: 0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1}
    fc_neurons_values = {0: 0, 1: 100, 2: 256, 3: 512, 4: 1024, 5: 2048, 6: 4096, 7: 8192, 8: 16384}

    x_mapped = []
    i = 0

    if len(x)==22:
        while i<20:
            x_mapped.append(filter_dict[x[i]])
            x_mapped.append(kernel_dict[x[i+1]])
            x_mapped.append(x[i+2])
            x_mapped.append(x[i+3])
            i = i + 4
        x_mapped.append(fc_neurons_values[x[20]])
        x_mapped.append(fc_neurons_values[x[21]])

    elif len(x)==13:
        x_mapped.append(filter_dict[x[0]])
        x_mapped.append(kernel_dict[x[1]])
        x_mapped.append(x[2])
        while i<2:
            x_mapped.append(filter_dict[x[(i * 4) + 3]])
            x_mapped.append(kernel_dict[x[(i * 4) + 4]])
            x_mapped.append(x[(i*4) + 5])
            x_mapped.append(x[(i*4) + 6])
            i = i + 1
        x_mapped.append(fc_neurons_values[x[11]])
        x_mapped.append(fc_neurons_values[x[12]])


    return x_mapped

    
