import numpy as np
import random
import math
import matplotlib.pyplot as plt

data_size = 10000
aoi_size = 100
sensors = 16

# create a dataset
input = np.zeros((data_size, aoi_size, aoi_size))
output = np.zeros((data_size, aoi_size, aoi_size))
sensor_values = np.zeros((data_size, sensors+1))
# 9 sensors evenly distributed
sensor_locs = np.array([[int(x),int(y)] for x in [aoi_size*0.2, aoi_size*0.4, aoi_size*0.6, aoi_size*0.8] 
                        for y in [aoi_size*0.2, aoi_size*0.4, aoi_size*0.6, aoi_size*0.8]])
print(sensor_locs)

fire_loc_x = []
fire_loc_y = []

def generate_sensor_data(input, output, sensor_values, sensor_locs, aoi_size, data_size):
    t = 0
    for i in range(data_size):
        if t==0:
            fire_loc_x = random.randint(0,aoi_size-1)
            fire_loc_y = random.randint(0,aoi_size-1)
            # input[i][fire_loc_x][fire_loc_y] = 1
            image = input[i].copy()
            for a in range(aoi_size):
                for b in range(aoi_size):
                    if a == fire_loc_x and b == fire_loc_y:
                        pass
                    else: image[a][b] = 1/(math.dist([fire_loc_x, fire_loc_y], [a+(0.5*aoi_size/100), b+(0.5*aoi_size/100)])**2)
            output[i] = image
            for l in range(sensor_locs.shape[0]):
                sensor_values[i][l] = output[i][sensor_locs[l][0]][sensor_locs[l][1]]
            sensor_values[i][-1] = t
            print("Starting from t=0")
            t = 1
        else:
            if np.count_nonzero(image == 1) < (aoi_size**2) * 0.8:
                ## dissipate fire for 50 time units
                for z in range(20): 
                    t += 1
                    high_val = 1
                    high_val_2 = 0
                    for a in range(aoi_size):
                        for b in range(aoi_size):
                            if a == fire_loc_x and b == fire_loc_y:
                                pass
                            else:
                                if image[a][b] != high_val and image[a][b]>high_val_2:
                                    high_val_2 = image[a][b]
                    ## most recent places where fire was spread
                    dissipation_indices = np.argwhere(image == high_val_2)
                    # print(dissipation_indices)
                    for d in dissipation_indices:
                        image[d[0]][d[1]] = high_val
                # store dataset after every 50 time units
                # output is the dissipated fire
                output[i] = image
                # input[i][fire_loc_x][fire_loc_y] = 1
                fire_indices = np.argwhere(image == high_val)
                # input is the lowest distant fire from the sensor
                for l in range(sensor_locs.shape[0]):
                    min_dist=1000
                    for d in fire_indices:
                        if min_dist > math.dist(sensor_locs[l],d):
                            min_dist = math.dist(sensor_locs[l],d)
                    # if min_dist==0: sensor_values[i][l] = 1 
                    # else: sensor_values[i][l] = 1/(min_dist**2) 
                    if min_dist==0: 
                        sensor_values[i][l] = 1 
                        input[i][sensor_locs[l][0]][sensor_locs[l][1]] = 1
                    else: 
                        sensor_values[i][l] = 1/(min_dist**2) 
                        input[i][sensor_locs[l][0]][sensor_locs[l][1]] = 1/(min_dist**2) 
                sensor_values[i][-1] = t
                #print("Time: ",t)
            else:   
                print(f"Time out at {i}, starting new fire")
                t = 0
        # plt.imshow(input[i], cmap='hot', interpolation='nearest')
        # plt.pause(0.0000001)
    return input, output, sensor_values


input, output, sensor_values = generate_sensor_data(input, output, sensor_values, sensor_locs, aoi_size, data_size)
# plt.show()
np.save("input_wofire_16.npy", input)
np.save("output_wofire_16.npy", output)
np.save("sensor_values_wofire_16.npy", sensor_values)


                    
# def fire_dissipation(image, fire_loc_x, fire_loc_y, aoi_size):
#     while np.count_nonzero(image == 1) < (aoi_size**2) * 0.8:
#         t += 1
#         high_val = 1
#         high_val_2 = 0
#         for a in range(aoi_size):
#             for b in range(aoi_size):
#                 if a == fire_loc_x and b == fire_loc_y:
#                     pass
#                 else:
#                     if image[a][b] != high_val and image[a][b]>high_val_2:
#                         high_val_2 = image[a][b]
#         dissipation_indices = np.argwhere(image == high_val_2)
#         # print(dissipation_indices)
#         for d in dissipation_indices:
#             image[d[0]][d[1]] = high_val
#         if t%50==0:
#             plt.imshow(image, cmap='hot', interpolation='nearest')
#             plt.pause(0.0000001)
#     print("Time: ", t)           
    
    


# # Create a heatmap that dissipates heat over time until 80% of the area is covered with fire
# def fire_dissipation(image, fire_loc_x, fire_loc_y, aoi_size):
#     t = 0
#     image[fire_loc_x][fire_loc_y] = 1
#     for a in range(aoi_size):
#         for b in range(aoi_size):
#             if a == fire_loc_x and b == fire_loc_y:
#                 pass
#             else: image[a][b] = 1/(math.dist([fire_loc_x, fire_loc_y], [a+(0.5*aoi_size/100), b+(0.5*aoi_size/100)])**2)
#     while np.count_nonzero(image == 1) < (aoi_size**2) * 0.8:
#         t += 1
#         high_val = 1
#         high_val_2 = 0
#         for a in range(aoi_size):
#             for b in range(aoi_size):
#                 if a == fire_loc_x and b == fire_loc_y:
#                     pass
#                 else:
#                     if image[a][b] != high_val and image[a][b]>high_val_2:
#                         high_val_2 = image[a][b]
#         dissipation_indices = np.argwhere(image == high_val_2)
#         # print(dissipation_indices)
#         for d in dissipation_indices:
#             image[d[0]][d[1]] = high_val
#         if t%50==0:
#             plt.imshow(image, cmap='hot', interpolation='nearest')
#             plt.pause(0.0000001)
#     print("Time: ", t)
        

# fire_dissipation(input[0], fire_loc_x, fire_loc_y, aoi_size)
# plt.show()
# plt.pause(3)
# plt.close()