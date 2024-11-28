# ready to run example: HongJun/Tasks/TestTask1/main.py
import os

import airsim

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, vehicle_name="UAV7")
client.armDisarm(True, vehicle_name="UAV7")

# Async methods returns Future. Call join() to wait for task to complete.
client.takeoffAsync(vehicle_name="UAV7").join()
client.moveToPositionAsync(-150.0, 10, -80, 15, vehicle_name="UAV7").join()

# take images
responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.DepthVis),
    airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True)])
print('Retrieved images: %d', len(responses))

# do something with the images
for response in responses:
    if response.pixels_as_float:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        airsim.write_pfm(os.path.normpath('/temp/py1.pfm'), airsim.get_pfm_array(response))
    else:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        airsim.write_file(os.path.normpath('/temp/py1.png'), response.image_data_uint8)
