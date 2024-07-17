from time import sleep 

import airsim

from multirotor_session import MultirotorSession

# Some notes:
# -z is up

def get_images(directory, yaw, weather_param, weather_value):
    index = 10
    for start_x, start_y, start_z, end_x, end_y, end_z in [
        # Down up
        # (0, 0, 0, 0, 0, -40),
        # (-10, 0, 0, 0, 0, -40),
        # (10, 0, 0, 0, 0, -40),
        # (0, -10, 0, 0, 0, -40),
        # (0, 10, 0, 0, 0, -40),

        # # Back forward
        # (-10, 0, -10, 0, 0, -10),
        # (10, 0, -10, 0, 0, -10),
        # (0, -10, -10, 0, 0, -10),
        # (0, 10, -10, 0, 0, -10),
        
        # Down up
        (-10, -10, 0, -10, -10, -20),
        (-10, 0, 0, -10, 0, -20),
        (-10, 10, 0, -10, 10, -20),
        
        (0, -10, 0, 0, -10, -20),
        (0, 0, 0, 0, 0, -20),
        (0, 10, 0, 0, 10, -20),
        
        (10, -10, 0, 10, -10, -20),
        (10, 0, 0, 10, 0, -20),
        (10, 10, 0, 10, 10, -20),

    ]:
        if weather_param != None: multirotor.set_weather_parameter(weather_param, weather_value)
        multirotor.snap_image_along_path(
            start_x=start_x, 
            start_y=start_y, 
            start_z=start_z,
            end_x=end_x, 
            end_y=end_y, 
            end_z=end_z, 
            speed=5, 
            directory=f"{directory}/sample_{index}/rotation_{yaw}/"
        )
        index += 1

for yaw in [
    0,
    90,
    180,
    270,
]:
    for weather_param, weather_value, image_path in [
        # (airsim.WeatherParameter.Fog, 0.4, "images/fog-0.4"),
        # (airsim.WeatherParameter.Rain, 0.4, "images/rain-0.4"),
        # (airsim.WeatherParameter.Snow, 0.4, "images/snow-0.4"),
        (airsim.WeatherParameter.Dust, 0.4, "images/dust-0.4"),
        (airsim.WeatherParameter.MapleLeaf, 0.4, "images/mapleleaf-0.4"),
        (None, None, "images/normal"),
    ]:
        multirotor = MultirotorSession()
        multirotor.rotate_to_yaw_async(yaw)
        multirotor.reset_weather_parameters()
        multirotor.take_off()
        sleep(3)
        get_images(image_path, yaw, weather_param, weather_value)