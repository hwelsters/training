import os

import airsim
import numpy as np
import cv2

class MultirotorSession:
  def __init__(self):
    self.client = airsim.MultirotorClient()
    self.client.confirmConnection()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)
    self.client.simEnableWeather(True)

  def __del__(self):
    self.reset()
    self.reset_weather_parameters()
    self.client.armDisarm(False)
    self.client.enableApiControl(False)
    self.client.simEnableWeather(False)

  def rotate_to_yaw_async(self, yaw):
    self.client.rotateToYawAsync(yaw).join()

  def reset_weather_parameters(self):
    self.set_weather_parameter(airsim.WeatherParameter.Rain, 0)
    self.set_weather_parameter(airsim.WeatherParameter.Snow, 0)
    self.set_weather_parameter(airsim.WeatherParameter.MapleLeaf, 0)
    self.set_weather_parameter(airsim.WeatherParameter.RoadLeaf , 0)
    self.set_weather_parameter(airsim.WeatherParameter.Dust, 0)
    self.set_weather_parameter(airsim.WeatherParameter.Fog, 0)

  def set_segmentation_id(self, name, id):
    return self.client.simSetSegmentationObjectID(name, id, True)

  def start_recording(self):
    self.client.startRecording()

  def stop_recording(self):
    self.client.stopRecording()
  
  # ==============================
  # Convenience methods
  # ==============================
  def snap_image_at_position(self, x: float, y: float, z: float, speed: float, directory: str):
    """
    Flies to a specific position. Takes an image and then save it to a directory.

    Args:
      x (float): The x coordinate of the position
      y (float): The y coordinate of the position
      z (float): The z coordinate of the position
      speed (float): The speed at which to move to the position
      directory (str): The directory to save the image to
    """
    self.move_to_position(x, y, z, speed)
    self.hover()
    self.snap_image(directory)

  def snap_image_along_path(self, start_x, start_y, start_z, end_x, end_y, end_z, speed: float, directory: str):
    """
    Flies to a specific position. Takes an image and then save it to a directory.

    Args:
      start_x (float): The x coordinate of the starting position
      start_y (float): The y coordinate of the starting position
      start_z (float): The z coordinate of the starting position
      end_x (float): The x coordinate of the ending position
      end_y (float): The y coordinate of the ending position
      end_z (float): The z coordinate of the ending position
      speed (float): The speed at which to move to the position
      directory (str): The directory to save the image to
    """

    # Take 10 images along path
    current_x, current_y, current_z = start_x, start_y, start_z

    for i in range(10):
      current_x = start_x + (end_x - start_x) * i / 10
      current_y = start_y + (end_y - start_y) * i / 10
      current_z = start_z + (end_z - start_z) * i / 10

      self.snap_image_at_position(current_x, current_y, current_z, speed, directory + f"/{i}")

  def snap_image(self, directory: str):
    """
    A convenience method to take a single image and save it to a directory
    """
    responses = self.get_images()
    self.save_images(responses, directory)

  def reset(self):
    self.client.reset()

  def wait_key(self, message):
    airsim.wait_key(message)

  # ==============================
  # Movement methods
  # ==============================
  def take_off(self):
    self.client.takeoffAsync().join()

  def move_to_position(self, x, y, z, speed):
    self.client.moveToPositionAsync(x, y, z, speed).join()

  def hover(self):
    self.client.hoverAsync().join()



  # ==============================
  # Image methods
  # ==============================
  def get_images(self):
    """
    Get images from the drone's cameras
    """
    responses = self.client.simGetImages([
      airsim.ImageRequest("1", airsim.ImageType.Scene), # scene vision image in png format
      airsim.ImageRequest("1", airsim.ImageType.Segmentation), #get ground truth segmentation
    ])
    return responses
  
  def save_images(self, responses, directory):
    """
    Save images to a directory.
    """

    # Create path if it doesn't exist
    if not os.path.exists(directory):
      os.makedirs(directory)

    for index, response in enumerate(responses):
      filename = os.path.join(directory, f"{self.imagetype_to_string(response.image_type)}")
      self.save_response_to_file(response, filename)
  
  # ==============================
  # Helper methods
  # ==============================
  def imagetype_to_string(self, image_type):
    if image_type == airsim.ImageType.Scene: return "Scene"
    if image_type == airsim.ImageType.DepthPlanar: return "DepthPlanar"
    if image_type == airsim.ImageType.DepthPerspective: return "DepthPerspective"
    if image_type == airsim.ImageType.DepthVis: return "DepthVis"
    if image_type == airsim.ImageType.DisparityNormalized: return "DisparityNormalized"
    if image_type == airsim.ImageType.Segmentation: return "Segmentation"
    if image_type == airsim.ImageType.SurfaceNormals: return "SurfaceNormals"
    if image_type == airsim.ImageType.Infrared: return "Infrared"
    else: return "Unknown"

  def save_response_to_file(self, response, filename):
    if response.pixels_as_float:
      airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
    elif response.compress: #png format
      airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    else: #uncompressed array
      img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
      img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 4 channel image array H X W X 3
      cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png


  # ==============================
  # Weather methods
  # ==============================
  def set_weather_parameter(self, weather, amount):
    self.client.simSetWeatherParameter(weather, amount)
  
  def set_time_of_day(self, time_of_day):
    self.client.simSetTimeOfDay(time_of_day)


  # ==============================
  # Getter methods
  # ==============================
  def get_multirotor_state(self):
    state = self.client.getMultirotorState()
    return state
  
  def get_imu_data(self):
    imu_data = self.client.getImuData()
    return imu_data
  
  def get_barometer_data(self):
    barometer_data = self.client.getBarometerData()
    return barometer_data
  
  def get_magnetometer_data(self):
    magnetometer_data = self.client.getMagnetometerData()
    return magnetometer_data