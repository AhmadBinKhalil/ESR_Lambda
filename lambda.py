from PIL import Image
import json
import tensorflow_hub as hub
import boto3
from io import BytesIO
import numpy as np
from tensorflow.image import decode_image
from tensorflow.io import read_file
from tensorflow import convert_to_tensor
from tensorflow.image import crop_to_bounding_box
from tensorflow import float32 
from tensorflow import expand_dims
from tensorflow import clip_by_value
from tensorflow import cast
from tensorflow import uint8
from tensorflow import squeeze

s3 = boto3.client('s3')

SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

def preprocess_image(image_path):
  """ Loads image from path and preprocesses to make it model ready
      Args:
        image_path: Path to the image file
  """
  hr_image = decode_image(read_file(image_path))
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = cast(hr_image, float32)
  return expand_dims(hr_image, 0)
              
def save_image(image, filename):
  """
    Saves Tensor Images.
    Args:
      image: 3D image tensor. [height, width, channels]
      filename: Name of the file to save.
  """
  if not isinstance(image, Image.Image):
    image = clip_by_value(image, 0, 255)
    image = Image.fromarray(cast(image, uint8).numpy())
  image.save("%s.jpg" % filename)
  print("Saved as %s.jpg" % filename)


def lambda_handler_BUCKET(event, context):
  # Retrieve the S3 bucket and object key from the event
  s3_bucket = event['Records'][0]['s3']['bucket']['name']
  s3_key = event['Records'][0]['s3']['object']['key']

  # Download the image from S3
  response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
  image_data = response['Body'].read()

  # Load the image into memory using Pillow ---------HERE
  image = Image.open(BytesIO(image_data))
  save_image(image, filename="upscaled_Img")

  hr_image = preprocess_image('upscaled_Img')
  model = hub.load(SAVED_MODEL_PATH)
  upscaled = model(hr_image)
  upscaled = squeeze(upscaled)
  save_image(squeeze(hr_image), filename="upscaled_Img")
  return {'body': 'Done',
  'code': 200}


def handler(event, context):
  hr_image = preprocess_image(event['Image_path'])
  model = hub.load(SAVED_MODEL_PATH)
  upscaled = model(hr_image)
  upscaled = squeeze(upscaled)
  save_image(squeeze(upscaled), filename="upscaled_Img")
  return {'body': 'Done', 'code': 200}


# curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"payload":"hello world!"}'
event = {'Image_path': 'C:\\Users\\Ahmad\\Desktop\\esr\\images.jpeg'}

handler(event, 'context')