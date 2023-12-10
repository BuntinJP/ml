from feat import Detector
import os
import json
from pathlib import Path


def calc_feat_in_images(images_directory):
  feats = {}
  face_model = "retinaface"
  landmark_model = "mobilefacenet"
  au_model = "svm"
  au_model = 'xgb'
  emotion_model = "resmasknet"
  facepose_model = "img2pose"
  detector = Detector(
      face_model=face_model,
      landmark_model=landmark_model,
      au_model=au_model,
      emotion_model=emotion_model,
      facepose_model=facepose_model,
      verbose=True
  )
  for image_name in os.listdir(images_directory):
    if image_name.endswith('.jpg'):
      image_path = os.path.join(images_directory, image_name)
      try:
        result = detector.detect_image(image_path)
        feat = {
            "aus": result.aus.to_dict(),
            "emotions": result.emotions.to_dict(),
            "facepose": result.facepose.to_dict(),
            "facebox": result.facebox.to_dict(),
            "landmarks": result.landmarks.to_dict(),
        }
        feats[image_name] = feat
      except Exception as e:
        feats[image_name] = str(e)
  return feats


def calc_feat_in_dir(for_ml_directory):
  results = {}
  for dir_name in os.listdir(for_ml_directory):
    images_directory = os.path.join(for_ml_directory, dir_name, 'images')
    if os.path.isdir(images_directory):
      feat_data = calc_feat_in_images(images_directory)
      results[dir_name] = feat_data
  return results

def save_feats_to_json(feats, filepath):
  with open(filepath, 'w') as fp:
    json.dump(feats, fp)


def calculate_eye_areas_in_directory(for_ml_directory):
  results = {}
  for dir_name in os.listdir(for_ml_directory):
    images_directory = os.path.join(for_ml_directory, dir_name, 'images')
    if os.path.isdir(images_directory):
      feat_data = calc_feat_in_images(test_dir)
      results[dir_name] = feat_data
    else:
      print(f"Skipping {images_directory}")
  return results


def save_data_to_json(eye_areas, filepath):
  with open(filepath, 'w') as fp:
    json.dump(eye_areas, fp)


def load_data_from_json(filepath):
  try:
    with open(filepath, 'r') as fp:
      return json.load(fp)
  except FileNotFoundError:
    return None
def process_images_in_directory(directory_path):
  subdir_name = Path(directory_path).name
  json_filename = f"{subdir_name}_feat_data.json"
  results = load_data_from_json(json_filename)
  if not results:
    results = calc_feat_in_images(directory_path)
  save_data_to_json(results, json_filename)


def process_all_directories(base_directory):
  for subdir in Path(base_directory).iterdir():
    if subdir.is_dir():
      images_path = subdir.joinpath('images')
      print(f"Processing directory: {subdir}")
      process_images_in_directory(str(images_path))

base_directory = 'ForMachineLearning'
process_all_directories(base_directory)
