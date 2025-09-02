import json
import yaml
import csv
import cv2


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_text(path):
    with open(path, "r") as f:
        return f.read()

def load_cv2_image(path , format="RGB"):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img