# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:12:09 2020

@author: lesis
"""

import os
import sys
from flask import Flask, request, jsonify, redirect, url_for, flash
from helper import allowed_file, load_image_into_numpy_array, get_num_classes, run_inference_for_single_image
from werkzeug.utils import secure_filename
import tensorflow as tf
import glob
import numpy as np
from PIL import Image
from io import StringIO

sys.path.append("..")
from object_detection.utils import label_map_util


app = Flask(__name__)

UPLOAD_FOLDER = '/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(12)


@app.route('/upload_file', methods = ["POST"])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    # if user does not select file or submits an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save("images/"+ filename)
        return redirect(url_for('detect'))
    

@app.route('/detect_image')
def detect():

    PATH_TO_CKPT = 'models/research/fine_tuned_model/frozen_inference_graph.pb'
    TEST_IMAGE_PATHS = glob.glob(os.path.join('images', "*.*"))
    label_map_pbtxt_fname = "object_detection_twitter/data/annotations/label_map.pbtxt"
    
    num_classes = get_num_classes(label_map_pbtxt_fname)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map_pbtxt_fname = 'object_detection_twitter/data/annotations/label_map.pbtxt'
    PATH_TO_LABELS = label_map_pbtxt_fname


    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    
    cordinates = []
    for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        cordinates.append(output_dict)
    boxes = output_dict['detection_boxes']
    max_boxes_to_draw = boxes.shape[0]
    scores = output_dict['detection_scores']
    min_score_thresh=.5

    cordinates = []
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        
        if scores is None or scores[i] > min_score_thresh:
            class_name = category_index[output_dict['detection_classes'][i]]['name']
            cordinates.append({"Array of bounding boxes":{"cordinates":boxes[i].tolist(),
            "detection_classes": (output_dict['detection_classes'][i].tolist())}})

            
    return jsonify(cordinates)


if __name__ =='__main__':
    app.run()
