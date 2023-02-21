__author__ = 'GalihSuyoga'

# import required module
import os
# blueprint for managing route to several pythonfile, jsonify to return json value, render_template to render html page
from flask import request, Blueprint, jsonify
# flasgger for api documentation
from flasgger import swag_from
from main.model.text_processing import Abusive, KamusAlay, TextLog, FileTextLog, RawText
# pandas for data manipulation
from main.cleanser import text_normalization_on_db_raw_data, predict_text, training_model_evaluate, \
    training_model_evaluate_tensor, cleanser_string_step, predict_neural_network_text, predict_LSTM, test_LSTM
from sqlalchemy import or_
import pandas as pd

import re

from main.model import db

# initializing front root for project asset and template
api = Blueprint('api', __name__, template_folder='templates', static_folder='assets')


@swag_from("docs/text_neural_network.yml", methods=['POST'])
@api.route('/text-neural-network', methods=['POST'])
def text_neural_network():
    text = request.form.get('text', '')
    text_clean = cleanser_string_step(text=text, step=3)

    json_response = {
        'status_code': 200,
        'raw_text': text,
        'sentimen': predict_neural_network_text(text_clean)
    }
    return jsonify(json_response)

@swag_from("docs/text_LSTM.yml", methods=['POST'])
@api.route('/text-LSTM', methods=['POST'])
def text_LSTM():
    text = request.form.get('text', '')
    text_clean = cleanser_string_step(text=text, step=3)
    json_response = {
        'status_code': 200,
        'raw_text': text,
        'sentimen': predict_LSTM(text_clean)
    }
    return jsonify(json_response)

# @swag_from("docs/text_pre_processing.yml", methods=['POST'])
# @api.route('/text-pre-processing', methods=['POST'])
# def text_pre_processing():
#     text = request.form.get('text', '')
#     print(text_normalization_on_db_raw_data())
#     json_response = {
#         'status_code': 200,
#         'raw_text': text,
#         'cleaned_text': (text)
#     }
#     return jsonify(json_response)
#
#
@swag_from("docs/start_training_save_model.yml", methods=['GET'])
@api.route('/start_training_save_model', methods=['GET'])
def ml_training():
    result = {}
    x=test_LSTM()
    y = training_model_evaluate()

    json_response = {
        'status_code': 200,
        'raw_text': 'text',
        'result': result
    }
    return jsonify(json_response)
#
# @swag_from("docs/text_input_raw_data.yml", methods=['POST'])
# @api.route('/text-input_raw_data', methods=['POST'])
# def text_input_raw_data():
#     description = "text sukses diinput"
#     http_code = 200
#     """get the file"""
#     file = request.files.get('text')
#     array_text = []
#
#     if file:
#         """split filename to get the file extension"""
#         array_name = file.filename.split(".")
#         file_ext = array_name[-1].lower()
#
#         """make sure it was tsv"""
#         if file_ext != "tsv":
#             """if it's not tsv"""
#             description = "file is not tsv"
#             http_code = 400
#         else:
#             """if tsv"""
#             """masukkan csv ke panda dataframe variabel data_frame"""
#             data_frame = pd.read_csv(file.stream, sep='\t', header=None)
#
#             # data_frame.to_sql('file_text_log', con=db.engine, if_exists='replace', index_label='id')
#             # print("data saved")
#             print(data_frame.head())
#
#             for index, data in data_frame.iterrows():
#
#                 duplicate = RawText.query.filter(RawText.kalimat == data[0], RawText.sentimen == data[1]).first()
#                 if duplicate is None:
#
#                     new_data = RawText(kalimat=data[0], sentimen=data[1])
#                     new_data.save()
#
#     else:
#         http_code = 400
#         description = "file not found"
#
#     json_response = {
#         'status_code': http_code,
#         'description': description,
#         'data': array_text
#     }
#     return jsonify(json_response)

@swag_from("docs/LSTM_file.yml", methods=['POST'])
@api.route('/LSTM-file', methods=['POST'])
def lstm_file():

    description = "text sukses diproses"
    http_code = 200
    """get the file"""
    file = request.files.get('text')
    limit = int(request.form.get('limit', 0))
    dict_text = {}

    if file:
        """split filename to get the file extension"""
        array_name = file.filename.split(".")
        file_ext = array_name[-1].lower()

        """make sure it was csv"""
        if file_ext != "csv":
            """if it's not csv"""
            description = "file is not csv"
            http_code = 400
        else:
            """if csv"""
            """masukkan csv ke panda dataframe variabel data_frame"""

            data_frame = pd.read_csv(file.stream, encoding='latin-1')
            # print(data_frame.head())
            # print(data_frame.head())

            if limit > 0:
                new_df = data_frame[{'Tweet'}][:limit]
                # print(new_df)
            else:
                new_df = data_frame[{'Tweet'}]
            # print(new_df)
            new_df['clean_tweet'] = new_df['Tweet'].apply(lambda x: cleanser_string_step(text=x, step=3))
            # print(new_df)
            new_df['sentimen'] = new_df['clean_tweet'].apply(lambda x: predict_LSTM(text=x))
            dict_text = new_df[{'Tweet', 'sentimen'}].to_dict('records')

    else:
        http_code = 400
        description = "file not found"

    json_response = {
        'status_code': http_code,
        'description': description,
        'data': dict_text
    }
    return jsonify(json_response)

@swag_from("docs/neural_network_file.yml", methods=['POST'])
@api.route('/neural_network-file', methods=['POST'])
def neural_network_file():

    description = "text sukses diproses"
    http_code = 200
    """get the file"""
    file = request.files.get('text')
    limit = int(request.form.get('limit', 0))

    dict_text = {}

    if file:
        """split filename to get the file extension"""
        array_name = file.filename.split(".")
        file_ext = array_name[-1].lower()

        """make sure it was csv"""
        if file_ext != "csv":
            """if it's not csv"""
            description = "file is not csv"
            http_code = 400
        else:
            """if csv"""
            """masukkan csv ke panda dataframe variabel data_frame"""

            data_frame = pd.read_csv(file.stream, encoding='latin-1')
            # print(data_frame.head())

            if limit > 0:
                new_df = data_frame[{'Tweet'}][:limit]
                # print(new_df)
            else:
                new_df = data_frame[{'Tweet'}]
            new_df['clean_tweet'] = new_df['Tweet'].apply(lambda x: cleanser_string_step(text=x, step=3))
            # print(new_df)
            new_df['sentimen'] = new_df['clean_tweet'].apply(lambda x: predict_neural_network_text(text=x))
            dict_text = new_df[{'Tweet', 'sentimen'}].to_dict('records')

    else:
        http_code = 400
        description = "file not found"

    json_response = {
        'status_code': http_code,
        'description': description,
        'data': dict_text
    }
    return jsonify(json_response)
