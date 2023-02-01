__author__ = 'GalihSuyoga'

# import required module
import os
# blueprint for managing route to several pythonfile, jsonify to return json value, render_template to render html page
from flask import request, Blueprint, jsonify
# flasgger for api documentation
from flasgger import swag_from
from main.model.text_processing import Abusive, KamusAlay, TextLog, FileTextLog, RawText
# pandas for data manipulation
from main.cleanser import bersihkan_tweet_dari_file, bersihkan_tweet_dari_text
from sqlalchemy import or_
import pandas as pd

import re

from main.model import db

# initializing front root for project asset and template
api = Blueprint('api', __name__, template_folder='templates', static_folder='assets')

@swag_from("docs/text_pre_processing.yml", methods=['POST'])
@api.route('/text-pre-processing', methods=['POST'])
def text_pre_processing():
    text = request.form.get('text', '')
    json_response = {
        'status_code': 200,
        'raw_text': text,
        'cleaned_text': bersihkan_tweet_dari_text(text)
    }
    return jsonify(json_response)

@swag_from("docs/text_input_raw_data.yml", methods=['POST'])
@api.route('/text-input_raw_data', methods=['POST'])
def text_input_raw_data():
    description = "text sukses diinput"
    http_code = 200
    """get the file"""
    file = request.files.get('text')
    array_text = []

    if file:
        """split filename to get the file extension"""
        array_name = file.filename.split(".")
        file_ext = array_name[-1].lower()

        """make sure it was tsv"""
        if file_ext != "tsv":
            """if it's not tsv"""
            description = "file is not tsv"
            http_code = 400
        else:
            """if tsv"""
            """masukkan csv ke panda dataframe variabel data_frame"""
            data_frame = pd.read_csv(file.stream, sep='\t', header=None)

            # data_frame.to_sql('file_text_log', con=db.engine, if_exists='replace', index_label='id')
            # print("data saved")
            print(data_frame.head())

            for index, data in data_frame.iterrows():
                # if index < 20:
                #     print(data[1])
                duplicate = RawText.query.filter(RawText.kalimat == data[0], RawText.sentimen == data[1]).first()
                if duplicate is None:

                    new_data = RawText(kalimat=data[0], sentimen=data[1])
                    new_data.save()
            # query all abusive word to become dataframe
            # abusive_df = pd.read_sql_query(
            #     sql=db.select([Abusive.word]),
            #     con=db.engine
            # )
            # # query all alay word to become dataframe
            # alay_df = pd.read_sql_query(
            #     sql=db.select([KamusAlay.word, KamusAlay.meaning]),
            #     con=db.engine
            # )
            #
            # for index, row in data_frame.iterrows():
            #     array_text.append(bersihkan_tweet_dari_file(tweet=str(row['Tweet']), df_abusive=abusive_df,
            #                                                 df_alay=alay_df, full=row.to_dict()))
    else:
        http_code = 400
        description = "file not found"

    json_response = {
        'status_code': http_code,
        'description': description,
        'data': array_text
    }
    return jsonify(json_response)


