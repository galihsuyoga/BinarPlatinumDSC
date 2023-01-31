__author__ = 'GalihSuyoga'

# import required module
import os
# blueprint for managing route to several pythonfile, jsonify to return json value, render_template to render html page
from flask import request, Blueprint, jsonify
# flasgger for api documentation
from flasgger import swag_from
from main.model.text_processing import Abusive, KamusAlay, TextLog, FileTextLog
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