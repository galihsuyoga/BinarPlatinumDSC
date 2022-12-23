__author__ = 'GalihSuyoga'

# import required module
import pandas as pd
import io

import numpy as np
from main.model import db

# blueprint for managing route to several pythonfile, jsonify to return json value, render_template to render html page
from flask import Blueprint, render_template, redirect, url_for, send_file
# initializing front root for project asset and template
front = Blueprint('front', __name__, template_folder='templates', static_folder='assets')


@front.route('/')
def index():
    return render_template('frontend/index.html')


# url redirection to swagger documentation
@front.route('/swagger_index', methods=['GET'])
def swagger_index():
    return redirect(f"{url_for('front.index')}docs/")
