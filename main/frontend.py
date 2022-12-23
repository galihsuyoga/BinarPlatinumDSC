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
def index_page():
    return "index bro"
