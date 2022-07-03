# INCell MxIF Quality Assessment [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository contains code linked to pre-publication work of ....
The code contained within has been developed to generate metrics and report the image quality coming from he Cytiva/Leica INCEll MxIF Microscope.


## Installation & Setup (MacOS/Linux)

Clone this Repository:

    git clone https://github.com/smujiang/MxIF_QC.git

Configure your Python enviroment: 
   
    #Check Version (> 3.7.x):
    python --version
    
    #Install Dependancies:
    pip install --upgrade -r requirements.txt 


# Usage

## Python Utility Script

### Compute metrics on images and create reports

__Filename:__ `run_qc.py`

__Purpose:__ This python script is intended to automatically evaluate MxIF image qualities using DAPI rounds as reference.
1. QC reports will be created, including json and html versions.
2. Intermedia evaluation metrics will be saved for advanced users.

__CLI:__ `python run_qc.py -i <path to processor dir> -o <path to output dir> -c <case id to be evaluated>`











