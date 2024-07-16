# -*- coding: utf-8 -*-
import datetime
import glob
import gzip
import os
import re
import shutil
import numpy as np
import pandas as pd
import lightgbm as lgb


def prepare_weather_data(input_dir, output_gzip_dir, output_result_dir):
    """
    output_gzip_dir = "Z:\Работа\DataScience\ГТЕ\Погода\Архивы\out"
    output_result_dir = "Z:\Работа\DataScience\ГТЕ\Погода\Архивы\datasets"
    input_dir =  "Z:\Работа\DataScience\ГТЕ\Погода\Архивы"
    """
    try:
        for filename in os.listdir(input_dir):
            if filename.endswith(".gz"):
                with gzip.open(input_dir + filename, "rb") as f_in:
                    with open(f"{output_gzip_dir}/{filename[:-3]}", "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
    except Exception as e:
        raise e

    try:
        for filename in os.listdir(output_gzip_dir):
            if filename.endswith(".xls"):
                df = pd.read_excel(f"{output_gzip_dir}/{filename}", skiprows=6)
                df.to_excel(f"{output_result_dir}/{filename}x")
    except Exception as e:
        raise e
