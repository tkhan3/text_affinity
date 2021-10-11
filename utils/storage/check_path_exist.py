import os
import configparser
from pathlib import Path

def check_path_exist(check_path):

    output = dict()
    path = Path(check_path)

    if path.exists():
        output['status_code'] = 0
        output['path'] = path
    else:
        output['status_code'] = 1

    return output
