import zipfile
from pathlib import Path
import json
import imageio.v2 as imageio


def read_file(zfile:zipfile.PyZipFile, file_path:str | Path):
    try:
        return _read_file(zfile=zfile, file_path=file_path)
    except Exception as e:
        print(f"Couldnt read {file_path}")
        print(f"Exception: {e}")
        return None
def _read_file(zfile:zipfile.PyZipFile, file_path:str | Path):
    with zfile.open(file_path) as myfile:
        file_type = str(file_path).split(".")[-1]
        if file_type == "json":
            content = myfile.read().decode('utf-8')
            # Load the JSON content
            json_content = json.loads(content)
            return json_content
        
        elif file_type == "png":
            im = imageio.imread(myfile)
            return im

        else:
            raise Exception(f"Didnt recognize {file_type}. Must be either png or json")