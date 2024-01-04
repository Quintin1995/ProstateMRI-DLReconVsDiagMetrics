import yaml
from typing import List
from collections import deque
from fastMRI_PCa.utils import print_p
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import uuid
import sqlite3
import os

def does_table_exist(tablename: str, db_path: str):

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    #get the count of tables with the name
    c.execute(f''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{tablename}' ''')

    does_exist = False

    #if the count is 1, then table exists
    if c.fetchone()[0] == 1:
        print(f"Table '{tablename}' exists.")
        does_exist = True
    else:
        print(f"Table '{tablename}' does not exists.")
                
    #commit the changes to db			
    conn.commit()
    #close the connection
    conn.close()
    return does_exist


def get_unique_fname():
    return str(uuid.uuid4().hex)


def list_from_file(path: str) -> List:
    """ Returns a list of all items on each line of the text file referenced in
        the given path
    
    Parameters:
    `path (str)`: path to the text file
    """
    print_p(path)
    return [line.strip() for line in open(path, "r")]


def dump_dict_to_yaml(
    data: dict,
    target_dir: str,
    filename: str = "settings",
    verbose: bool = True) -> None:
    """ Writes the given dictionary as a yaml to the target directory.
    
    Parameters:
    `data (dict)`: dictionary of data.
    `target_dir (str)`: directory where the yaml will be saved.
    `` filename (str)`: name of the file without extension
    """
    if verbose:
        print("\nParameters")
        for pair in data.items():
            print(f"\t{pair}")
        print()

    path = os.path.join(target_dir, f"{filename}.yml")
    with open(path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    print_p(f"Wrote yaml to: {path}")


def read_yaml_to_dict(path: str) -> dict:
    with open(path, 'r') as stream:
        try:
            parsed_yaml=yaml.safe_load(stream)
            # print(parsed_yaml)
            return parsed_yaml
        except yaml.YAMLError as exc:
            print(exc)


def create_dirs_if_not_exists(dirs: List[str]) -> None:
    """ Creates the list of supplied directories if they don't exist yet.
    
    Parameters:
    `dirs (List[str]) or str`: list of strings representing the directories that
    need to be created
    """
    
    if isinstance(dirs, str):
        dirs = [dirs]

    for folder in dirs:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print_p(f"Created directory {folder} or it already existed.")


def save_np_to_nifti(data_np: np.ndarray,
                    target_dir: str,
                    fname: str,
                    target_space: List[float] = [0.5, 0.5, 3.0]):

    img_s = sitk.GetImageFromArray(data_np.T)
    img_s.SetSpacing(target_space)
    path = f"{target_dir}/{fname}.nii.gz"
    sitk.WriteImage(img_s, path)
    print_p(f"Saved numpy array to nifti: {path}")