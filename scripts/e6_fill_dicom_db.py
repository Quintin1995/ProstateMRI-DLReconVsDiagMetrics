import os

from matplotlib.pyplot import table
from fastMRI_PCa.data import create_connection
from fastMRI_PCa.utils import print_p
import glob
import SimpleITK as sitk
import pandas as pd
from sqlite3 import Error


################################  README  ######################################
# NEW - 

KEY_HEADERS = [
    "0010|0020",
    "0018|0089",
    "0018|0093",
    "0008|103e",
    "0028|0010",
    "0028|0011",
    "0018|9087"
    ]


INCLUDE_TAGS = [    # Attributes
    "0008|0005",    # Specific Character Set
    "0008|0008",    # Image Type
    "0008|0012",    # Instance Creation Date
    "0008|0013",    # Instance Creation Time
    "0008|0016",    # SOP Class UID
    "0008|0018",    # SOP Instance UID
    "0008|0020",    # Study Date
    "0008|0021",    # Series Date
    "0008|0022",    # Acquisition Date
    "0008|0023",    # Content Date
    "0008|0030",    # Study Time
    "0008|0031",    # Series Time
    "0008|0032",    # Acquisition Time
    "0008|0033",    # Content Time
    "0008|0050",    # Accession Number
    "0008|0060",    # Modality
    "0008|0070",    # Manufacturer
    "0008|1010",    # Station Name
    "0008|1030",    # Study Description
    "0008|103e",    # Series Description
    "0008|1040",    # Institutional Department Name
    "0008|1090",    # Manufacturer's Model Name
    "0010|0020",    # Patient ID
    "0010|0030",    # Patient's Birth Date
    "0010|0040",    # Patient's Sex
    "0010|1010",    # Patient's Age
    "0010|1020",    # Patient's Size
    "0010|1030",    # Patient's Weight
    "0010|21b0",    # Additional Patient History
    "0012|0062",    # Patient Identity Removed
    "0012|0063",    # De-identification Method
    "0018|0015",    # Body Part Examined
    "0018|0020",    # Scanning Sequence
    "0018|0021",    # Sequence Variant
    "0018|0022",    # Scan Options
    "0018|0023",    # MR Acquisition Type
    "0018|0024",    # Sequence Name
    "0018|0050",    # Slice Thickness
    "0018|0080",    # Repetition Time
    "0018|0081",    # Echo Time
    "0018|0083",    # Number of Averages
    "0018|0084",    # Imaging Frequency
    "0018|0085",    # Imaged Nucleus
    "0018|0087",    # Magnetic Field Strength
    "0018|0088",    # Spacing Between Slices
    "0018|0089",    # Number of Phase Encoding Steps  IMPORTANT
    "0018|0091",    # Echo Train Length
    "0018|0093",    # Percent Sampling                IMPORTANT
    "0018|0094",    # Percent Phase Field of View
    "0018|1000",    # Device Serial Number
    "0018|1030",    # Protocol Name                   IMPORTANT -> sequence type
    "0018|1310",    # Acquisition Matrix              IMPORTANT
    "0018|1312",    # In-plane Phase Encoding Direction
    "0018|1314",    # Flip Angle
    "0018|1315",    # Variable Flip Angle Flag
    "0018|5100",    # Patient Position
    "0018|9087",    # Diffusion b-value               IMPORTANT
    "0020|000d",    # Study Instance UID
    "0020|000e",    # Series Instance UID
    "0020|0010",    # Study ID
    "0020|0032",    # Image Position (Patient)
    "0020|0037",    # Image Orientation (Patient)
    "0020|0052",    # Frame of Reference UID
    "0020|1041",    # Slice Location
    "0028|0002",    # Samples per Pixel
    "0028|0010",    # Rows                            IMPORTANT
    "0028|0011",    # Columns                         IMPORTANT
    "0028|0030",    # Pixel Spacing
    "0028|0100",    # Bits Allocated
    "0028|0101",    # Bits Stored
    "0028|0106",    # Smallest Image Pixel Value
    "0028|0107",    # Largest Image Pixel Value
    "0028|1050",    # Window Center
    "0028|1051",    # Window Width
    "0040|0244",    # Performed Procedure Step Start Date
    "0040|0254"     # Performed Procedure Step Description
    ]


################################################################################


def get_dict_from_dicom(reader, verbose=False):
    headers = {}
    for header_name in INCLUDE_TAGS:
        headers[header_name] = None

    for k in reader.GetMetaDataKeys():
        if k in INCLUDE_TAGS:
            v = reader.GetMetaData(k)
            headers[k] = f"{v}"
            if verbose:
                print_p(f"({k}) = \"{v}\"")
    headers["path"] = ""
    return headers


def has_different_key_headers(current_header_dict: dict, prev_header_dict):
    """ This function returns False if one of the key headers is different in 
    both dictionaries supplied as arguments.
    
    Parameters:
    `current_header_dict (dict)`: dict from dicom (Headers from DICOM)
    `prev_header_dict (dict)`: dict from dicom (Headers from DICOM)
    returns (bool): True if important headers are different, else False
    """
    for header in KEY_HEADERS:
        try:
            if current_header_dict[header] != prev_header_dict.get(header, None):
                return True
        except:
            continue
    return False


def is_patient_in_database(conn, tablename, patient):
    # Get all results from patient from database
    cur = conn.cursor()
    query = f"SELECT [0010|0020] FROM {tablename} WHERE [0010|0020] like '%{patient}%';"
    result = cur.execute(query).fetchall() #list of tuples
    if len(result) == 0:
        return False
    return True

    
def fill_dicom_table_RUMC_UMCG(
    tablename: str,
    database: str,
    patients_dir_RUMC: str,
    patients_dir_UMCG: str,
    devmode = False):
    """ Fills the given table with headers/tags from DICOM files from UMCG and
    RUMC. The tags are cross referenced with an include list of tags.
    
    Parameters:
    `tablename (string)`: table in sqlite that will be inserted into
    `database (string)`: relative project path to .db (database) file for sqlite.
    `patients_dir_RUMC (string)`: path where patient directories are stored (RUMC)
    `patients_dir_UMCG (string)`: path where patient directories are stored (UMCG)
    """

    # Connect with database
    db_path = f"{os.getcwd()}{database}"
    conn = create_connection(db_path)
    print_p(f"connection made: {db_path}")

    patients = os.listdir(patients_dir_UMCG) + os.listdir(patients_dir_RUMC)
    prev_headers = {}

    with conn:
        # Drop all rows from table if it exists.
        if False:
            conn.execute(f"DELETE FROM {tablename};")
            print_p("done deleting all records from database")

        # loop over all patients. (RUMC and UMCG)
        for p_idx, patient in enumerate(patients):

            print_p(f"\nPatient {p_idx}: {patient}")
            
            if is_patient_in_database(conn, tablename, patient):
                print_p(f"PATIENT IS ALREADY IN DATABASE {tablename}")
                continue

            print_p(f"patient: {patient} is not in database")
            # Find all DICOM files
            glob_pattern = f"data/raw/*/{patient}/**/*.dcm"
            dicoms_paths = glob.glob(glob_pattern, recursive=True)
            rows = []

            # Loop over DICOM files
            for f_idx, dcm_path in enumerate(dicoms_paths):
                if f_idx > 10 and devmode:
                    continue
                print_p(f"f{f_idx}", end=' ')

                try:
                    reader = sitk.ImageFileReader()
                    reader.SetFileName(dcm_path)
                    reader.LoadPrivateTagsOn()
                    reader.ReadImageInformation()

                except:
                    print_p(f"Read Image Information EXCEPTION... Skipping: {dcm_path}")
                    continue

                curr_headers = get_dict_from_dicom(reader, verbose=False)
                curr_headers['path'] = dcm_path
                if not has_different_key_headers(curr_headers, prev_headers):
                    continue
                prev_headers = curr_headers
                rows.append(curr_headers)

            df = pd.DataFrame.from_dict(rows, orient='columns')
            print_p(f"\nwriting headers to sqlite database. {tablename} - num rows: {len(rows)}")
            df.to_sql(name=tablename, con=conn, if_exists='append')
                
    print_p(f"\n--- DONE writing data to {tablename}---")


################################################################################
print_p("start script")
fill_dicom_table_RUMC_UMCG(
    tablename = "dicom_headers_v2",
    database = r"/sqliteDB/dicoms.db",
    patients_dir_RUMC = r"data/raw/RUMC",
    patients_dir_UMCG = r"data/raw/UMCG",
    devmode=False)