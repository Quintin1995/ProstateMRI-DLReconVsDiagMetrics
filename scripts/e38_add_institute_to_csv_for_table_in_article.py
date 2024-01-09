import pandas as pd
import re
from pathlib import Path

# Description
# This script is used to add the institute column to the csv file that is used to generate the table in the article.
# and also to rename the columns to the correct names.
# and also to filter the manufacturer column to only contain the manufacturer name so if it contains the word 'Philips' we only keep that word
# and also to rename the manufacturer name SIEMENS to Siemens
# and also to split the column 'in_plane_resolution' to only contain the first part of the string if we string split it on '\'
# and also to save the resuls as csv with sep = ';' to file called result_of_query_for_table_article1_processed.csv

def determine_institute(path):
    if re.search(r'\d+-M-\d+', path):
        return 'MHG'
    elif re.search(r'\d+-U-\d+', path):
        return 'UMCG'
    elif 'pat' in path:
        return 'RUMC'
    return 'Unknown'

if __name__ == '__main__':
    
    csv_root = Path('sqliteDB/result_of_query_for_table_article1.csv')

    # Read the csv with pandas
    df = pd.read_csv(csv_root, sep=';')

    # Add the institute column based on regex of the path column.
    df['institute'] = df['path'].apply(determine_institute)

    # check if we have any 'Uknown' institutes and print if it does becuase we need to add them manually
    if 'Unknown' in df['institute']:
        print('We have some unknown institutes, please add them manually')

    # lets rename the columns
    new_col_names = ['in_plane_resolution', 'slice_thickness', 'spacing_between_slices', 'number_of_averages', 'echo_train_length', 'manufacturer', 'scanner_models', 'path', 'institute']

    # rename the columns
    df.rename(columns=dict(zip(df.columns[0:], new_col_names)), inplace=True)

    # lets filter the manufacturer column to only contain the manufacturer name so if it contains the word 'Philips' we only keep that word
    df['manufacturer'] = df['manufacturer'].apply(lambda x: x.split()[0] if 'Philips' in x else x)
    
    # rename the manufacturer name SIEMENS to Siemens
    df['manufacturer'] = df['manufacturer'].apply(lambda x: 'Siemens' if 'SIEMENS' in x else x)

    # the column 'in_plane_resolution' is structured like: 0.34226191043853\0.34226191043853 of which we only want the first part if we string split it on '\'
    df['in_plane_resolution'] = df['in_plane_resolution'].apply(lambda x: x.split('\\')[0])

    # lets save the resuls as csv with sep = ';' to file called result_of_query_for_table_article1_processed.csv
    fname = 'sqliteDB/result_of_query_for_table_article1_processed.csv'
    df.to_csv(fname, sep=';', index=False)
    print(f'Saved the results to: {fname}')

    print(df.head())

    # lets print all unique values of the manufacturer column
    print(df['manufacturer'].unique())