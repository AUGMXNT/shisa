import pandas as pd
import glob

feather_files = glob.glob('*.feather')

for file_path in feather_files:
    print(file_path)
    data =  pd.read_feather(file_path)
    warnings_data = data[data['warning'] != ""]
    print(warnings_data)
    print()
