import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read data
data_2026 = pd.read_csv('2026-data.csv', index_col=0, low_memory=False)
data_2025 = pd.read_csv('2025-data.csv', index_col=0, low_memory=False)
data_2024 = pd.read_csv('2024-data.csv', index_col=0, low_memory=False)
data_2023 = pd.read_csv('2023-data.csv', index_col=0, low_memory=False)
data_2022 = pd.read_csv('2022-data.csv', index_col=0, low_memory=False)
data_2021 = pd.read_csv('2021-data.csv', index_col=0, low_memory=False)

# concat all data
data = pd.concat([data_2026, data_2025, data_2024, data_2023, data_2022, data_2021], ignore_index=True)

# delete unnecessary columns
 #   Column                                Non-Null Count   Dtype 
# ---  ------                                --------------   ----- 
#  0   Jurisdiction                          233727 non-null  str   
#  1   Kind                                  233727 non-null  str   
#  2   Display Key                           233727 non-null  str   
#  3   Lens ID                               233727 non-null  str   
#  4   Publication Date                      233727 non-null  str   
#  5   Publication Year                      233727 non-null  int64 
#  6   Application Number                    233727 non-null  str   
#  7   Application Date                      233727 non-null  str   
#  8   Priority Numbers                      233727 non-null  str   
#  9   Earliest Priority Date                233727 non-null  str   
#  10  Title                                 233727 non-null  str   
#  11  Abstract                              215279 non-null  str   
#  12  Applicants                            233620 non-null  str   
#  13  Inventors                             232441 non-null  str   
#  14  Owners                                34702 non-null   str   
#  15  URL                                   233727 non-null  str   
#  16  Document Type                         233727 non-null  str   
#  17  Has Full Text                         233727 non-null  str   
#  18  Cites Patent Count                    233727 non-null  int64 
#  19  Cited by Patent Count                 233727 non-null  int64 
#  20  Simple Family Size                    233727 non-null  int64 
#  21  Simple Family Members                 233727 non-null  str   
#  22  Simple Family Member Jurisdictions    233727 non-null  str   
#  23  Extended Family Size                  233727 non-null  int64 
#  24  Extended Family Members               233727 non-null  str   
#  25  Extended Family Member Jurisdictions  233727 non-null  str   
#  26  Sequence Count                        233727 non-null  int64 
#  27  CPC Classifications                   223646 non-null  str   
#  28  IPCR Classifications                  230760 non-null  str   
#  29  US Classifications                    1 non-null       object
#  30  NPL Citation Count                    233727 non-null  int64 
#  31  NPL Resolved Citation Count           233727 non-null  int64 
#  32  NPL Resolved Lens ID(s)               62881 non-null   str   
#  33  NPL Resolved External ID(s)           62881 non-null   str   
#  34  NPL Citations                         89044 non-null   str   
#  35  Legal Status                          233727 non-null  str  
data.drop(columns=[
	'Jurisdiction', 'Kind', 'Display Key', 'Lens ID', 'Application Number', 'Priority Numbers',
	'Inventors', 'Owners', 'URL', 'Document Type', 'Has Full Text', 'Cites Patent Count',
	'Simple Family Size', 'Simple Family Members', 'Simple Family Member Jurisdictions',
	'Extended Family Size', 'Extended Family Members', 'Extended Family Member Jurisdictions',
	'Sequence Count', 'IPCR Classifications', 'US Classifications', 'NPL Citation Count',
	'NPL Resolved Citation Count', 'NPL Resolved Lens ID(s)', 'NPL Resolved External ID(s)',
	'NPL Citations'
], inplace=True)

# show data info
print(data.info())
print(data.head())
print(data.tail())
print(data.describe())