import pandas as pd

my_sheet = 'Amazon_Reviews_Keurig RAW'
file_name = 'Amazon_Reviews_Keurig RAW.xlsx' # name of the excel file
output_filename = 'Amazon_Reviews_Keurig RAW.csv'

# Select the reviews that are under 1024 character length, and random select top 100
df = pd.read_excel(file_name, sheet_name = my_sheet)
df['Field1'] = df['Field1'].astype('str')
mask = (df['Field1'].str.len() < 1024)
df = df.loc[mask]

df_elements = df.sample(n=100)
df_elements.to_csv(output_filename, encoding='utf-8', index=False)

print(df_elements.size)
print("randomized 100 output successfully")
    

