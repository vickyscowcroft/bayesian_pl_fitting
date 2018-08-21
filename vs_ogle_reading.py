from astropy.io import ascii

ogle_raw = ascii.read(data_dir + 'ogle_lmc_cepheid_cat')
ogle_raw.convert_bytestring_to_unicode() ### some of the columns read in as bytestrings instead of strings. I think it was an astroconda update.

ogle_raw_temp = ogle_raw.to_pandas()  # you can convert directly from an astropy table to a pandas df
ogle_raw_df = ogle_raw_temp[['Star', 'Mode', '_RAJ2000', '_DEJ2000', '<Imag>', '<Vmag>', 'Per', 'e_Per']]
ogle_raw_df = ogle_raw_df.iloc[2:] ## scrapping the first two lines - remnants of the header
ogle_raw_df.reset_index(drop=True, inplace=True) ## resetting the index because i dropped the first two lines
ogle_raw_df.rename(columns = {'Star': 'ID', 'Mode': 'type', '_RAJ2000': 'ra', '_DEJ2000' : 'dec',  '<Imag>' : 'm_I', '<Vmag>': 'm_V', 'Per': 'P', 'e_Per' : 'P_sigma'}, inplace=True) ## renaming the columns to match your naming scheme.

num_cols = ['ra', 'dec', 'm_I', 'm_V', 'P', 'P_sigma'] ## numeric columns - need to convert to floats
for cols in num_cols:
    ogle_raw_df[cols] = pd.to_numeric(ogle_raw_df[cols], errors='coerce') ## errors='coerce' sets invalid reads to NaN, means that missing data is flagged properly


#### Only thing below here in batdog.ipynb that I changed was commenting out your reading into ogle_raw_df. I left everything from where you were putting things into ogle_cepheids onwards.

## plots make a nice PL now!!


