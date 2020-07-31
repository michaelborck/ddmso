from utils import  to_data_frame

PATH =  '/home/michael/Projects/ddmso/data/processed/visible/rgb/Twilight/'
ANNOTS = '/home/michael/Projects/ddmso/data/processed/visible/annots/'

df = to_data_frame(PATH,ANNOTS)
print(df.head())
