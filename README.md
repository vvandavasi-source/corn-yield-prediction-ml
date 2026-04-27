# Machine Learning for Corn Yield Prediction 
County-level corn yield prediction using remote sensing and machine learning
This project is broken into a few parts.
1.)Getting the yield data. Our yield data was collected from the USDA NASS Website.
2.)Getting the GEE data. We created code that parses the corn belt states from 2008-2024 using Modis satelite imagery. This data is collected in 16 day intervals from day 65-273. That code can be found in the file labeled GEE Code. This can be copy and pasted into Google Earth Engine and Ran!
3.)The yield file from 2008-2024 and the gee files from each state for 2008-2024 can be found in the zip file labeled final lagged cdls and yield file. please download this zip file to be able to run the model if interested. 
4.) The code and its outputs can be found in in the file offline_proper_cdl.ipynb making this replicatable.
