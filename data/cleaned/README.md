# /data/cleaned

This directory contains cleaned versions of the *given datasets* 
(which are stored in /data for compatibility). 

When adding new data to this directory, please add a corresponding entry below. 

# Directory Notes

The current recommended dataset is: `train-scaled.csv`. 

`train-interpolated.csv`: 
Consists of `train.csv`'s data, linearly interpolated to remove all NaN values. 
Values at the *beginning* of a dataset may still be NaN. 
If NaNs end up causing issues, I'd advise dropping all rows that have a NaN
value for the column in question. 

`train-scaled.csv`: 
Consists of `train.csv`'s data, linearly interpolated to remove all NaN values. 
(via the same process as `train-interpolated.csv`. After interpolation, data
was normalized via Justin's preprocessing code. 
