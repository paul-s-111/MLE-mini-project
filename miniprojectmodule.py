import pandas as pd
import numpy as np
from datetime import datetime


def make_submission_file(predictions, filename=None, path='submissions/'):
    '''
    Create a Kaggle submission file
    ===============================

    Creates valid submission file from a NumPy array.

    Parameters
    ----------
        predictions : (2880 x 3) NumPy array
            Predicted values, `predictions[i][j]` is the predicted power output
            of the `i + 1`-th power plant at the `i`-th time step.
        filename : str
            Name of the CSV file.
        path : str
            Path of the CSV file.
    
    Returns
    -------
    None.

    Raises
    ------
        ValueError
            If `predictions` are not a NumPy array of the required shape or  if
            the contents of `predictions` are invalid.
    '''
    if filename is None:
        filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S')+' Our team\'s submission.csv'
    
    # Check the type
    if not isinstance(predictions, np.ndarray):
        raise ValueError('Predictions have to be a NumPy array of shape (2880, 3).')
    
    # Check the shape
    if predictions.shape != (2880, 3):
        raise ValueError(f'Predictions are expected to have shape (2880, 3) but have shape {predictions.shape}.')

    # Check for NaNs
    if np.sum(np.isnan(predictions)) > 0:
        print(f'Predictions contain {np.sum(np.isnan(predictions))} NaN values.')

    # Check for negative values
    if np.sum(predictions < 0):
        print(f'Predictions contain {np.sum(predictions < 0)} negative values.')
    
    # Check for values larger than the nominal power
    if np.sum(predictions > 1.0):
        print(f'Predictions contain {np.sum(predictions > 1.0)} values larger than 1.0.')

    # Create Pandas DataFrame and save it to CSV
    index = pd.Index(pd.date_range(start='2014/1/1 0:00', end='2014/04/30 23:00', freq='H'), name='datetime')
    columns = ('power_produced_1', 'power_produced_2', 'power_produced_3')
    y_pred_df = pd.DataFrame(data=predictions, index=index, columns=columns)
    y_pred_df.to_csv(path + filename, float_format='%.6f')
    print('Submission CSV file created successfully!')
