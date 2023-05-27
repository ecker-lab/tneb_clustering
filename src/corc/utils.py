from os.path import exists, join
import os
import pickle


def cond_mkdir(path):
    """ create a folder if there is none.

    Parameters
    ----------
    path : str
        path or foldername
    """
    if not exists(path):
        os.makedirs(path)


def save(data, filename, outdir):
    """ save data using pickle in folder under given name 

    Parameters
    ----------
    data : ndarray
        data that should be saved
    filename : str
        name under which data is saved
    outdir : str
        folder name where to save data
    """
    with open(join(outdir, f'{filename}.pkl'), 'wb') as f:
        pickle.dump(data, f)
    with open(join(outdir, f'{filename}.txt'), 'w') as f:
        f.write(f'{data}')