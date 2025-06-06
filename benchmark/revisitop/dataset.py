import os
import pickle
import pandas as pd

DATASETS = ['roxford5k', 'rparis6k', 'revisitop1m']

def configdataset(dataset, dir_main):

    dataset = dataset.lower()

    if dataset not in DATASETS:    
        raise ValueError('Unknown dataset: {}!'.format(dataset))

    if dataset == 'roxford5k' or dataset == 'rparis6k':
        # cargar .pkl
        gnd_fname = os.path.join(dir_main, dataset, f'gnd_{dataset}.pkl')
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        cfg['gnd_fname'] = gnd_fname
        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'

        # cargar .csv
        csv_fname = os.path.join(dir_main, dataset, f'{dataset}_image_data.csv')
        if os.path.exists(csv_fname):
            df = pd.read_csv(csv_fname)
            cfg['image_to_place'] = dict(zip(df["image_name"], df["place"]))
        else:
            print(f"⚠️ Archivo CSV no encontrado: {csv_fname}")
            cfg['image_to_place'] = {}

    elif dataset == 'revisitop1m':
        cfg = {}
        cfg['imlist_fname'] = os.path.join(dir_main, dataset, f'{dataset}.txt')
        cfg['imlist'] = read_imlist(cfg['imlist_fname'])
        cfg['qimlist'] = []
        cfg['ext'] = ''
        cfg['qext'] = ''
        cfg['image_to_place'] = {}

    cfg['dir_data'] = os.path.join(dir_main, dataset)
    cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')

    cfg['n'] = len(cfg['imlist'])
    cfg['nq'] = len(cfg['qimlist'])

    cfg['im_fname'] = config_imname
    cfg['qim_fname'] = config_qimname

    cfg['dataset'] = dataset

    return cfg

def config_imname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])

def config_qimname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])

def read_imlist(imlist_fn):
    with open(imlist_fn, 'r') as file:
        imlist = file.read().splitlines()
    return imlist