import os
from benchmark.revisitop.dataset import configdataset
import numpy as np
from benchmark.revisitop.evaluate import compute_map

def run_evaluation(
        df_result, 
        descriptors_final,   
        dataset: str = 'rparis6k',   # Set test dataset: roxford5k | rparis6k
):
    DATASETS_PATH = os.path.abspath("datasets")

    #---------------------------------------------------------------------
    # Evaluate
    #---------------------------------------------------------------------

    print('>> {}: Evaluating test dataset...'.format(dataset)) 
    # config file for the dataset
    # separates query image list from database image list, when revisited protocol used
    cfg = configdataset(dataset, DATASETS_PATH)


    # Extract query results
    query_image_names = np.array(cfg['qimlist']) + cfg['ext']
    q_idx_map = {name: idx for idx, name in enumerate(query_image_names)}
    df_result['q_img_id'] = np.nan
    df_result.loc[:, 'q_img_id'] = df_result.loc[:, 'image_name'].map(q_idx_map)
    df_result['q_img_id'] = df_result['q_img_id'].fillna(-1).astype(int)

    # Extract database results
    db_image_names = np.array(cfg['imlist']) + cfg['ext']
    db_idx_map = {name: idx for idx, name in enumerate(db_image_names)}
    df_result['db_img_id'] = np.nan
    df_result.loc[:, 'db_img_id'] = df_result.loc[:, 'image_name'].map(db_idx_map)
    df_result['db_img_id'] = df_result['db_img_id'].fillna(-1).astype(int)

    # load query and database features
    print('>> {}: Loading features...'.format(dataset))  

    mask_img_full = df_result['class_id'] == -1

    # Extract query descriptors
    mask_query = df_result['image_name'].isin(query_image_names)
    query_index = df_result[mask_query & mask_img_full].index
    Q = descriptors_final[query_index]

    # Extract database descriptors
    mask_db = df_result['image_name'].isin(db_image_names)
    db_index = df_result[mask_db & mask_img_full].index
    X = descriptors_final[db_index]

    # perform search
    print('>> {}: Retrieval...'.format(dataset))
    sim = np.dot(X, Q.T)
    ranks = np.argsort(-sim, axis=0)

    # Transform df_result ranks to db ranks
    ranks_id = db_index.values[ranks]
    final_ranks = df_result['db_img_id'].values[ranks_id]

    # revisited evaluation
    gnd = cfg['gnd']

    # evaluate ranks
    ks = [1, 5, 10]

    # search for easy
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(final_ranks, gnd_t, ks)

    # search for easy & hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(final_ranks, gnd_t, ks)

    # search for hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(final_ranks, gnd_t, ks)

    print('>> {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
    print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(dataset, np.array(ks), np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))