import numpy as np
import pandas as pd

def preprocess(pdb2seq, prot2annot, goterms, gonames, train, valid, test, maxlen=300):
    """
    Preprocess the protein sequence data and assign GO term labels.
    """
    df = pd.DataFrame.from_dict(pdb2seq, orient='index').reset_index()
    df.columns = ['name', 'seq']
    
    df['length'] = df['seq'].map(lambda x: len(str(x)))
    df['seq'] = df['seq'].map(lambda x: ' '.join(list(x)))
    
    df['data_type'] = np.nan
    df['data_type'] = df['data_type'].astype('object')
    df.loc[df.name.isin(train), 'data_type'] = 'train'
    df.loc[df.name.isin(valid), 'data_type'] = 'valid'
    df.loc[df.name.isin(test), 'data_type'] = 'test'
    
    df = df[df['length'] <= maxlen]
    df = df.dropna()
    
    y = np.array([prot2annot[n]['mf'] for n in df.name])
    df['label'] = [goterms[x] for x in y.argmax(axis=1)]
    df['idx'] = [x for x in y.argmax(axis=1)]
    
    index_keep = np.where(y.sum(axis=1) == 1)[0]
    df = df.iloc[index_keep]
    y_cut = y[index_keep]
    
    df = df.reset_index(drop=True)
    
    counts = df.groupby(['label'])['seq'].count().reset_index().rename(columns={'seq': 'population_size'})
    df = pd.merge(df, counts, on='label').reset_index()

    df_grouped = df.groupby(["idx", "label"])[["name"]].count().reset_index()
    df_grouped = df_grouped.rename(columns={'name': 'population_size'}).sort_values("population_size")
    
    return df, y_cut, df_grouped

import numpy as np

def filter_and_label(df, prot2annot, goterms, gonames, col_cut=[257, 463, 214, 135]):
    """
    Filter and assign labels for specific GO-MF terms.
    """
    # Filter the dataframe based on idx values in col_cut
    df_filtered = df[df.idx.isin(col_cut)].reset_index(drop=True)
    
    # Update y_cut based on the filtered df's name
    y_cut = np.array([prot2annot[n]['mf'] for n in df_filtered.name])
    y_cut = y_cut[:, col_cut]
    
    # Assign the labels based on the filtered y_cut
    df_filtered['label'] = [goterms[x] for x in y_cut.argmax(axis=1)]
    df_filtered['goterm'] = [goterms[x] for x in y_cut.argmax(axis=1)]
    df_filtered['goname'] = [gonames[x] for x in y_cut.argmax(axis=1)]
    df_filtered['idx'] = [x for x in y_cut.argmax(axis=1)]
    
    return df_filtered, y_cut
