import os
import numpy as np # linear algebra
from numpy.matlib import repmat
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from fuzzywuzzy import fuzz
import datetime
import sklearn

#for image generation
from scipy import stats
from scipy.special import expit
import matplotlib.image as mpimg

import time
from tqdm import tqdm_notebook

import pickle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as kb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Activation, 
    BatchNormalization, 
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D
)

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical

import h5py

pd.options.display.max_rows = 500
pd.options.display.min_rows = 500
pd.options.display.max_columns = 500
pd.options.display.max_colwidth = 500

def clean_df(df):
    # first, re-map a few team names
    di = {"ARZ":"ARI", "BLT":"BAL", "CLV":"CLE", "HST":"HOU"}
    df = df.replace({'PossessionTeam':di, 'FieldPosition':di})
    di = {"ACE":"SINGLEBACK", np.nan:"NONE"}
    df = df.replace({'OffenseFormation':di})

    df = (df 
            .assign(ToLeft=df['PlayDirection']=='left')
           )

    df = df.assign(TeamOnOffense=np.where(df['PossessionTeam']==df['HomeTeamAbbr'],'home','away'))

    df = df.assign(Team=np.where(df['Team']==df['HomeTeamAbbr'],'home','away'))

    df = (df
            .assign(IsOnOffense=df['Team']==df['TeamOnOffense'])
            .assign(YardsFromOwnGoal=np.where(df['FieldPosition']==df['PossessionTeam'], df['YardLine'], 50 + (50-df['YardLine'])))
           )

    # standardize field positions
    df = (df
            .assign(YardsFromOwnGoal=np.where(df['YardLine']==50, 50, df['YardsFromOwnGoal']))
            .assign(X=np.where(df['ToLeft'], 120-df['X'], df['X'])-10)
            .assign(Y=np.where(df['ToLeft'], 53.33-df['Y'], df['Y']))
           )

    # standardize player directions (- to swtich from cw to ccw, + 90 to rotate so 0 = x-axis, -180 if going left to flip field)
    df = (df
            .assign(Dir=np.radians(np.where(~df['ToLeft'], -df['Dir'], -df['Dir']-180)+90))
           )
    
    # play duration so far
    df = df.assign(Duration=0)
    
    return df

#ADJUST FOR WHAT'S NEEDED

## helper codes to retrieve game state information
def personnel_features(df):
    pers_dict = {'GameId':[], 'PlayId':[], 'QB':[], 'RB':[], 'WR':[], 'TE':[], 'OL':[], 'DL':[], 'LB':[], 'DB':[]}
    df2 = df#.loc[(df['NflIdRusher'].isna() == False)].reset_index(drop=True)
    games = df2[['GameId', 'PlayId']].drop_duplicates().reset_index(drop=True)
    for idx, i in enumerate(games['PlayId']):
        pers_dict['GameId'].append(games['GameId'].iloc[idx])
        pers_dict['PlayId'].append(i)
        QB = len(df2.loc[(df2['GameId'] == games['GameId'][idx]) & (df2['PlayId'] == i) & (df2['Position'] == 'QB')])
        RB = len(df2.loc[(df2['GameId'] == games['GameId'][idx]) & (df2['PlayId'] == i) & ((df2['Position'] == 'RB') | (df2['Position'] == 'FB'))])
        WR = len(df2.loc[(df2['GameId'] == games['GameId'][idx]) & (df2['PlayId'] == i) & (df2['Position'] == 'WR')])
        TE = len(df2.loc[(df2['GameId'] == games['GameId'][idx]) & (df2['PlayId'] == i) & (df2['Position'] == 'TE')])
        OL = 11 - QB - RB - WR - TE
        DL = len(df2.loc[(df2['GameId'] == games['GameId'][idx]) & (df2['PlayId'] == i) & ((df2['Position'] == 'DT') | (df2['Position'] == 'DE') | (df2['Position'] == 'NT'))])
        LB = len(df2.loc[(df2['GameId'] == games['GameId'][idx]) & (df2['PlayId'] == i) & (df2['Position'].str.contains('LB'))])
        DB = 11 - DL - LB
        pers_dict['QB'].append(QB)
        pers_dict['RB'].append(RB)
        pers_dict['WR'].append(WR)
        pers_dict['TE'].append(TE)
        pers_dict['OL'].append(OL)
        pers_dict['DL'].append(DL)
        pers_dict['LB'].append(LB)
        pers_dict['DB'].append(DB)
        print(idx)
        
    personnel = pd.DataFrame(pers_dict).reset_index(drop=True)

    # Let's create some features to specify if the OL is covered
    personnel['OL_diff'] = personnel['OL'] - personnel['DL']
    personnel['OL_TE_diff'] = (personnel['OL'] + personnel['TE']) - personnel['DL']
    # Let's create a feature to specify if the defense is preventing the run
    # Let's just assume 7 or more DL and LB is run prevention
    personnel['run_def'] = (personnel['DL'] + personnel['LB'] > 6).astype(int)

    return personnel

def time_remaining(row):
    gc = row['GameClock']
    tmp = gc.split(':')[:-1]
    tr = (int(tmp[0])*3600) + (int(tmp[1]))
    tr = tr/3600/15
    return tr

def get_score_diff(row):
    if row['TeamOnOffense'] == 'home':
        scoreDiff = row['HomeScoreBeforePlay'] - row['VisitorScoreBeforePlay']
    else: 
        scoreDiff = row['VisitorScoreBeforePlay'] - row['HomeScoreBeforePlay']
    return scoreDiff

def one_hot_enc(df2, var):
    one_hot = pd.get_dummies(df2[var])
    df2 = (df2
              .drop(var, axis=1)
              .join(one_hot)
         )
    return df2

def generateImages(df,nx,ny,alpha):
    plays = df.iloc[:, :].groupby('PlayId')
    nPlays = plays.ngroups
    playDict = {}
    
    xg = np.linspace(0,120,nx)
    yg = np.linspace(0,53.3,ny)
    x, y = np.meshgrid(xg,yg)
    grid = np.stack((x, y), axis=-1)
    
    
    for playId, playData in tqdm_notebook(plays):
        playDensities = makeFields(playData,grid)
        playTensor = makeTensor(playDensities,alpha)
        if np.isnan(playTensor).any():
            print(playId)
        playDict[playId] = playTensor
        
    return playDict

def makeTensor(rho,alpha):
    #alpha should be in the range [1e-3, 1e2]
    #dens_list[0] = defense
    #dens_list[1] = offense
    #dens_list[2] = ball carrier
    
    rho_def = rho[0]/np.max(rho[0])*127
    rho_off = rho[1]/np.max(rho[1])*127
    rho_bc = rho[2]/np.max(rho[2])*127
    rho_comp = (expit(alpha*(rho_off-rho_def)))*127
    playTensor = np.stack([rho_def,rho_off,rho_bc,rho_comp], axis = -1)
    playTensor = playTensor.astype('int8')
    #converting to int8 to save memory
    
    return playTensor

def makeFields(df,grid): 
    ny, nx, _ = grid.shape
    rho_def = np.zeros((ny,nx))
    rho_off = np.zeros((ny,nx))
    rho_bc = np.zeros((ny,nx))
    
    for _, row in df.iterrows():
        pos = [row['X'],53.3 - row['Y']]
        spe = row['S']
        ori = row['Dir']
        
        if np.isnan(ori):
            ori = 0
            
        rho = dens(pos,spe,ori,grid)
   
        if row['IsOnOffense']:
            rho_off += rho
            if row['BallCarrier']:
                rho_bc += rho
        else:
            rho_def += rho
            
    return [rho_def,rho_off,rho_bc]

def dens(pos,spe,ori,grid):
    #need to convert units on parameters and estimate proper values for football vs. soccer
    roc = 4
    srat = spe**2/13**2
    
    R = np.array([[np.cos(ori),-np.sin(ori)], [np.sin(ori),np.cos(ori)]])
    S2 = np.array([[((roc-roc*srat)/2)**2,0],[0,((roc+roc*srat)/2)**2+1e-8]])
    sigma = np.matmul(np.matmul(R,S2),np.transpose(R))
    mu = (pos[0]+spe*np.cos(ori)*0.5, pos[1]+spe*np.sin(ori)*0.5)
    
    return stats.multivariate_normal.pdf(grid, mean = mu, cov = sigma)

def saveToFile(gameStates,yards,playIds,imageDict, prefix):
    n = gameStates.shape[0]
    print('Saving files for each play...')
    for ii in tqdm_notebook(range(n)):
        gameState = gameStates[ii,:]
        y = int(yards[ii]) 
        yvec = np.concatenate((np.zeros((1,y+99)),np.ones((1,100-y))), axis = 1)
        
        playId = playIds[ii]
        image = imageDict[playId]
        
        np.save(prefix + '/files/gameState'+str(playId)+'.npy',gameState)
        np.save(prefix + '/files/image'+str(playId)+'.npy',image)
        np.save(prefix + '/files/yardage'+str(playId)+'.npy',yvec)
        
    return 

def splitGS(gs, yard_var, epa_var):
    n = gs.shape[0]
    yards = gs[:,yard_var]
    pids = gs[:,-1].astype('int64')
    
    gs = np.delete(gs, -1, axis=1)
    gs = np.delete(gs, yard_var, axis=1)
    gs = np.delete(gs, epa_var, axis=1)
    gs[:,7] = (100-gs[:,7])/100.
        
    gs = gs.astype('float')
    return gs, yards, pids

def create_cnn(height, width, depth, filters=(16, 32, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1

    # define the model input
    inputs = Input(shape=inputShape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
            filt = (5,5)
        else:
            filt = (3,3)
        # CONV => RELU => BN => POOL
        
        x = Conv2D(f, filt, padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = BatchNormalization()(x)
    
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    
    
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(128)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, prefix, batch_size=32, imageDim=(100,200),
                 nChannels = 4, gsDim = 50, shuffle=True):
        'Initialization'
        self.prefix = prefix
        self.imageDim = imageDim
        self.gsDim = gsDim
        self.nChannels = nChannels
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        #Xcombined, yardage = self.__data_generation(list_IDs_temp)
        Xim, yardage = self.__data_generation(list_IDs_temp)

        #return Xcombined, yardage
        return Xim, yardage
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        Xim = np.empty((self.batch_size, *self.imageDim, self.nChannels))
        #Xgs = np.empty((self.batch_size, self.gsDim))
        yardage = np.empty((self.batch_size,199), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            temp = np.load(self.prefix + '/files/image' + str(int(ID)) + '.npy')
            Xim[i,] = temp.astype('float')/128.
            #Xgs[i,] = np.load(prefix + '/files/gameState' + str(int(ID)) + '.npy')
            
            # Store output
            yardage_temp = np.load(self.prefix + '/files/yardage' + str(int(ID)) + '.npy')
            yardage_temp_adj = []
            pl = 0
            for j in yardage_temp[0]:
                if (j == 1) & (pl == 0):
                    yardage_temp_adj.append(1)
                    pl+=1
                else:
                    yardage_temp_adj.append(0)
            yardage_temp_adj = np.array(yardage_temp_adj)
            yardage[i,] = yardage_temp_adj
            
        #return (Xgs, Xim), yardage  # Return a tuple instead of a list
        return Xim, yardage  # Return a tuple instead of a list


#Formation Clustering
### Define functions that generate spatial control fields for each play in input dataframe
def offense_locations(play_data):
    locations={'X':[], 'Y':[], 'Position':[], 'NflId':[]}

    center = play_data.loc[(play_data['Position'] =='C')].reset_index(drop=True)
    center = center.sort_values(by = 'Y').reset_index(drop=True)
    guard = play_data.loc[(play_data['Position'] =='G')].reset_index(drop=True)
    guard = guard.sort_values(by = 'Y').reset_index(drop=True)
    tackle = play_data.loc[(play_data['Position'] =='T')].reset_index(drop=True)
    tackle = tackle.sort_values(by = 'Y').reset_index(drop=True)
    qb = play_data.loc[(play_data['Position'] =='QB')].reset_index(drop=True)
    qb = qb.sort_values(by = 'Y').reset_index(drop=True)
    skills = play_data.loc[(play_data['Position'].isin(['QB', 'C', 'G', 'T']) == False)].reset_index(drop=True)
    skills = skills.sort_values(by = 'Y').reset_index(drop=True)
    offense_pl = pd.concat([center, guard]).reset_index(drop=True)
    offense_pl = pd.concat([offense_pl, tackle]).reset_index(drop=True)
    offense_pl = pd.concat([offense_pl, skills]).reset_index(drop=True)
    offense_pl = pd.concat([offense_pl, qb]).reset_index(drop=True)
    
    locations['X'].append(offense_pl['X'].iloc[0])
    locations['Y'].append(offense_pl['Y'].iloc[0])
    locations['Position'].append(offense_pl['Position'].iloc[0])
    locations['NflId'].append(offense_pl['NflId'].iloc[0])
    locations['X'].append(offense_pl['X'].iloc[1])
    locations['Y'].append(offense_pl['Y'].iloc[1])
    locations['Position'].append(offense_pl['Position'].iloc[1])
    locations['NflId'].append(offense_pl['NflId'].iloc[1])
    locations['X'].append(offense_pl['X'].iloc[2])
    locations['Y'].append(offense_pl['Y'].iloc[2])
    locations['Position'].append(offense_pl['Position'].iloc[2])
    locations['NflId'].append(offense_pl['NflId'].iloc[2])
    locations['X'].append(offense_pl['X'].iloc[3])
    locations['Y'].append(offense_pl['Y'].iloc[3])
    locations['Position'].append(offense_pl['Position'].iloc[3])
    locations['NflId'].append(offense_pl['NflId'].iloc[3])
    locations['X'].append(offense_pl['X'].iloc[4])
    locations['Y'].append(offense_pl['Y'].iloc[4])
    locations['Position'].append(offense_pl['Position'].iloc[4])
    locations['NflId'].append(offense_pl['NflId'].iloc[4])
    locations['X'].append(offense_pl['X'].iloc[5])
    locations['Y'].append(offense_pl['Y'].iloc[5])
    locations['Position'].append(offense_pl['Position'].iloc[5])
    locations['NflId'].append(offense_pl['NflId'].iloc[5])
    locations['X'].append(offense_pl['X'].iloc[6])
    locations['Y'].append(offense_pl['Y'].iloc[6])
    locations['Position'].append(offense_pl['Position'].iloc[6])
    locations['NflId'].append(offense_pl['NflId'].iloc[6])
    locations['X'].append(offense_pl['X'].iloc[7])
    locations['Y'].append(offense_pl['Y'].iloc[7])
    locations['Position'].append(offense_pl['Position'].iloc[7])
    locations['NflId'].append(offense_pl['NflId'].iloc[7])
    locations['X'].append(offense_pl['X'].iloc[8])
    locations['Y'].append(offense_pl['Y'].iloc[8])
    locations['Position'].append(offense_pl['Position'].iloc[8])
    locations['NflId'].append(offense_pl['NflId'].iloc[8])
    locations['X'].append(offense_pl['X'].iloc[9])
    locations['Y'].append(offense_pl['Y'].iloc[9])
    locations['Position'].append(offense_pl['Position'].iloc[9])
    locations['NflId'].append(offense_pl['NflId'].iloc[9])
    locations['X'].append(offense_pl['X'].iloc[10])
    locations['Y'].append(offense_pl['Y'].iloc[10])
    locations['Position'].append(offense_pl['Position'].iloc[10])
    locations['NflId'].append(offense_pl['NflId'].iloc[10])

    return locations

def defense_locations(play_data):
    locations={'X':[], 'Y':[], 'Position':[], 'NflId':[]}

    de = play_data.loc[(play_data['Position'] =='DE')].reset_index(drop=True)
    de = de.sort_values(by = 'Y').reset_index(drop=True)
    dt = play_data.loc[(play_data['Position'] =='DT')].reset_index(drop=True)
    dt = dt.sort_values(by = 'Y').reset_index(drop=True)
    nt = play_data.loc[(play_data['Position'] =='NT')].reset_index(drop=True)
    nt = nt.sort_values(by = 'Y').reset_index(drop=True)
    lb = play_data.loc[(play_data['Position'] =='LB')].reset_index(drop=True)
    lb = lb.sort_values(by = 'Y').reset_index(drop=True)
    olb = play_data.loc[(play_data['Position'] =='OLB')].reset_index(drop=True)
    olb = olb.sort_values(by = 'Y').reset_index(drop=True)
    mlb = play_data.loc[(play_data['Position'] =='MLB')].reset_index(drop=True)
    mlb = mlb.sort_values(by = 'Y').reset_index(drop=True)
    ilb = play_data.loc[(play_data['Position'] =='ILB')].reset_index(drop=True)
    ilb = ilb.sort_values(by = 'Y').reset_index(drop=True)
    cb = play_data.loc[(play_data['Position'] =='CB')].reset_index(drop=True)
    cb = cb.sort_values(by = 'Y').reset_index(drop=True)
    s = play_data.loc[(play_data['Position'] =='S')].reset_index(drop=True)
    s = s.sort_values(by = 'Y').reset_index(drop=True)
    db = play_data.loc[(play_data['Position'] =='DB')].reset_index(drop=True)
    db = db.sort_values(by = 'Y').reset_index(drop=True)
    ss = play_data.loc[(play_data['Position'] =='SS')].reset_index(drop=True)
    ss = ss.sort_values(by = 'Y').reset_index(drop=True)
    fs = play_data.loc[(play_data['Position'] =='FS')].reset_index(drop=True)
    fs = fs.sort_values(by = 'Y').reset_index(drop=True)
    leftover = play_data.loc[(play_data['Position'].isin(['DE', 'DT', 'NT', 'LB', 'OLB', 'MLB', 'ILB', 'CB', 'S', 'DB', 'SS', 'FS']) == False)].reset_index(drop=True)
    leftover = leftover.sort_values(by = 'Y').reset_index(drop=True)
    defense_pl = pd.concat([de, dt]).reset_index(drop=True)
    defense_pl = pd.concat([defense_pl, nt]).reset_index(drop=True)
    defense_pl = pd.concat([defense_pl, lb]).reset_index(drop=True)
    defense_pl = pd.concat([defense_pl, olb]).reset_index(drop=True)
    defense_pl = pd.concat([defense_pl, mlb]).reset_index(drop=True)
    defense_pl = pd.concat([defense_pl, ilb]).reset_index(drop=True)
    defense_pl = pd.concat([defense_pl, cb]).reset_index(drop=True)
    defense_pl = pd.concat([defense_pl, s]).reset_index(drop=True)
    defense_pl = pd.concat([defense_pl, db]).reset_index(drop=True)
    defense_pl = pd.concat([defense_pl, ss]).reset_index(drop=True)
    defense_pl = pd.concat([defense_pl, fs]).reset_index(drop=True)
    
    locations['X'].append(defense_pl['X'].iloc[0])
    locations['Y'].append(defense_pl['Y'].iloc[0])
    locations['Position'].append(defense_pl['Position'].iloc[0])
    locations['NflId'].append(defense_pl['NflId'].iloc[0])
    locations['X'].append(defense_pl['X'].iloc[1])
    locations['Y'].append(defense_pl['Y'].iloc[1])
    locations['Position'].append(defense_pl['Position'].iloc[1])
    locations['NflId'].append(defense_pl['NflId'].iloc[1])
    locations['X'].append(defense_pl['X'].iloc[2])
    locations['Y'].append(defense_pl['Y'].iloc[2])
    locations['Position'].append(defense_pl['Position'].iloc[2])
    locations['NflId'].append(defense_pl['NflId'].iloc[2])
    locations['X'].append(defense_pl['X'].iloc[3])
    locations['Y'].append(defense_pl['Y'].iloc[3])
    locations['Position'].append(defense_pl['Position'].iloc[3])
    locations['NflId'].append(defense_pl['NflId'].iloc[3])
    locations['X'].append(defense_pl['X'].iloc[4])
    locations['Y'].append(defense_pl['Y'].iloc[4])
    locations['Position'].append(defense_pl['Position'].iloc[4])
    locations['NflId'].append(defense_pl['NflId'].iloc[4])
    locations['X'].append(defense_pl['X'].iloc[5])
    locations['Y'].append(defense_pl['Y'].iloc[5])
    locations['Position'].append(defense_pl['Position'].iloc[5])
    locations['NflId'].append(defense_pl['NflId'].iloc[5])
    locations['X'].append(defense_pl['X'].iloc[6])
    locations['Y'].append(defense_pl['Y'].iloc[6])
    locations['Position'].append(defense_pl['Position'].iloc[6])
    locations['NflId'].append(defense_pl['NflId'].iloc[6])
    locations['X'].append(defense_pl['X'].iloc[7])
    locations['Y'].append(defense_pl['Y'].iloc[7])
    locations['Position'].append(defense_pl['Position'].iloc[7])
    locations['NflId'].append(defense_pl['NflId'].iloc[7])
    locations['X'].append(defense_pl['X'].iloc[8])
    locations['Y'].append(defense_pl['Y'].iloc[8])
    locations['Position'].append(defense_pl['Position'].iloc[8])
    locations['NflId'].append(defense_pl['NflId'].iloc[8])
    locations['X'].append(defense_pl['X'].iloc[9])
    locations['Y'].append(defense_pl['Y'].iloc[9])
    locations['Position'].append(defense_pl['Position'].iloc[9])
    locations['NflId'].append(defense_pl['NflId'].iloc[9])
    locations['X'].append(defense_pl['X'].iloc[10])
    locations['Y'].append(defense_pl['Y'].iloc[10])
    locations['Position'].append(defense_pl['Position'].iloc[10])
    locations['NflId'].append(defense_pl['NflId'].iloc[10])

    return locations

def generateFormationImages(df,nx,ny,alpha):
    plays = df.groupby('PlayId')
    nPlays = plays.ngroups
    playDict = {}
    
    xg = np.linspace(0,120,nx)
    yg = np.linspace(0,53.3,ny)
    x, y = np.meshgrid(xg,yg)
    grid = np.stack((x, y), axis=-1)
    
    
    for playId, playData in tqdm_notebook(plays):
        playDensities = makeFormationFields(playData,grid)
        playTensor = makeFormationTensor(playDensities,alpha)
        if np.isnan(playTensor).any():
            print(playId)
        playDict[playId] = playTensor
        
    return playDict

def makeFormationTensor(rho,alpha):
    #alpha should be in the range [1e-3, 1e2]
    #dens_list[0] = defense
    #dens_list[1] = offense
    #dens_list[2] = ball carrier
    
    rho_0 = rho[0]/np.max(rho[0])*127
    rho_1 = rho[1]/np.max(rho[1])*127
    rho_2 = rho[2]/np.max(rho[2])*127
    rho_3 = rho[3]/np.max(rho[3])*127
    rho_4 = rho[4]/np.max(rho[4])*127
    rho_5 = rho[5]/np.max(rho[5])*127
    playTensor = np.stack([rho_0,rho_1,rho_2,rho_3,rho_4,rho_5], axis = -1)
    playTensor = playTensor.astype('int8')
    #converting to int8 to save memory
    
    return playTensor

def makeFormationFields(df,grid): 
    ny, nx, _ = grid.shape
    rho_0 = np.zeros((ny,nx))
    rho_1 = np.zeros((ny,nx))
    rho_2 = np.zeros((ny,nx))
    rho_3 = np.zeros((ny,nx))
    rho_4 = np.zeros((ny,nx))
    rho_5 = np.zeros((ny,nx))
    
    
    for _, row in df.iterrows():
        pos = [row['X'],53.3 - row['Y']]
        #spe = row['S']
        spe = 0
        ori = row['Dir']
        
        if np.isnan(ori):
            ori = 0
            
        rho = Formationdens(pos,spe,ori,grid)
        
        if _ % 6 == 0:
            rho_0 += rho
        elif _ % 6 == 1:
            rho_1 += rho
        elif _ % 6 == 2:
            rho_2 += rho
        elif _ % 6 == 3:
            rho_3 += rho
        elif _ % 6 == 4:
            rho_4 += rho
        else:
            rho_5 += rho
            
    return [rho_0,rho_1,rho_2,rho_3,rho_4,rho_5]

def Formationdens(pos,spe,ori,grid):
    #need to convert units on parameters and estimate proper values for football vs. soccer
    #roc = 4 Original
    roc = 8 #Adjustment for formations
    srat = 0#spe**2/13**2
    
    R = np.array([[np.cos(ori),-np.sin(ori)], [np.sin(ori),np.cos(ori)]])
    S2 = np.array([[((roc-roc*srat)/2)**2,0],[0,((roc+roc*srat)/2)**2+1e-8]])
    sigma = np.matmul(np.matmul(R,S2),np.transpose(R))
    #mu = (pos[0]+spe*np.cos(ori)*0.5, pos[1]+spe*np.sin(ori)*0.5)
    new_x = pos[0]**.4
    new_y = pos[1]
    mu = (new_x+spe*np.cos(ori)*0.5, new_y+spe*np.sin(ori)*0.5)
    
    return stats.multivariate_normal.pdf(grid, mean = mu, cov = sigma)

def generateDefFormationImages(df,nx,ny,alpha):
    plays = df.groupby('PlayId')
    nPlays = plays.ngroups
    playDict = {}
    
    xg = np.linspace(0,120,nx)
    yg = np.linspace(0,53.3,ny)
    x, y = np.meshgrid(xg,yg)
    grid = np.stack((x, y), axis=-1)
    
    
    for playId, playData in tqdm_notebook(plays):
        playDensities = makeDefFormationFields(playData,grid)
        playTensor = makeDefFormationTensor(playDensities,alpha)
        if np.isnan(playTensor).any():
            print(playId)
        playDict[playId] = playTensor
        
    return playDict

def makeDefFormationTensor(rho,alpha):
    #alpha should be in the range [1e-3, 1e2]
    #dens_list[0] = defense
    #dens_list[1] = offense
    #dens_list[2] = ball carrier
    
    rho_0 = rho[0]/np.max(rho[0])*127
    rho_1 = rho[1]/np.max(rho[1])*127
    rho_2 = rho[2]/np.max(rho[2])*127
    rho_3 = rho[3]/np.max(rho[3])*127
    rho_4 = rho[4]/np.max(rho[4])*127
    rho_5 = rho[5]/np.max(rho[5])*127
    rho_6 = rho[6]/np.max(rho[6])*127
    rho_7 = rho[7]/np.max(rho[7])*127
    rho_8 = rho[8]/np.max(rho[8])*127
    rho_9 = rho[9]/np.max(rho[9])*127
    rho_10 = rho[10]/np.max(rho[10])*127
    playTensor = np.stack([rho_0,rho_1,rho_2,rho_3,rho_4,rho_5,rho_6,rho_7,rho_8,rho_9,rho_10], axis = -1)
    playTensor = playTensor.astype('int8')
    #converting to int8 to save memory
    
    return playTensor

def makeDefFormationFields(df,grid): 
    ny, nx, _ = grid.shape
    rho_0 = np.zeros((ny,nx))
    rho_1 = np.zeros((ny,nx))
    rho_2 = np.zeros((ny,nx))
    rho_3 = np.zeros((ny,nx))
    rho_4 = np.zeros((ny,nx))
    rho_5 = np.zeros((ny,nx))
    rho_6 = np.zeros((ny,nx))
    rho_7 = np.zeros((ny,nx))
    rho_8 = np.zeros((ny,nx))
    rho_9 = np.zeros((ny,nx))
    rho_10 = np.zeros((ny,nx))
    
    for _, row in df.iterrows():
        pos = [row['X'],53.3 - row['Y']]
        #spe = row['S']
        spe = 0
        ori = row['Dir']
        
        if np.isnan(ori):
            ori = 0
            
        rho = DefFormationdens(pos,spe,ori,grid)
        
        if _ % 11 == 0:
            rho_0 += rho
        elif _ % 11 == 1:
            rho_1 += rho
        elif _ % 11 == 2:
            rho_2 += rho
        elif _ % 11 == 3:
            rho_3 += rho
        elif _ % 11 == 4:
            rho_4 += rho
        elif _ % 11 == 5:
            rho_5 += rho
        elif _ % 11 == 6:
            rho_6 += rho
        elif _ % 11 == 7:
            rho_7 += rho
        elif _ % 11 == 8:
            rho_8 += rho
        elif _ % 11 == 9:
            rho_9 += rho
        else:
            rho_10 += rho
            
    return [rho_0,rho_1,rho_2,rho_3,rho_4,rho_5,rho_6,rho_7,rho_8,rho_9,rho_10]

def DefFormationdens(pos,spe,ori,grid):
    #need to convert units on parameters and estimate proper values for football vs. soccer
    #roc = 4 Original
    roc = 8 #Adjustment for formations
    srat = 0#spe**2/13**2
    
    R = np.array([[np.cos(ori),-np.sin(ori)], [np.sin(ori),np.cos(ori)]])
    S2 = np.array([[((roc-roc*srat)/2)**2,0],[0,((roc+roc*srat)/2)**2+1e-8]])
    sigma = np.matmul(np.matmul(R,S2),np.transpose(R))
    #mu = (pos[0]+spe*np.cos(ori)*0.5, pos[1]+spe*np.sin(ori)*0.5)
    new_x = pos[0]**.4
    new_y = pos[1]
    mu = (new_x+spe*np.cos(ori)*0.5, new_y+spe*np.sin(ori)*0.5)
    
    return stats.multivariate_normal.pdf(grid, mean = mu, cov = sigma)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

class KMeansClusteringWithVGG16:
    def __init__(self, min_clusters=20, max_clusters=100):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

    def _load_vgg16_features(self, images):
        """
        Modify the VGG16 model to accept 6-channel input images and extract features.
        """
        # Load the original VGG16 model without the top layers
        vgg16 = VGG16(weights='imagenet', include_top=False)

        # Modify the first Conv2D layer to accept 6 input channels
        original_conv1 = vgg16.layers[1]  # First Conv2D layer
        config = original_conv1.get_config()
        config['batch_input_shape'] = (None, 100, 200, 6)  # Adjust input shape to 6 channels
        config['input_shape'] = (100, 200, 6)

        # Create a new Conv2D layer with updated configuration
        new_conv1 = Conv2D(
            filters=config['filters'],
            kernel_size=config['kernel_size'],
            strides=config['strides'],
            padding=config['padding'],
            activation=config['activation'],
            name='block1_conv1_custom',
        )

        # Reconstruct the model
        input_tensor = Input(shape=(100, 200, 6))  # Adjust input shape
        x = new_conv1(input_tensor)
        for layer in vgg16.layers[2:]:
            x = layer(x)  # Connect subsequent layers to the modified input
        feature_extractor = Model(inputs=input_tensor, outputs=x)

        # Preprocess and extract features
        images = images.astype(np.float32) / 128  # Normalize images to [0, 1]
        features = feature_extractor.predict(images)
        features_flat = features.reshape(features.shape[0], -1)

        return features_flat

    def _optimize_clusters(self, features):
        """
        Find the optimal number of clusters based on silhouette scores.
        """
        silhouette_scores = []
        range_n_clusters = range(self.min_clusters, self.max_clusters)

        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            silhouette_avg = silhouette_score(features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"Number of clusters: {n_clusters}, Silhouette score: {silhouette_avg:.4f}")

        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(range_n_clusters, silhouette_scores, marker='o')
        plt.title('Silhouette Scores for Different Numbers of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid()
        plt.show()

        # Select the number of clusters with the highest silhouette score
        self.optimal_clusters = range_n_clusters[np.argmax(silhouette_scores)]
        print(f"Optimal Number of Clusters: {self.optimal_clusters}")

        return self.optimal_clusters

    def fit(self, images):
        """
        Perform k-means clustering with VGG16 feature extraction.
        Expects images of shape (n_samples, 100, 200, 6).
        """
        # Step 1: Extract features using the modified VGG16 model
        features = self._load_vgg16_features(images)

        # Step 2: Perform dimensionality reduction with PCA
        pca = PCA(n_components=min(images.shape[0], self.max_clusters))
        features_reduced = pca.fit_transform(features)

        # Step 3: Optimize clusters based on silhouette scores
        optimal_clusters = self._optimize_clusters(features_reduced)

        # Step 4: Perform final clustering with optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        self.labels_ = kmeans.fit_predict(features_reduced)

        return self.labels_

    def plot_cluster_distribution(self):
        """
        Plot the distribution of samples across clusters.
        """
        if not hasattr(self, 'labels_'):
            raise ValueError("Clustering has not been performed yet. Call fit() first.")

        unique, counts = np.unique(self.labels_, return_counts=True)
        plt.figure(figsize=(10, 6))
        plt.bar(unique, counts)
        plt.title('Distribution of Samples Across Clusters')
        plt.xlabel('Cluster Label')
        plt.ylabel('Number of Samples')
        plt.show()

class DefKMeansClusteringWithVGG16:
    def __init__(self, min_clusters=1, max_clusters=100):
        if max_clusters <= min_clusters:
            raise ValueError("max_clusters must be greater than min_clusters.")
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

    def _def_load_vgg16_features(self, images):
        """
        Modify the VGG16 model to accept 11-channel input images and extract features.
        """
        # Load the original VGG16 model without the top layers
        vgg16 = VGG16(weights='imagenet', include_top=False)

        # Modify the first Conv2D layer to accept 11 input channels
        original_conv1 = vgg16.layers[1]
        config = original_conv1.get_config()
        config['input_shape'] = (100, 200, 11)  # Updated to 11 channels
        config['filters'] = original_conv1.filters
        config['kernel_size'] = original_conv1.kernel_size
        config['strides'] = original_conv1.strides
        config['padding'] = original_conv1.padding

        # Create a new Conv2D layer with the updated configuration
        new_conv1 = Conv2D(**config)
        new_conv1.build((None, 100, 200, 11))

        # Construct the model
        input_tensor = Input(shape=(100, 200, 11))  # Updated input shape
        x = new_conv1(input_tensor)
        for layer in vgg16.layers[2:]:
            x = layer(x)
        feature_extractor = Model(inputs=input_tensor, outputs=x)

        # Preprocess and extract features
        images = images.astype(np.float32) / 128.0
        features = feature_extractor.predict(images)
        features_flat = features.reshape(features.shape[0], -1)

        return features_flat

    def _def_optimize_clusters(self, features):
        """
        Find the optimal number of clusters based on silhouette scores.
        """
        silhouette_scores = []
        range_n_clusters = range(self.min_clusters, self.max_clusters)

        if not range_n_clusters:  # Handle empty range
            raise ValueError("Invalid cluster range. Check min_clusters and max_clusters values.")

        for n_clusters in range_n_clusters:
            if n_clusters == 1:
                # Handle k=1 case: no silhouette score, assign a default score
                silhouette_scores.append(0)
                print(f"Number of clusters: {n_clusters}, Silhouette score: N/A (defaulted to 0)")
                continue

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            silhouette_avg = silhouette_score(features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"Number of clusters: {n_clusters}, Silhouette score: {silhouette_avg:.4f}")

        # Handle case where no valid silhouette scores are computed
        if not silhouette_scores:
            raise ValueError("No silhouette scores were computed. Check the input data and cluster range.")

        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(range_n_clusters, silhouette_scores, marker='o')
        plt.title('Silhouette Scores for Different Numbers of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid()
        plt.show()

        # Select the number of clusters with the highest silhouette score
        self.optimal_clusters = range_n_clusters[np.argmax(silhouette_scores)]
        print(f"Optimal Number of Clusters: {self.optimal_clusters}")

        return self.optimal_clusters

    def def_fit(self, images):
        """
        Perform k-means clustering with VGG16 feature extraction.
        Expects images of shape (n_samples, 100, 200, 11).
        """
        if images.shape[0] == 1:
            print("Only one sample provided. Assigning to a single cluster.")
            self.labels_ = np.zeros(1, dtype=int)  # Single sample in one cluster
            return self.labels_

        # Step 1: Extract features using the modified VGG16 model
        features = self._def_load_vgg16_features(images)

        # Step 2: Perform dimensionality reduction with PCA
        pca = PCA(n_components=min(images.shape[0], self.max_clusters))
        features_reduced = pca.fit_transform(features)

        # Step 3: Optimize clusters based on silhouette scores
        optimal_clusters = self._def_optimize_clusters(features_reduced)

        # Step 4: Perform final clustering with optimal number of clusters
        if optimal_clusters == 1:
            self.labels_ = np.zeros(features_reduced.shape[0], dtype=int)  # All data points in one cluster
        else:
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            self.labels_ = kmeans.fit_predict(features_reduced)

        return self.labels_

    def def_plot_cluster_distribution(self):
        """
        Plot the distribution of samples across clusters.
        """
        if not hasattr(self, 'labels_'):
            raise ValueError("Clustering has not been performed yet. Call fit() first.")

        unique, counts = np.unique(self.labels_, return_counts=True)
        plt.figure(figsize=(10, 6))
        plt.bar(unique, counts)
        plt.title('Distribution of Samples Across Clusters')
        plt.xlabel('Cluster Label')
        plt.ylabel('Number of Samples')
        plt.show()



    def def_plot_cluster_distribution(self):
        """
        Plot the distribution of samples across clusters.
        """
        if not hasattr(self, 'labels_'):
            raise ValueError("Clustering has not been performed yet. Call fit() first.")

        unique, counts = np.unique(self.labels_, return_counts=True)
        plt.figure(figsize=(10, 6))
        plt.bar(unique, counts)
        plt.title('Distribution of Samples Across Clusters')
        plt.xlabel('Cluster Label')
        plt.ylabel('Number of Samples')
        plt.show()

# Usage example:
# clustering = KMeansClusteringWithVGG16(max_clusters=20)
# labels = clustering.fit(images)
# clustering.plot_cluster_distribution()

def clean_football_df(df):
    df = (df 
            .assign(ToLeft=df['PlayDirection']=='left')
           )# standardize field positions
    df = (df
            .assign(X=np.where(df['ToLeft'], 120-df['X'], df['X'])-10)
            .assign(Y=np.where(df['ToLeft'], 53.33-df['Y'], df['Y']))
           )
    
    return df

### Define functions that generate spatial control fields for each play in input dataframe
def generateSpaces(df,nx,ny,alpha):
    plays = df.iloc[:, :-1].groupby('PlayId')
    nPlays = plays.ngroups
    playDict = {}
    
    xg = np.linspace(0,120,nx)
    yg = np.linspace(0,53.3,ny)
    x, y = np.meshgrid(xg,yg)
    grid = np.stack((x, y), axis=-1)
    
    for playId, playData in tqdm_notebook(plays):
        playDensities = space_makeFields(playData,grid)
        playTensor = space_makeTensor(playDensities[0],playDensities[1],alpha)
        playDict[playId] = playTensor
        
    return playDict

def space_makeTensor(rho,pcpp,alpha):
    #alpha should be in the range [1e-3, 1e2]
    #dens_list[0] = defense
    #dens_list[1] = offense
    #dens_list[2] = ball carrier
    pcpp2 = {}
    rho_off = rho[1]/np.max(rho[1])*127
    rho_def = rho[0]/np.max(rho[0])*127
    rho_comp = (expit(alpha*(rho_off-rho_def)))*127
    for idx, i in enumerate(pcpp['NflId']):
        PlayId = pcpp['PlayId'][idx]
        rho_indy = pcpp['rho'][idx]/np.max(pcpp['rho'][idx])*127
        rho_indy = np.clip(((expit(alpha*(rho_indy - rho_off-rho_def)))*127) - 63.5, 0, 127)
        pcpp2[i]=(rho_indy)
    
    return pcpp2

def space_makeFields(df,grid): 
    ny, nx, _ = grid.shape
    rho_def = np.zeros((ny,nx))
    rho_off = np.zeros((ny,nx))
    rho_bc = np.zeros((ny,nx))
    pcpp = {'PlayId':[], 'NflId':[], 'rho':[]}
    
    for _, row in df.iterrows():
        pos = [row['X'],53.3 - row['Y']]
        spe = row['S']
        ori = row['Dir']
        
        if np.isnan(ori):
            ori = 0
            
        rho = space_dens(pos,spe,ori,grid)
   
        if row['IsOnOffense']:
            rho_off += rho
            if row['BallCarrier']:
                rho_bc += rho
        else:
            rho_def += rho

        pcpp['PlayId'].append(row['PlayId'])
        pcpp['NflId'].append(row['NflId'])
        pcpp['rho'].append(rho)
            
    return [rho_def,rho_off,rho_bc], pcpp

def space_dens(pos,spe,ori,grid):
    #need to convert units on parameters and estimate proper values for football vs. soccer
    roc = 4
    srat = spe**2/13**2
    
    R = np.array([[np.cos(ori),-np.sin(ori)], [np.sin(ori),np.cos(ori)]])
    S2 = np.array([[((roc-roc*srat)/2)**2,0],[0,((roc+roc*srat)/2)**2+1e-8]])
    sigma = np.matmul(np.matmul(R,S2),np.transpose(R))
    mu = (pos[0]+spe*np.cos(ori)*0.5, pos[1]+spe*np.sin(ori)*0.5)
    
    return stats.multivariate_normal.pdf(grid, mean = mu, cov = sigma)