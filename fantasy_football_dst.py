# This script has 3 modules:
#
# (1) Data scraping/processing
#        (a) fantasy football stats from fftoday.com
#        (b) nfl schedule from pro-football-reference.com
#        (c) vegas lines
#        (d) expert rankings
#
# (2) Analysis
#        (a) establish predictor/response variables
#        (b) train models
#        (c) test models
#        (d) calculate testing error
#
# (3) Visualization
#        (a) comparison of expert, machine learning, and heuristic rankings
#        (b) plots of feature importances in machine learning models



################ Data Scraping/Processing ################

import pandas as pd
import numpy as np
import pickle
n_weeks = 16


##### FANTASY FOOTBALL STATS

if False:
    
    # scrape data
    import requests
    stats = []
    for week in range(1,n_weeks+1):
        #print(week)
        url= 'http://www.fftoday.com/stats/playerstats.php?Season=2017&GameWeek=%d&PosID=99&LeagueID=' % week
        html = requests.get(url).content
        df = pd.read_html(html)
        stats.append(df[9])
        #print(len(stats))
    
    with open("ff_stats_2017.file", "wb") as f:
        pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)

else:
    
    # load scraped data
    with open("ff_stats_2017.file", "rb") as f:
        stats = pickle.load(f)
        
# list of teams sorted alphabetically
team_list = stats[1].iloc[:,0:2]
team_list = team_list.rename(columns = team_list.iloc[2,:])
team_list = team_list.iloc[3:,:]
team_list.iloc[:,0] = stats[1].iloc[:,0].str.split('\\. ').str[1]
team_list = team_list.sort_values('Team')
team_list = np.array(team_list.iloc[:,0])


##### SCHEDULE

if False:
    url = 'https://www.pro-football-reference.com/years/2017/games.htm'
    html = requests.get(url).content
    schedule_raw = pd.read_html(html)[0]

    with open("nfl_schedule_2017.file", "wb") as f:
        pickle.dump(schedule_raw, f, pickle.HIGHEST_PROTOCOL)

else:
    # load raw schedule
    with open("nfl_schedule_2017.file", "rb") as f:
        schedule_raw = pickle.load(f)

# schedule is stored here as stacked connectivity matrices
schedule = np.zeros([len(team_list),len(team_list),16])

# information on game time and home/away
game_info = np.zeros([16,len(team_list),3], dtype='U5')
game_info.fill(np.nan)

for week_idx in range(16):
    week = week_idx + 1
    df_week = np.array(schedule_raw.loc[df['Week'] == str(week)])
    
    for row_idx in range(df_week.shape[0]):
        row = df_week[row_idx,:]
        team1_idx = np.where(team_list == row[4])[0]
        team2_idx = np.where(team_list == row[6])[0]
        schedule[team1_idx, team2_idx, week_idx] = 1
        schedule[team2_idx, team1_idx, week_idx] = 1

        # day of the week
        game_info[week_idx, [team1_idx, team2_idx], 0] = row[1]
        
        # time of day
        time_civ = row[3]
        
        if find_str(time_civ, 'P') > -1:
            hour = np.int(time_civ.split(':')[0]) + 12
            minute = np.int(time_civ.split(':')[1].split('P')[0])
        elif find_str(time_civ, 'A') > -1:
            hour = np.int(time_civ.split(':')[0])
            minute = np.int(time_civ.split(':')[1].split('A')[0])
        
        time_dec = hour + minute/60
        game_info[week_idx, [team1_idx, team2_idx], 1] = time_dec

        if row[5] == '@':
            game_info[week_idx, team2_idx, 2] = 1
            game_info[week_idx, team1_idx, 2] = 0
        else:
            game_info[week_idx, team1_idx, 2] = 1
            game_info[week_idx, team2_idx, 2] = 0


##### VEGAS LINES

# function to identify positions in a string matching to a substring
def find_str(s, char):
    index = 0

    if char in s:
        c = char[0]
        for ch in s:
            if ch == c:
                if s[index:index+len(char)] == char:
                    return index

            index += 1

    return -1

if False:

    # irregular IDs inside of ESPN urls that must be input manually
    url_ids = [19203589, 20663439, 20739796, 20808335, 20886007, 20955567, 21036470, 21114574,
               21218174, 21307531, 21388874, 21476348, 21570070, 21652389, 21730080, 21792884]

    # vegas lines are stored here
    vegas = np.zeros([n_weeks,len(team_list),2])
    vegas.fill(np.nan)

    # lists containing all possible spreads and over/unders to search through
    spread_list = np.arange(-30, 0, 0.5)
    overunder_list = np.arange(70, 30, -0.5)

    for week_idx in range(16):
        if week_idx == 0:
            url = 'http://www.espn.com/chalk/story/_/id/%d/nfl-full-list-opening-week-%d-odds-westgate-las-vegas-superbook' % (url_ids[week_idx], week_idx + 1)
        else:
            url = 'http://www.espn.com/chalk/story/_/id/%d/nfl-full-list-week-%d-odds-westgate-las-vegas-superbook' % (url_ids[week_idx], week_idx + 1)
        page = requests.get(url)
        bs = BeautifulSoup(page.content, 'html.parser')
        rows = bs.select('p')

        for row_idx in range(len(rows)):
            row = str(rows[row_idx])
            team_isinrow = []
            for team_name in team_list: team_isinrow.append(find_str(row, team_name))
            team_isinrow = np.array(team_isinrow)
            team_locations = [i for i in team_isinrow if i > -1]
            if (len(team_locations) == 2):
                #print(row_idx)
                team_indices = np.where(team_isinrow > -1)[0]
                #print(team_list[team_indices])

                # find spread
                for spread in spread_list:
                    if spread % 1 == 0: spread = np.int(spread)
                    spread_location = find_str(row, str(spread))
                    if spread_location > -1: break

                # find over/under
                for overunder in overunder_list:
                    if overunder % 1 == 0: overunder = np.int(overunder)
                    if find_str(row, str(overunder)) > -1: break


                if np.all(spread_location > np.array(team_locations)):
                    if team_locations[0] > team_locations[1]:
                        vegas[week_idx, team_indices[0], 0] = spread
                        vegas[week_idx, team_indices[1], 0] = -spread

                    else:
                        vegas[week_idx, team_indices[1], 0] = spread
                        vegas[week_idx, team_indices[0], 0] = -spread

                else:
                    if team_locations[0] > team_locations[1]:
                        vegas[week_idx, team_indices[1], 0] = spread
                        vegas[week_idx, team_indices[0], 0] = -spread

                    else:
                        vegas[week_idx, team_indices[0], 0] = spread
                        vegas[week_idx, team_indices[1], 0] = -spread

                vegas[week_idx, team_indices[0], 1] = overunder
                vegas[week_idx, team_indices[1], 1] = overunder



        with open("nfl_vegas_2017.file", "wb") as f:
            pickle.dump(vegas, f, pickle.HIGHEST_PROTOCOL)

else:
    # load scraped vegas lines
    with open("nfl_vegas_2017.file", "rb") as f:
        vegas = pickle.load(f)
    
    
# exception for the Miami/Tampa game that was moved to week 11
vegas[0,[18,29],:] = np.nan

# need to impute missing vegas data (from thursday night games)
for week_idx in range(16):
    vegas_miss = np.where(np.isnan(vegas[week_idx,:,0]))[0]
    game_info_miss = np.where(game_info[week_idx,:,0] == 'nan')[0]
    diff = list(set(vegas_miss) - set(game_info_miss))
    vegas[week_idx, diff, 0] = 0
    vegas[week_idx, diff, 1] = np.nanmean(vegas[week_idx, :, 1])
    
# make sure that bye weeks match in game_info and stats  
count = 0
for week_idx in range(16):
      count = count + len(set(np.where(np.isnan(s[week_idx, :, 0]))[0]) -
                          set(np.where(np.isnan(vegas[week_idx, :, 0]))[0]))
      count = count + len(set(np.where(np.isnan(vegas[week_idx, :, 0]))[0]) -
                          set(np.where(np.isnan(s[week_idx, :, 0]))[0]))

if count == 0: print('bye weeks DO match')
else: print('bye weeks DO NOT match')


##### EXPERT RANKINGS

if False:
    dwc_rankings =  np.empty((16))
    fmt1 = ['09','09','09','09','10','10','10','10','11','11','11','11','11','12','12','12']
    for i in range(16):
        
        if i == 4:
            url= 'https://www.fantasypros.com/2016/%s/defense-wins-championships-week-%d-2/' % (fmt1[i], i+1)
        else:    
            url= 'https://www.fantasypros.com/2016/%s/defense-wins-championships-week-%d-2016/' % (fmt1[i], i+1)
        
        page = requests.get(url)
        bs = BeautifulSoup(page.content, 'html.parser')
        skip = len(bs.select('ol li')) - 16
        rankings_raw = bs.select('ol li')[skip:]
        
        rankings_temp = []
        for j in range(16):
            rankings_temp.append(str(rankings_raw[j]).split('>')[2].split('<')[0])
    
        dwc_rankings = np.vstack((dwc_rankings,np.array(rankings_temp)))
    
    dwc_rankings = dwc_rankings[1:,:]
    
    import pickle
    with open('dwc_rankings_2016.file', 'wb') as f:
        pickle.dump(dwc_rankings, f, pickle.HIGHEST_PROTOCOL)

elif False:
    
    # load scraped data
    with open('dwc_rankings_2016.file', 'rb') as f:
        stats = pickle.load(f)
        
else:
    
    # load from csv
    dwc_rankings_raw = pd.read_csv('2017_projections_raw.csv',encoding = 'ISO-8859-1')
    
    # check to see if team names match those from team_list
    overlap = []
    for week_idx in range(16):
        overlap.append(len(intersect(np.array(dwc_rankings_raw.iloc[:,2*week_idx]),team_list)))
    
    if len(np.where(np.array(overlap) < len(team_list))[0]) > 0: print('Team names DO NOT match')
    else: print('Team names DO match')
    
    
##### PROCESSING

# function to calculate dst fantasy scores for nfl.com standard scoring
def fantasy_score(s_row):
    score = s_row[1]*1 #sacks
    score = score + s_row[2]*2 #fumble recoveries
    score = score + s_row[3]*2 #interceptions
    score = score + (s_row[4]+s_row[9])*6 #touchdowns
    score = score + s_row[8]*2 #safeties
    
    pa = [0,1,7,14,21,28,35]
    fp = [10,7,4,1,0,-1,-4]
    score = score + fp[np.sum([s_row[5] >= x for x in pa])-1]
    return(score)

# begin storing stats in numpy array
s = np.zeros([n_weeks,len(team_list),11])
s_vs = np.zeros([n_weeks,len(team_list),11])

for week_idx in range(0,n_weeks):
    
    # pull stats from specified week
    stats_sub = stats[week_idx]
    
    # define column names
    colnames = stats_sub.iloc[2]
    # remove useless rows
    stats_sub = stats_sub[3:]
    # set column names
    stats_sub = stats_sub.rename(columns = colnames)
    
    # parse team names
    stats_sub.iloc[:,0] = stats_sub.iloc[:,0].str.split('\\. ').str[1]
    # define row names
    rownames = stats_sub.iloc[:,0]
    # remove useless columns
    stats_sub = stats_sub.iloc[:,range(1,12)]
    # set row names
    stats_sub = stats_sub.rename(index = rownames)
    # reorder rows based on team name
    #stats_sub = stats_sub.sort_values('Team')

    # initialize 2D array for specified week
    stats_store = np.empty((32,11))
    stats_store[:] = np.nan
    
    stats_store_vs = np.empty((32,11))
    stats_store_vs[:] = np.nan
    
    for team_idx in range(len(team_list)):
        team_name = team_list[team_idx]
        if team_name in np.array(stats_sub.index):
            stats_store[team_idx, :] = stats_sub.loc[team_name]
            stats_store_vs[np.where(schedule[team_idx, :, week_idx] == 1)[0][0], :] = stats_sub.loc[team_name]

    stats_store[:, 10] = np.apply_along_axis(fantasy_score, 1, stats_store)
    stats_store_vs[:, 10] = np.apply_along_axis(fantasy_score, 1, stats_store_vs)

    s[week_idx, :, :] = stats_store
    s_vs[week_idx, :, :] = stats_store_vs


# histogram of all scores
scores_all = s[:,:,10]
scores_all = scores_all[~np.isnan(scores_all)]
plt.hist(scores_all)
plt.hist(np.log(scores_all - np.min(scores_all) + 5))

if False:
    # log transform in order to normalize scores
    s[:,:,10] = np.log(s[:,:,10] - np.min(scores_all) + 1)
    s_vs[:,:,10] = np.log(s_vs[:,:,10] - np.min(scores_all) + 1)

# little experiment: confirm that best transformation involves adding smallest possible constant
if False:
    from scipy import stats as statistics
    print(statistics.kstest(scores_all, 'norm')[0])
    for i in range(1,10): print(statistics.kstest(np.log(scores_all - np.min(scores_all) + i), 'norm')[0])



################ Analysis ################

import scipy as sp
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def matchups(x): 
    if np.sum(x) == 1: return np.where(x == 1)[0][0]
    else: out = -1
    return out

# objects for storing results
rmse_train = np.zeros([17,3])
rmse_test = np.zeros([17,3])

cor_k = np.zeros([17,8])
cor_s = np.zeros([17,8])

imp_ridge = np.zeros([len(np.arange(4, 16)), 25])
imp_rf = np.zeros([len(np.arange(4, 16)), 25])

# feature names
X_names = ['Sack_ssn', 'FR_ssn', 'INT_ssn', 'DTD_ssn', 'PA_ssn', 'PaYd_ssn', 'RuYd_ssn', 'Saf_ssn', 'KTD_ssn', 'FPts_ssn',
           'Sack_rec', 'FR_rec', 'INT_rec', 'DTD_rec', 'PA_rec', 'PaYd_rec', 'RuYd_rec', 'Saf_rec', 'KTD_rec', 'FPts_rec',
           'Spread', 'Ovr/Undr', 'Time', 'Home', 'Day']

for test_idx in range(4, 16):
    
    ##### TRAINING
    
    train_idx = test_idx - 1
    
    Y_train = s[train_idx, :, 10]
    
    X_train = np.hstack((np.nanmean(s[:(train_idx-1),:,1:], axis=0), 
                         np.nanmean(np.array([ s[train_idx-1,:,1:], s[train_idx-2,:,1:]]), axis=0),
                         vegas[train_idx, :, :], game_info[train_idx, :, 1:]))
    
    # "day of week" feature handled separately due to categorical nature
    dow_train = game_info[train_idx, :, 0]
    
    for train_idx in range(test_idx-2, 2, -1):
        
        Y_temp = s[train_idx, :, 10]
        
        X_temp = np.hstack((np.nanmean(s[:(train_idx-1),:,1:], axis=0), 
                             np.nanmean(np.array([ s[train_idx-1,:,1:], s[train_idx-2,:,1:]]), axis=0),
                             vegas[train_idx, :, :], game_info[train_idx, :, 1:]))
        
        dow_temp = game_info[train_idx, :, 0]
        
        Y_train = np.concatenate((Y_train, Y_temp))
        X_train = np.concatenate((X_train, X_temp))
        dow_train = np.concatenate((dow_train, dow_temp))
        
    notnan = np.where(~np.isnan(Y_train))[0]
    Y_train = Y_train[notnan]
    X_train = X_train[notnan, :]
    dow_train = dow_train[notnan]
    
    enc =  LabelEncoder()
    enc.fit(['Thu', 'Sat', 'Sun', 'Mon'])
    dow_train = enc.transform(dow_train)
    dow_train.shape = (len(dow_train), 1)
    
    X_train = np.float_(X_train)
    X_train = np.hstack((X_train, dow_train))
    
    # qc step to show that models detect signal if given a feature correlated to the response
    #X_train = np.hstack((X_train, (Y_train + np.random.normal(0, 1, X_train.shape[0])).reshape((X_train.shape[0], 1))))
    
    # train models and calculate training error
    
    # L2 regularized regression (Ridge)
    alpha_list_coarse = 10 ** np.arange(-10, 11, 1, dtype='float')
    ridge = linear_model.RidgeCV(alphas = alpha_list_coarse, cv=10)
    ridge.fit (X_train, Y_train)
    
    alpha_list_fine = linspace(ridge.alpha_/10, ridge.alpha_*10, num=101, endpoint=True, retstep=False)
    ridge = linear_model.RidgeCV(alphas = alpha_list_fine, cv=10)
    ridge.fit (X_train, Y_train)
    
    imp_ridge[test_idx - 4, :] = np.absolute(ridge.coef_)/sum(np.absolute(ridge.coef_))
    
    # random forest
    rf = RandomForestRegressor(n_estimators=1000, max_features='sqrt',
                                 min_samples_split=10, random_state=0)
    rf.fit(X_train, Y_train)
    
    imp_rf[test_idx - 4, :] = rf.feature_importances_
    
    # calculate root mean squared error on training data
    rmse_train[test_idx, 0] = sqrt(mean_squared_error(Y_train, ridge.predict(X_train)))
    rmse_train[test_idx, 1] = sqrt(mean_squared_error(Y_train, rf.predict(X_train)))    
    
    ##### TESTING
    
    Y_test = s[test_idx, :, 10]
    
    X_season = np.nanmean(s[:(test_idx-1), :, 1:], axis=0)
    X_recent = np.nanmean(np.array([ s[test_idx-1,:,1:], s[test_idx-2, :, 1:]]), axis=0)
    X_test = np.hstack((X_season, X_recent, vegas[test_idx, :, :], game_info[test_idx, :, 1:]))
    
    # remove bye weeks and add "day of week" feature
    notnan = np.where(~np.isnan(Y_test))[0]
    Y_test = Y_test[notnan]
    X_test = X_test[notnan, :]
    dow_test = game_info[test_idx, :, 0]
    dow_test = dow_test[notnan]
    dow_test = enc.transform(dow_test)
    dow_test.shape = (len(dow_test), 1)
    X_test = np.hstack((np.float_(X_test), dow_test))
    
    # calculate root mean squared error on test data
    rmse_test[test_idx, 0] = sqrt(mean_squared_error(Y_test, ridge.predict(X_test)))
    rmse_test[test_idx, 1] = sqrt(mean_squared_error(Y_test, rf.predict(X_test)))
    
    # performance assessment
    
    # true rankings based on actual fantasy points scored
    true_rankings = s[test_idx, :, 10]
    not_nan = ~np.isnan(true_rankings)
    true_rankings = true_rankings[not_nan]
    
    # expert rankings based on Defense Wins Championships (Dylan Lerch)
    dwc_week_names = np.array(dwc_rankings_raw.iloc[:,2*test_idx])
    dwc_week_scores = np.array(dwc_rankings_raw.iloc[:,2*test_idx+1])
    dwc_rankings = dwc_week_names[[x != '#VALUE!' for x in dwc_week_scores]]
    
    expert_rankings = np.argsort(np.flip(dwc_rankings,axis=0))
    cor_k[test_idx, 0] = sp.stats.kendalltau(true_rankings, expert_rankings)[0]
    cor_s[test_idx, 0] = sp.stats.spearmanr(true_rankings, expert_rankings)[0]

    # ridge rankings based on ridge regression predictions
    ridge_rankings = ridge.predict(X_test)
    cor_k[test_idx, 1] = sp.stats.kendalltau(true_rankings, ridge_rankings)[0]
    cor_s[test_idx, 1] = sp.stats.spearmanr(true_rankings, ridge_rankings)[0]
    
    # rf rankings based on random forest predictions
    rf_rankings = rf.predict(X_test)
    cor_k[test_idx, 2] = sp.stats.kendalltau(true_rankings, rf_rankings)[0]
    cor_s[test_idx, 2] = sp.stats.spearmanr(true_rankings, rf_rankings)[0]
    
    # naive rankings based on average fantasy scores over season-to-date
    naive_rankings = np.nanmean(np.array(s[0:(test_idx-1),:,10]), axis=0)[not_nan]
    cor_k[test_idx, 3] = sp.stats.kendalltau(true_rankings, naive_rankings)[0]
    cor_s[test_idx, 3] = sp.stats.spearmanr(true_rankings, naive_rankings)[0]

    # naiverecent rankings based on average fantasy scores over last 3 weeks
    naiverecent_rankings = np.nanmean(np.array(s[(test_idx-4):(test_idx-1),:,10]), axis=0)[not_nan]
    cor_k[test_idx, 4] = sp.stats.kendalltau(true_rankings, naiverecent_rankings)[0]
    cor_s[test_idx, 4] = sp.stats.spearmanr(true_rankings, naiverecent_rankings)[0]
    
    # lessnaive rankings based on fantasy scores over season-to-date and strength of opponent
    matchup_order = np.apply_along_axis(matchups, 0, schedule[:,:,test_idx])[not_nan]
    
    lessnaive_rankings =  (np.nanmean(np.array(s[0:(test_idx-1), :, 10]), axis=0)[not_nan] +
    (1)*np.nanmean(np.array(s_vs[0:(test_idx-1), :, 10]), axis=0)[matchup_order]) / (2)
                          
    cor_k[test_idx, 5] = sp.stats.kendalltau(true_rankings, lessnaive_rankings)[0]  
    cor_s[test_idx, 5] = sp.stats.spearmanr(true_rankings, lessnaive_rankings)[0]           
         
    # random rankings based on randoly guessing order
    random_rankings = np.random.permutation(true_rankings)
    cor_k[test_idx, 6] = sp.stats.kendalltau(true_rankings, random_rankings)[0] 
    cor_s[test_idx, 6] = sp.stats.spearmanr(true_rankings, random_rankings)[0]



################ Visualization ################

##### PLOT: PERFORMANCE
    
import matplotlib.pyplot as plt

# prepare plotting arrays
y = np.array([])
y_means = np.array([])
x = np.array([])
c = np.array([])
c_list = ['purple','orange','green','magenta','blue', 'darkred', 'darkgray']
count = 0
for i in [0, 1, 2, 3, 4, 5, 6]:
    y = np.append(y, cor_s[4:16,i])
    y_means = np.append(y_means, np.mean(cor_s[4:16,i]))
    x = np.append(x, np.random.normal(count, 0.025, 12))
    c = np.append(c, np.repeat(c_list[count],12))
    count = count + 1

# scatterplot
ax = plt.subplot(1, 1, 1)
ax.plot([-10, 10],[0, 0], c = 'gray', alpha = 0.5, linewidth=0.5)
ax.scatter(x, y, c=c, alpha=0.5)
plt.title('Rank correlation between true and predicted rankings\nWeeks 5-16 (2017)\n')
plt.ylabel("Kendall's Tau")
x_ticklabels = ['Expert', 'Ridge Regression', 'Random Forest', 'Naive', 'Naive Recent', 'Naive + Versus', 'Random Guess']
plt.xticks([0, 1, 2, 3, 4, 5, 6], x_ticklabels, rotation=45)
plt.xlim(-0.5, 6.5)
plt.ylim(-0.5, 0.5)
for margin in [-0.23, -0.18, 0.15, 0.2]:
    ax.plot(np.arange(7) + margin, y_means, "or", c='black', marker='_')
plt.show()
plt.savefig('rank_correlation.png', bbox_inches='tight', dpi=100)


##### PLOT: FEATURE IMPORTANCES

import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# create color palette for stacked bars
colors = sns.color_palette("tab10", n_colors=n)
cmap1 = LinearSegmentedColormap.from_list("my_colormap", colors)

palette = []
for i in range(n): palette.append(cmap1(linspace(0, 1, num=12, endpoint=True, retstep=False)[i]))

# format strings for use in legend (there is definitely a better way)
week_list = []
for i in (np.arange(4, 16) + 1): week_list.append('Week %d' % i)

# top subplot is for ridge regression
imp_temp = imp_ridge
n = imp_temp.shape[0]
N = imp_temp.shape[1]
ind = np.arange(N)
width = 0.5

# stacked barplots
plt.subplot(2, 1, 1)
p0 = plt.bar(ind, imp_temp[0, :], width, color=palette[0])
store = []
store.append(p0)
for i in np.arange(1, n):
    p0 = plt.bar(ind, imp_temp[i, :], width, color=palette[i],
                 bottom=np.apply_along_axis(np.sum, 0, imp_temp[:i, :]))
    store.append(p0)

# formatting
plt.ylabel('Feature Importance')
plt.title('Feature Importances - Ridge Regression')
plt.xticks(ind, [])
plt.yticks([0, 1, 2], [0, 1, 2])
leg = plt.legend(list(reversed(store)), list(reversed(week_list)), 
                 bbox_to_anchor=(1.13, 1.05), prop={'size': 5})
leg.get_frame().set_linewidth(0.0)

# bottom subplot is for random forest
imp_temp = imp_rf

# stacked barplots
plt.subplot(2, 1, 2)
p0 = plt.bar(ind, imp_temp[0, :], width, color=palette[0])
store = []
store.append(p0)
for i in np.arange(1, n):
    p0 = plt.bar(ind, imp_temp[i, :], width, color=palette[i],
                 bottom=np.apply_along_axis(np.sum, 0, imp_temp[:i, :]))
    store.append(p0)

# formatting
plt.ylabel('Feature Importance')
plt.title('Feature Importances - Random Forest')
plt.xticks(ind, X_names, rotation=90)
plt.yticks([0, 1, 2], [0, 1, 2])
leg = plt.legend(list(reversed(store)), list(reversed(week_list)), 
                 bbox_to_anchor=(1.13, 1.05), prop={'size': 5})
leg.get_frame().set_linewidth(0.0)

plt.tight_layout()

plt.savefig('feature_importance.pdf', bbox_inches='tight')
