import numpy as np
import pandas as pd

training_set = pd.read_excel("Data_Train.xlsx")
test_set = pd.read_excel("Data_Test.xlsx")
training_set.head()
test_set.head()

###############################################################################################################################################

# chechking the features in the Datasets

###############################################################################################################################################


#Training Set

print("\nEDA on Training Set\n")
print("#"*30)
print("\nFeatures/Columns : \n", training_set.columns)
print("\n\nNumber of Features/Columns : ", len(training_set.columns))
print("\nNumber of Rows : ",len(training_set))
print("\n\nData Types :\n", training_set.dtypes)
print("\nContains NaN/Empty cells : ", training_set.isnull().values.any())
print("\nTotal empty cells by column :\n", training_set.isnull().sum(), "\n\n")


# Test Set
print("#"*30)
print("\nEDA on Test Set\n")
print("#"*30)
print("\nFeatures/Columns : \n",test_set.columns)
print("\n\nNumber of Features/Columns : ",len(test_set.columns))
print("\nNumber of Rows : ",len(test_set))
print("\n\nData Types :\n", test_set.dtypes)
print("\nContains NaN/Empty cells : ", test_set.isnull().values.any())
print("\nTotal empty cells by column :\n", test_set.isnull().sum())

# Data Analysisng

###############################################################################################################################################


#Combining trainig set and test sets for analysing data and finding patterns

data_temp = [training_set[['TITLE', 'RESTAURANT_ID', 'CUISINES', 'TIME', 'CITY', 'LOCALITY','RATING', 'VOTES']], test_set]

data_temp = pd.concat(data_temp)


# Analysing Titles 

titles = list(data_temp['TITLE'])

# Finding Maximum number of titles mentioned in a single cell
maxim = 1
for i in titles :
    if len(i.split(',')) > maxim:
         maxim = len(i.split(','))
         
print("\n\nMaximum Titles in a Cell : ", maxim)    

all_titles = []

for i in titles :
    if len(i.split(',')) == 1:
         all_titles.append(i.split(',')[0].strip().upper())
    else :
        for it in range(len(i.split(','))):
            all_titles.append(i.split(',')[it].strip().upper())

print("\n\nNumber of Unique Titles : ", len(pd.Series(all_titles).unique()))
print("\n\nUnique Titles:\n", pd.Series(all_titles).unique())

all_titles = list(pd.Series(all_titles).unique())

# Analysing cuisines 

cuisines = list(data_temp['CUISINES'])

maxim = 1
for i in cuisines :
    if len(i.split(',')) > maxim:
         maxim = len(i.split(','))
         
print("\n\nMaximum cuisines in a Cell : ", maxim)    

all_cuisines = []

for i in cuisines :
    if len(i.split(',')) == 1:
         #print(i.split(',')[0])
         all_cuisines.append(i.split(',')[0].strip().upper())
    else :
        for it in range(len(i.split(','))):
            #print(i.split(',')[it])
            all_cuisines.append(i.split(',')[it].strip().upper())

print("\n\nNumber of Unique Cuisines : ", len(pd.Series(all_cuisines).unique()))
print("\n\nUnique Cuisines:\n", pd.Series(all_cuisines).unique())

all_cuisines = list(pd.Series(all_cuisines).unique())

# Analysing CITY

all_cities = list(data_temp['CITY'])

for i in range(len(all_cities)):
    if type(all_cities[i]) == float:
        all_cities[i] = 'NOT AVAILABLE'
    all_cities[i] = all_cities[i].strip().upper()
        
print("\n\nNumber of Unique cities (Including NOT AVAILABLE): ", len(pd.Series(all_cities).unique()))
print("\n\nUnique Cities:\n", pd.Series(all_cities).unique())
 
all_cities = list(pd.Series(all_cities).unique())


# Cleaning LOCALITY

all_localities = list(data_temp['LOCALITY'])

for i in range(len(all_localities)):
    if type(all_localities[i]) == float:
        all_localities[i] = 'NOT AVAILABLE'
    all_localities[i] = all_localities[i].strip().upper()
        
print("\n\nNumber of Unique Localities (Including NOT AVAILABLE) : ", len(pd.Series(all_localities).unique()))
print("\n\nUnique Localities:\n", pd.Series(all_localities).unique())

all_localities = list(pd.Series(all_localities).unique())

# Data Cleaning

###############################################################################################################################################


# Cleaning Training Set
#______________________

# TITLE


titles = list(training_set['TITLE'])

# Since Maximum number of titles in a cell is 2 will will split title in to 2 columns
T1 = []
T2 = []

for i in titles:
    T1.append(i.split(',')[0].strip().upper())
    try :
         T2.append(i.split(',')[1].strip().upper())
    except :
         T2.append('NONE')

# appending NONE to Unique titles list
all_titles.append('NONE')

#Cleaning CUISINES 

cuisines = list(training_set['CUISINES'])
   
# Since Maximum number of cuisines in a cell is 8 will will split title in to 8 columns
   
C1 = []
C2 = []
C3 = []
C4 = []
C5 = []
C6 = []
C7 = []
C8 = []


for i in cuisines:
        try :
            C1.append(i.split(',')[0].strip().upper())
        except :
            C1.append('NONE')
        try :
            C2.append(i.split(',')[1].strip().upper())
        except :
            C2.append('NONE')
        try :
            C3.append(i.split(',')[2].strip().upper())
        except :
            C3.append('NONE')
        try :
            C4.append(i.split(',')[3].strip().upper())
        except :
            C4.append('NONE')
        try :
            C5.append(i.split(',')[4].strip().upper())
        except :
            C5.append('NONE')
        try :
            C6.append(i.split(',')[5].strip().upper())
        except :
            C6.append('NONE')
        try :
            C7.append(i.split(',')[6].strip().upper())
        except :
            C7.append('NONE')
        try :
            C8.append(i.split(',')[7].strip().upper())
        except :
            C8.append('NONE')

# appending NONE to Unique cuisines list
all_cuisines.append('NONE')

# Cleaning CITY

cities = list(training_set['CITY'])

for i in range(len(cities)):
    if type(cities[i]) == float:
        cities[i] = 'NOT AVAILABLE'
    cities[i] = cities[i].strip().upper()
        

# Cleaning LOCALITY

localities = list(training_set['LOCALITY'])

for i in range(len(localities)):
    if type(localities[i]) == float:
        localities[i] = 'NOT AVAILABLE'
    localities[i] = localities[i].strip().upper()   
    

#Cleaning Rating

rates = list(training_set['RATING'])

for i in range(len(rates)) :
    try:
       rates[i] = float(rates[i])
    except :
       rates[i] = np.nan


# Votes
       
votes = list(training_set['VOTES'])

for i in range(len(votes)) :
    try:
       votes[i] = int(votes[i].split(" ")[0].strip())
    except :
       pass       
    
    

new_data_train = {}

new_data_train['TITLE1'] = T1
new_data_train['TITLE2'] = T2
new_data_train['RESTAURANT_ID'] = training_set["RESTAURANT_ID"]
new_data_train['CUISINE1'] = C1
new_data_train['CUISINE2'] = C2
new_data_train['CUISINE3'] = C3
new_data_train['CUISINE4'] = C4
new_data_train['CUISINE5'] = C5
new_data_train['CUISINE6'] = C6
new_data_train['CUISINE7'] = C7
new_data_train['CUISINE8'] = C8
new_data_train['CITY'] = cities
new_data_train['LOCALITY'] = localities
new_data_train['RATING'] = rates
new_data_train['VOTES'] = votes
new_data_train['COST'] = training_set["COST"]

new_data_train = pd.DataFrame(new_data_train)
#______________________



#______________________
# Cleaning Test Set
#______________________

# TITLE

titles = list(test_set['TITLE'])

# Since Maximum number of titles in a cell is 2 will will split title in to 2 columns
T1 = []
T2 = []

for i in titles:
    T1.append(i.split(',')[0].strip().upper())
    try :
         T2.append(i.split(',')[1].strip().upper())
    except :
         T2.append('NONE')


#Cleaning CUISINES 

cuisines = list(test_set['CUISINES'])
   
# Since Maximum number of cuisines in a cell is 8 will will split title in to 8 columns
   
C1 = []
C2 = []
C3 = []
C4 = []
C5 = []
C6 = []
C7 = []
C8 = []


for i in cuisines:
        try :
            C1.append(i.split(',')[0].strip().upper())
        except :
            C1.append('NONE')
        try :
            C2.append(i.split(',')[1].strip().upper())
        except :
            C2.append('NONE')
        try :
            C3.append(i.split(',')[2].strip().upper())
        except :
            C3.append('NONE')
        try :
            C4.append(i.split(',')[3].strip().upper())
        except :
            C4.append('NONE')
        try :
            C5.append(i.split(',')[4].strip().upper())
        except :
            C5.append('NONE')
        try :
            C6.append(i.split(',')[5].strip().upper())
        except :
            C6.append('NONE')
        try :
            C7.append(i.split(',')[6].strip().upper())
        except :
            C7.append('NONE')
        try :
            C8.append(i.split(',')[7].strip().upper())
        except :
            C8.append('NONE')


# Cleaning CITY

cities = list(test_set['CITY'])

for i in range(len(cities)):
    if type(cities[i]) == float:
        cities[i] = 'NOT AVAILABLE'
    cities[i] = cities[i].strip().upper()
        

# Cleaning LOCALITY

localities = list(test_set['LOCALITY'])

for i in range(len(localities)):
    if type(localities[i]) == float:
        localities[i] = 'NOT AVAILABLE'
    localities[i] = localities[i].strip().upper()   
    

#Cleaning Rating

rates = list(test_set['RATING'])

for i in range(len(rates)) :
    try:
       rates[i] = float(rates[i])
    except :
       rates[i] = np.nan


# Votes
       
votes = list(test_set['VOTES'])

for i in range(len(votes)) :
    try:
       votes[i] = int(votes[i].split(" ")[0].strip())
    except :
       pass       
    
    

new_data_test = {}

new_data_test['TITLE1'] = T1
new_data_test['TITLE2'] = T2
new_data_test['RESTAURANT_ID'] = test_set["RESTAURANT_ID"]
new_data_test['CUISINE1'] = C1
new_data_test['CUISINE2'] = C2
new_data_test['CUISINE3'] = C3
new_data_test['CUISINE4'] = C4
new_data_test['CUISINE5'] = C5
new_data_test['CUISINE6'] = C6
new_data_test['CUISINE7'] = C7
new_data_test['CUISINE8'] = C8
new_data_test['CITY'] = cities
new_data_test['LOCALITY'] = localities
new_data_test['RATING'] = rates
new_data_test['VOTES'] = votes

new_data_test = pd.DataFrame(new_data_test)

print("\n\nnew_data_train: \n", new_data_train.head())
print("\n\nnew_data_test: \n", new_data_test.head())

#______________________

from sklearn.ensemble import GradientBoostingRegressor

gbr=GradientBoostingRegressor( loss = 'huber',learning_rate=0.001,n_estimators=350, max_depth=6
                              ,subsample=1,
                              verbose=False,random_state=126)   # Leaderboard SCORE :  0.8364249755816828 @ RS =126 ,n_estimators=350, max_depth=6

gbr.fit(X_train,Y_train)

y_pred_gbr = sc.inverse_transform(gbr.predict(X_test))

y_pred_gbr = pd.DataFrame(y_pred_gbr, columns = ['COST']) # Converting to dataframe
print(y_pred_gbr)

y_pred_gbr.to_excel("GradientBoostingRegressor.xlsx", index = False ) # Saving the output in to an excel
