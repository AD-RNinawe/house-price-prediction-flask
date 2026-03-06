import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# Load the dataset
df1=pd.read_csv('bengaluru_house_prices.csv')
df1.head()
df1.shape
df1.columns

# DATA CLEANING
# Drop unnecessary columns
df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns', inplace=True)
df1.shape

# Check for missing values
df1.isnull().sum()
df1.shape

# Drop rows with missing values
df3 = df1.dropna()
df3.isnull().sum()
df3.shape

#FEATURE ENGINEERING
# Convert 'size' column to number of bedrooms (bhk)
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()

# Convert 'total_sqft' to numeric
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

df3[~df3['total_sqft'].apply(is_float)].head(10)

def convert_sqft_to_num(x):
    try:
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None
    
df3['total_sqft'] = df3['total_sqft'].apply(convert_sqft_to_num)
df3 = df3[df3['total_sqft'].notnull()]

df4=df3.copy()
df4.head()
df4.loc[30]

# Create 'price_per_sqft' column
df4['price_per_sqft'] = (df4['price']*100000)/df4['total_sqft']
df4.head()

df4_stats = df4['price_per_sqft'].describe()
df4_stats

# Reduce dimensionality of 'location' column
df4.location = df4.location.apply(lambda x: x.strip())
location_stats = df4['location'].value_counts(ascending=False)
location_stats
len(location_stats[location_stats<=10])
len(df4.location.unique())

# Group locations with less than 10 occurrences as 'other'
location_stats_less_than_10 = location_stats[location_stats<=10]
df4.location = df4.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df4.location.unique())
df4.head(10)

# OUTLIER REMOVAL
# Remove outliers based on 'total_sqft' to 'bhk' ratio
df5 = df4[~(df4.total_sqft/df4.bhk<300)]
df5.shape
df5.price_per_sqft.describe()

# Remove outliers based on 'price_per_sqft'
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df6 = remove_pps_outliers(df5)
df6.shape

# Remove outliers based on 'bhk' and 'price_per_sqft'
def plot_scatter_chart(location, df):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft, bhk2.price_per_sqft, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price_per_sqft, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price Per Square Feet")
    plt.title(location)
    plt.legend()
    plt.show()

plot_scatter_chart("Rajaji Nagar", df6)

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(
                    exclude_indices,
                    bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values
                )
    return df.drop(exclude_indices, axis='index')

df7 = remove_bhk_outliers(df6)
df7.shape
plot_scatter_chart("Rajaji Nagar", df7)

# Remove outliers based on bathroom count
df7.bath.unique()
df7[df7.bath>10]
df8 = df7[df7.bath < (df7.bhk + 2)]
df8.shape

df8.drop(['size', 'price_per_sqft'], axis='columns', inplace=True)
df8.head()

# ONE HOT ENCODING
dummies = pd.get_dummies(df8.location)
dummies.head()
df9 = pd.concat([df8, dummies.drop('other', axis='columns')], axis='columns')
df9.head()
df9.shape
df10 = df9.drop('location', axis='columns')
df10.head()
df10.shape

# MODEL BUILDING
X = df10.drop('price', axis='columns')
y = df10.price

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Train a Linear Regression model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

# Testing the model
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns==location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    
    return lr.predict([x])[0]
predict_price('1st Phase JP Nagar', 1000, 2, 2)
predict_price('Indira Nagar', 1000, 2, 2)
predict_price('Vijayanagar', 1000, 3, 3)

# Save the model using pickle
import pickle
with open('bengaluru_house_price_model.pickle', 'wb') as f:
    pickle.dump(lr, f)

# Save the columns used in the model
import json
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open("bengaluru_house_price_columns.json", "w") as f:
    f.write(json.dumps(columns))

# The code shows the basic steps to build a house price prediction model for Bengaluru using linear regression.
