+++
categories = ["MovieLens EDA"]
comments = false
date = "2018-02-21T22:28:13-04:00"
draft = false
showpagemeta = true
showcomments = true
slug = ""
tags = ["Python", "Pandas", "EDA", "MovieLens"]
title = "Exploratory Analysis with MovieLens Data"
description = "MovieLens EDA with Pandas"

+++



### 1. Load Data into Pandas Dataframe


```python
import pandas as pd
import numpy as np

# pass in column names for each CSV and read them using pandas. 
# Column names available in the readme file

# Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep = '|', names = u_cols, encoding = 'latin-1')

# Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep = '\t', names = r_cols, encoding='latin-1')

# Reading items file:
i_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action',
          'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('ml-100k/u.item', sep = '|', names = i_cols, encoding = 'latin-1')
```


```python
print('Data size of Users: ', users.shape, '\n')
print(users.head(), '\n')
print(users.groupby('sex')['user_id'].size())
```

    Data size of Users:  (943, 5) 
    
       user_id  age sex  occupation zip_code
    0        1   24   M  technician    85711
    1        2   53   F       other    94043
    2        3   23   M      writer    32067
    3        4   24   M  technician    43537
    4        5   33   F       other    15213 
    
    sex
    F    273
    M    670
    dtype: int64



```python
print('Data size of Ratings: ', ratings.shape, '\n')
print(ratings.head(), '\n')
```

    Data size of Ratings:  (100000, 4) 
    
       user_id  movie_id  rating  unix_timestamp
    0      196       242       3       881250949
    1      186       302       3       891717742
    2       22       377       1       878887116
    3      244        51       2       880606923
    4      166       346       1       886397596 
    



```python
print('Data size of Movies: ', movies.shape, '\n')
print(movies.head(), '\n')
movies.info()
```

    Data size of Movies:  (1682, 24) 
    
       movie_id        movie_title release_date  video_release_date  \
    0         1   Toy Story (1995)  01-Jan-1995                 NaN   
    1         2   GoldenEye (1995)  01-Jan-1995                 NaN   
    2         3  Four Rooms (1995)  01-Jan-1995                 NaN   
    3         4  Get Shorty (1995)  01-Jan-1995                 NaN   
    4         5     Copycat (1995)  01-Jan-1995                 NaN   
    
                                                IMDb_URL  unknown  Action  \
    0  http://us.imdb.com/M/title-exact?Toy%20Story%2...        0       0   
    1  http://us.imdb.com/M/title-exact?GoldenEye%20(...        0       1   
    2  http://us.imdb.com/M/title-exact?Four%20Rooms%...        0       0   
    3  http://us.imdb.com/M/title-exact?Get%20Shorty%...        0       1   
    4  http://us.imdb.com/M/title-exact?Copycat%20(1995)        0       0   
    
       Adventure  Animation  Children's   ...     Fantasy  Film-Noir  Horror  \
    0          0          1           1   ...           0          0       0   
    1          1          0           0   ...           0          0       0   
    2          0          0           0   ...           0          0       0   
    3          0          0           0   ...           0          0       0   
    4          0          0           0   ...           0          0       0   
    
       Musical  Mystery  Romance  Sci-Fi  Thriller  War  Western  
    0        0        0        0       0         0    0        0  
    1        0        0        0       0         1    0        0  
    2        0        0        0       0         1    0        0  
    3        0        0        0       0         0    0        0  
    4        0        0        0       0         1    0        0  
    
    [5 rows x 24 columns] 
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1682 entries, 0 to 1681
    Data columns (total 24 columns):
    movie_id              1682 non-null int64
    movie_title           1682 non-null object
    release_date          1681 non-null object
    video_release_date    0 non-null float64
    IMDb_URL              1679 non-null object
    unknown               1682 non-null int64
    Action                1682 non-null int64
    Adventure             1682 non-null int64
    Animation             1682 non-null int64
    Children's            1682 non-null int64
    Comedy                1682 non-null int64
    Crime                 1682 non-null int64
    Documentary           1682 non-null int64
    Drama                 1682 non-null int64
    Fantasy               1682 non-null int64
    Film-Noir             1682 non-null int64
    Horror                1682 non-null int64
    Musical               1682 non-null int64
    Mystery               1682 non-null int64
    Romance               1682 non-null int64
    Sci-Fi                1682 non-null int64
    Thriller              1682 non-null int64
    War                   1682 non-null int64
    Western               1682 non-null int64
    dtypes: float64(1), int64(20), object(3)
    memory usage: 315.5+ KB


**Merge all three informaton together as a dataframe**


```python
# create one merged DataFrame
movie_ratings = pd.merge(movies, ratings)
lens = pd.merge(movie_ratings, users)
```


```python
print('Size of movie_ratings: ', movie_ratings.shape, '\n')
print(movie_ratings.head(), '\n')
```

    Size of movie_ratings:  (100000, 27) 
    
       movie_id       movie_title release_date  video_release_date  \
    0         1  Toy Story (1995)  01-Jan-1995                 NaN   
    1         1  Toy Story (1995)  01-Jan-1995                 NaN   
    2         1  Toy Story (1995)  01-Jan-1995                 NaN   
    3         1  Toy Story (1995)  01-Jan-1995                 NaN   
    4         1  Toy Story (1995)  01-Jan-1995                 NaN   
    
                                                IMDb_URL  unknown  Action  \
    0  http://us.imdb.com/M/title-exact?Toy%20Story%2...        0       0   
    1  http://us.imdb.com/M/title-exact?Toy%20Story%2...        0       0   
    2  http://us.imdb.com/M/title-exact?Toy%20Story%2...        0       0   
    3  http://us.imdb.com/M/title-exact?Toy%20Story%2...        0       0   
    4  http://us.imdb.com/M/title-exact?Toy%20Story%2...        0       0   
    
       Adventure  Animation  Children's       ...        Musical  Mystery  \
    0          0          1           1       ...              0        0   
    1          0          1           1       ...              0        0   
    2          0          1           1       ...              0        0   
    3          0          1           1       ...              0        0   
    4          0          1           1       ...              0        0   
    
       Romance  Sci-Fi  Thriller  War  Western  user_id  rating  unix_timestamp  
    0        0       0         0    0        0      308       4       887736532  
    1        0       0         0    0        0      287       5       875334088  
    2        0       0         0    0        0      148       4       877019411  
    3        0       0         0    0        0      280       4       891700426  
    4        0       0         0    0        0       66       3       883601324  
    
    [5 rows x 27 columns] 
    



```python
print('Size of lens: ', lens.shape, '\n')
print(lens.head(), '\n')
```

    Size of lens:  (100000, 31) 
    
       movie_id            movie_title release_date  video_release_date  \
    0         1       Toy Story (1995)  01-Jan-1995                 NaN   
    1         4      Get Shorty (1995)  01-Jan-1995                 NaN   
    2         5         Copycat (1995)  01-Jan-1995                 NaN   
    3         7  Twelve Monkeys (1995)  01-Jan-1995                 NaN   
    4         8            Babe (1995)  01-Jan-1995                 NaN   
    
                                                IMDb_URL  unknown  Action  \
    0  http://us.imdb.com/M/title-exact?Toy%20Story%2...        0       0   
    1  http://us.imdb.com/M/title-exact?Get%20Shorty%...        0       1   
    2  http://us.imdb.com/M/title-exact?Copycat%20(1995)        0       0   
    3  http://us.imdb.com/M/title-exact?Twelve%20Monk...        0       0   
    4     http://us.imdb.com/M/title-exact?Babe%20(1995)        0       0   
    
       Adventure  Animation  Children's    ...     Thriller  War  Western  \
    0          0          1           1    ...            0    0        0   
    1          0          0           0    ...            0    0        0   
    2          0          0           0    ...            1    0        0   
    3          0          0           0    ...            0    0        0   
    4          0          0           1    ...            0    0        0   
    
       user_id  rating  unix_timestamp  age  sex  occupation  zip_code  
    0      308       4       887736532   60    M     retired     95076  
    1      308       5       887737890   60    M     retired     95076  
    2      308       4       887739608   60    M     retired     95076  
    3      308       4       887738847   60    M     retired     95076  
    4      308       5       887736696   60    M     retired     95076  
    
    [5 rows x 31 columns] 
    


Now we have to divide the ratings data set into test and train data for making models. Luckily GroupLens provides pre-divided data where in the test data has 10 ratings for each user, i.e. 9430 rows in total. Lets load that:


```python
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('ml-100k/ua.base', sep = '\t', names = r_cols, encoding = 'latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep = '\t', names = r_cols, encoding = 'latin-1')
print('Size of ratings_train: ', ratings_train.shape)
print('Size of ratings_test: ', ratings_test.shape)
```

    Size of ratings_train:  (90570, 4)
    Size of ratings_test:  (9430, 4)



```python
train = ratings_train
test = ratings_test
```

### 2. Rating Aggregations

**Top 20 movies with the highest average ratings**


```python
top_20_in_avg_rts = movie_ratings.groupby('movie_title')['rating'].mean().sort_values(ascending = False).head(20)
print(type(top_20_in_avg_rts), '\n')
print(top_20_in_avg_rts)
```

    <class 'pandas.core.series.Series'> 
    
    movie_title
    Marlene Dietrich: Shadow and Light (1996)                 5.000000
    Prefontaine (1997)                                        5.000000
    Santa with Muscles (1996)                                 5.000000
    Star Kid (1997)                                           5.000000
    Someone Else's America (1995)                             5.000000
    Entertaining Angels: The Dorothy Day Story (1996)         5.000000
    Saint of Fort Washington, The (1993)                      5.000000
    Great Day in Harlem, A (1994)                             5.000000
    They Made Me a Criminal (1939)                            5.000000
    Aiqing wansui (1994)                                      5.000000
    Pather Panchali (1955)                                    4.625000
    Anna (1996)                                               4.500000
    Everest (1998)                                            4.500000
    Maya Lin: A Strong Clear Vision (1994)                    4.500000
    Some Mother's Son (1996)                                  4.500000
    Close Shave, A (1995)                                     4.491071
    Schindler's List (1993)                                   4.466443
    Wrong Trousers, The (1993)                                4.466102
    Casablanca (1942)                                         4.456790
    Wallace & Gromit: The Best of Aardman Animation (1996)    4.447761
    Name: rating, dtype: float64


**Top 20 movies with the highest average ratings and corresponding rating count**


```python
top_20_in_rating_avg_count = ratings.groupby('movie_id') \
                                    .agg({'rating': ['mean', 'count', 'std']}) \
                                    .sort_values(by = ('rating', 'mean'), ascending = False)

print('Data type after aggregating for two metrics: ', type(top_20_in_rating_avg_count), '\n')
print('Column names for the new dataframe: ', top_20_in_rating_avg_count.columns.values, '\n')
print(top_20_in_rating_avg_count.head(20))
```

    Data type after aggregating for two metrics:  <class 'pandas.core.frame.DataFrame'> 
    
    Column names for the new dataframe:  [('rating', 'mean') ('rating', 'count') ('rating', 'std')] 
    
                rating                
                  mean count       std
    movie_id                          
    814       5.000000     1       NaN
    1599      5.000000     1       NaN
    1201      5.000000     1       NaN
    1122      5.000000     1       NaN
    1653      5.000000     1       NaN
    1293      5.000000     3  0.000000
    1500      5.000000     2  0.000000
    1189      5.000000     3  0.000000
    1536      5.000000     1       NaN
    1467      5.000000     2  0.000000
    1449      4.625000     8  0.517549
    119       4.500000     4  1.000000
    1398      4.500000     2  0.707107
    1642      4.500000     2  0.707107
    1594      4.500000     2  0.707107
    408       4.491071   112  0.771047
    318       4.466443   298  0.829109
    169       4.466102   118  0.823607
    483       4.456790   243  0.728114
    114       4.447761    67  0.764429


**Average rating group by user_id**


```python
ratings_train.groupby(['user_id'])['rating'].mean().sort_values(ascending = False).head(20)
```




    user_id
    688    4.928571
    849    4.846154
    507    4.708333
    225    4.647059
    583    4.647059
    928    4.636364
    118    4.622951
    907    4.605839
    628    4.588235
    469    4.575758
    252    4.545455
    686    4.540984
    850    4.536585
    427    4.523810
    513    4.500000
    330    4.496350
    767    4.481481
    136    4.480000
    522    4.450000
    383    4.442623
    Name: rating, dtype: float64



**25 Most Rated Movies**


```python
# most rated movies with rating mean and standard deviation
movie_stats = lens.groupby(['movie_id', 'movie_title']) \
                  .agg({'rating': ['count', 'mean', 'std']}) \
                  .sort_values(by = ('rating', 'count'), ascending = False)
        
# filter out movies with rating counts less than 100
at_least100 = movie_stats[movie_stats['rating']['count'] >= 100]
at_least100.sort_values([('rating', 'count')], ascending=False)[:15]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">rating</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>movie_id</th>
      <th>movie_title</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <th>Star Wars (1977)</th>
      <td>583</td>
      <td>4.358491</td>
      <td>0.881341</td>
    </tr>
    <tr>
      <th>258</th>
      <th>Contact (1997)</th>
      <td>509</td>
      <td>3.803536</td>
      <td>0.994427</td>
    </tr>
    <tr>
      <th>100</th>
      <th>Fargo (1996)</th>
      <td>508</td>
      <td>4.155512</td>
      <td>0.975756</td>
    </tr>
    <tr>
      <th>181</th>
      <th>Return of the Jedi (1983)</th>
      <td>507</td>
      <td>4.007890</td>
      <td>0.923955</td>
    </tr>
    <tr>
      <th>294</th>
      <th>Liar Liar (1997)</th>
      <td>485</td>
      <td>3.156701</td>
      <td>1.098544</td>
    </tr>
    <tr>
      <th>286</th>
      <th>English Patient, The (1996)</th>
      <td>481</td>
      <td>3.656965</td>
      <td>1.169401</td>
    </tr>
    <tr>
      <th>288</th>
      <th>Scream (1996)</th>
      <td>478</td>
      <td>3.441423</td>
      <td>1.113910</td>
    </tr>
    <tr>
      <th>1</th>
      <th>Toy Story (1995)</th>
      <td>452</td>
      <td>3.878319</td>
      <td>0.927897</td>
    </tr>
    <tr>
      <th>300</th>
      <th>Air Force One (1997)</th>
      <td>431</td>
      <td>3.631090</td>
      <td>0.998072</td>
    </tr>
    <tr>
      <th>121</th>
      <th>Independence Day (ID4) (1996)</th>
      <td>429</td>
      <td>3.438228</td>
      <td>1.116584</td>
    </tr>
    <tr>
      <th>174</th>
      <th>Raiders of the Lost Ark (1981)</th>
      <td>420</td>
      <td>4.252381</td>
      <td>0.891819</td>
    </tr>
    <tr>
      <th>127</th>
      <th>Godfather, The (1972)</th>
      <td>413</td>
      <td>4.283293</td>
      <td>0.934577</td>
    </tr>
    <tr>
      <th>56</th>
      <th>Pulp Fiction (1994)</th>
      <td>394</td>
      <td>4.060914</td>
      <td>1.150880</td>
    </tr>
    <tr>
      <th>7</th>
      <th>Twelve Monkeys (1995)</th>
      <td>392</td>
      <td>3.798469</td>
      <td>0.982037</td>
    </tr>
    <tr>
      <th>98</th>
      <th>Silence of the Lambs, The (1991)</th>
      <td>390</td>
      <td>4.289744</td>
      <td>0.836597</td>
    </tr>
  </tbody>
</table>
</div>



**Distribution of User Ages**


```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.hist(users.age, bins = 30, edgecolor = "k")

plt.style.use('seaborn')
plt.title("Distribution of users' ages")
plt.ylabel('count of users')
plt.xlabel('age')
plt.show()
```


![png](img/output_23_0.png)


We can see that most users who gave movie ratings are in the 20-40 age buckets. We are not going deeper into the user age factor, but we will look into the rating behavior difference by gender

### 3. Visualizations for Rating Difference by Gender


```python
# movie rating mean and size break down by reviewer gender
pivoted = lens.pivot_table(index =['movie_id', 'movie_title'],
                           columns = ['sex'],
                           values = 'rating',
                           aggfunc = [np.mean, np.size],
                           fill_value = 0)

print(type(pivoted), '\n')
print('Dataframe size for pivoted: ', pivoted.shape, '\n')
print('Column names for pivoted: ', pivoted.columns.values)
```

    <class 'pandas.core.frame.DataFrame'> 
    
    Dataframe size for pivoted:  (1682, 4) 
    
    Column names for pivoted:  [('mean', 'F') ('mean', 'M') ('size', 'F') ('size', 'M')]



```python
pivoted.sort_values(by = ('mean', 'M'), ascending = False).head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">mean</th>
      <th colspan="2" halign="left">size</th>
    </tr>
    <tr>
      <th></th>
      <th>sex</th>
      <th>F</th>
      <th>M</th>
      <th>F</th>
      <th>M</th>
    </tr>
    <tr>
      <th>movie_id</th>
      <th>movie_title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1175</th>
      <th>Hugo Pool (1997)</th>
      <td>2.333333</td>
      <td>5.0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1605</th>
      <th>Love Serenade (1996)</th>
      <td>2.333333</td>
      <td>5.0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1536</th>
      <th>Aiqing wansui (1994)</th>
      <td>0.000000</td>
      <td>5.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1144</th>
      <th>Quiet Room, The (1996)</th>
      <td>3.000000</td>
      <td>5.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1293</th>
      <th>Star Kid (1997)</th>
      <td>0.000000</td>
      <td>5.0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1189</th>
      <th>Prefontaine (1997)</th>
      <td>5.000000</td>
      <td>5.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1306</th>
      <th>Delta of Venus (1994)</th>
      <td>1.000000</td>
      <td>5.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1191</th>
      <th>Letter From Death Row, A (1998)</th>
      <td>4.000000</td>
      <td>5.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>814</th>
      <th>Great Day in Harlem, A (1994)</th>
      <td>0.000000</td>
      <td>5.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1653</th>
      <th>Entertaining Angels: The Dorothy Day Story (1996)</th>
      <td>0.000000</td>
      <td>5.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
pivoted[(pivoted['size', 'F'] + pivoted['size', 'M']) >= 20].shape
```




    (939, 4)




```python
# rating mean difference by gender
pivoted['mean', 'diff'] = pivoted['mean', 'M'] - pivoted['mean', 'F']
# rating count ratio within gender and then compare the difference as population difference by gender
pivoted['size', 'ratio'] = (pivoted['size', 'M'] / 670) - (pivoted['size', 'F'] / 273)
pivoted.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">mean</th>
      <th colspan="2" halign="left">size</th>
      <th>mean</th>
      <th>size</th>
    </tr>
    <tr>
      <th></th>
      <th>sex</th>
      <th>F</th>
      <th>M</th>
      <th>F</th>
      <th>M</th>
      <th>diff</th>
      <th>ratio</th>
    </tr>
    <tr>
      <th>movie_id</th>
      <th>movie_title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <th>Toy Story (1995)</th>
      <td>3.789916</td>
      <td>3.909910</td>
      <td>119</td>
      <td>333</td>
      <td>0.119994</td>
      <td>0.061117</td>
    </tr>
    <tr>
      <th>2</th>
      <th>GoldenEye (1995)</th>
      <td>3.368421</td>
      <td>3.178571</td>
      <td>19</td>
      <td>112</td>
      <td>-0.189850</td>
      <td>0.097567</td>
    </tr>
    <tr>
      <th>3</th>
      <th>Four Rooms (1995)</th>
      <td>2.687500</td>
      <td>3.108108</td>
      <td>16</td>
      <td>74</td>
      <td>0.420608</td>
      <td>0.051840</td>
    </tr>
    <tr>
      <th>4</th>
      <th>Get Shorty (1995)</th>
      <td>3.400000</td>
      <td>3.591463</td>
      <td>45</td>
      <td>164</td>
      <td>0.191463</td>
      <td>0.079941</td>
    </tr>
    <tr>
      <th>5</th>
      <th>Copycat (1995)</th>
      <td>3.772727</td>
      <td>3.140625</td>
      <td>22</td>
      <td>64</td>
      <td>-0.632102</td>
      <td>0.014936</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Top 50 most rated movies
most_50 = movie_stats[:50]
most_50.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">rating</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>movie_id</th>
      <th>movie_title</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <th>Star Wars (1977)</th>
      <td>583</td>
      <td>4.358491</td>
      <td>0.881341</td>
    </tr>
    <tr>
      <th>258</th>
      <th>Contact (1997)</th>
      <td>509</td>
      <td>3.803536</td>
      <td>0.994427</td>
    </tr>
    <tr>
      <th>100</th>
      <th>Fargo (1996)</th>
      <td>508</td>
      <td>4.155512</td>
      <td>0.975756</td>
    </tr>
    <tr>
      <th>181</th>
      <th>Return of the Jedi (1983)</th>
      <td>507</td>
      <td>4.007890</td>
      <td>0.923955</td>
    </tr>
    <tr>
      <th>294</th>
      <th>Liar Liar (1997)</th>
      <td>485</td>
      <td>3.156701</td>
      <td>1.098544</td>
    </tr>
  </tbody>
</table>
</div>




```python
pivoted.reset_index('movie_id', inplace = True)
most_50.reset_index('movie_id', inplace = True)
print(pivoted.head(), '\n')
print(most_50.head(), '\n')
```

                      movie_id      mean           size           mean      size
    sex                                F         M    F    M      diff     ratio
    movie_title                                                                 
    Toy Story (1995)         1  3.789916  3.909910  119  333  0.119994  0.061117
    GoldenEye (1995)         2  3.368421  3.178571   19  112 -0.189850  0.097567
    Four Rooms (1995)        3  2.687500  3.108108   16   74  0.420608  0.051840
    Get Shorty (1995)        4  3.400000  3.591463   45  164  0.191463  0.079941
    Copycat (1995)           5  3.772727  3.140625   22   64 -0.632102  0.014936 
    
                              movie_id rating                    
                                        count      mean       std
    movie_title                                                  
    Star Wars (1977)                50    583  4.358491  0.881341
    Contact (1997)                 258    509  3.803536  0.994427
    Fargo (1996)                   100    508  4.155512  0.975756
    Return of the Jedi (1983)      181    507  4.007890  0.923955
    Liar Liar (1997)               294    485  3.156701  1.098544 
    



```python
disagreements = pivoted[pivoted.index.isin(most_50.index)]['mean', 'diff']
print(disagreements.head())
disagreements.sort_values().plot(kind='barh', figsize=[9, 15])
plt.title('Male vs. Female Avg. Ratings\n(Difference > 0 = Favored by Men)')
plt.ylabel('Movie Title')
plt.xlabel('Average Rating Difference')
plt.show()
```

    movie_title
    Toy Story (1995)             0.119994
    Twelve Monkeys (1995)        0.300315
    Dead Man Walking (1995)     -0.043452
    Mr. Holland's Opus (1995)   -0.244160
    Braveheart (1995)            0.031136
    Name: (mean, diff), dtype: float64



![png](img/output_32_1.png)



```python
popularity_diff = pivoted[pivoted.index.isin(most_50.index)]['size', 'ratio']
print(popularity_diff.head())
popularity_diff.sort_values().plot(kind='barh', figsize=[9, 15])
plt.title('Male vs. Female Rating Size Ratio\n(Ratio > 0: more popular in Men, else: more popular in Women)')
plt.ylabel('Movie Title')
plt.xlabel('Rating Population Ratio Difference between Men and Women')
plt.show()
```

    movie_title
    Toy Story (1995)             0.061117
    Twelve Monkeys (1995)        0.162320
    Dead Man Walking (1995)      0.018359
    Mr. Holland's Opus (1995)   -0.036996
    Braveheart (1995)            0.118485
    Name: (size, ratio), dtype: float64



![png](img/output_33_1.png)


Between the two bar charts, we can see that The Terminator (1984) is mostly favored by men over women in terms of average ratings, but it is not the top 1 movie that mostly watched by men, while Star Trek: First Contact (1996) is.

Next, it would be the recommender system based on the MovieLens data. This blog is very helpful: [Intro to Recommender Systems: Collaborative Filtering](http://blog.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/)