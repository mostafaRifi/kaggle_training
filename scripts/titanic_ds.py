# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import os
# import pandas_profiling

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# set wd
os.getcwd()
os.chdir("/home/mos/Documents/python")

'''
The competition solution workflow goes through seven stages described in the Data Science Solutions book :

Question or problem definition.
Acquire training and testing data.
Wrangle, prepare, cleanse the data.
Analyze, identify patterns, and explore the data.
Model, predict and solve the problem.
Visualize, report, and present the problem solving steps and final solution.
Supply or submit the results.

The workflow indicates general sequence of how each stage may follow the other. However there are use cases with exceptions.

We may combine mulitple workflow stages. We may analyze by visualizing data.
Perform a stage earlier than indicated. We may analyze data before and after wrangling.
Perform a stage multiple times in our workflow. Visualize stage may be used multiple times.
Drop a stage altogether. We may not need supply stage to productize or service enable our dataset for a competition.
'''
################################################################################
#### Import data ####
################################################################################

# import titanic data (train & test data)
train_df = pd.read_csv('./kaggle_training/datas/train.csv')
test_df = pd.read_csv('./kaggle_training/datas/test.csv')

combine = [train_df, test_df]

################################################################################
#### Describe data ####
################################################################################

# view differents columns name
print(train_df.columns.values) 
"""
['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
 'Ticket' 'Fare' 'Cabin' 'Embarked']
"""

# columns types
train_df.dtypes
train_df.Embarked.nunique()
train_df.Embarked.value_counts()

# head & tail
train_df.head(10)
train_df.tail(10)

# some info about misses value in data 
train_df.info()
print('_'*40)
test_df.info()

# 
train_df.describe()

# 
train_df.describe(include=['O'])

'''
Assumtions based on data analysis
We arrive at following assumptions based on data analysis done so far. We may validate these assumptions further before taking appropriate actions.

Correlating.

We want to know how well does each feature correlate with Survival. We want to do this early in our project and match these quick correlations with modelled correlations later in the project.

Completing.

We may want to complete Age feature as it is definitely correlated to survival.
We may want to complete the Embarked feature as it may also correlate with survival or another important feature.

Correcting.

Ticket feature may be dropped from our analysis as it contains high ratio of duplicates (22%) and there may not be a correlation between Ticket and survival.
Cabin feature may be dropped as it is highly incomplete or contains many null values both in training and test dataset.
PassengerId may be dropped from training dataset as it does not contribute to survival.
Name feature is relatively non-standard, may not contribute directly to survival, so maybe dropped.

Creating.

We may want to create a new feature called Family based on Parch and SibSp to get total count of family members on board.
We may want to engineer the Name feature to extract Title as a new feature.
We may want to create new feature for Age bands. This turns a continous numerical feature into an ordinal categorical feature.
We may also want to create a Fare range feature if it helps our analysis.

Classifying.

We may also add to our assumptions based on the problem description noted earlier.

Women (Sex=female) were more likely to have survived.
Children (Age<?) were more likely to have survived.
The upper-class passengers (Pclass=1) were more likely to have survived.
'''

# Sample 
train_df.sample(5)
test_df.sample(5)

# Some info about data 
print ("The shape of the train_df data is (row, column):" + str(train_df.shape))
print (train_df.info())
print ("The shape of the test_df data is (row, column):" + str(test_df.shape))
print (test_df.info())

# Variables 

"""
### Categorical variable

# Nominal : variables that have two or more categories, but which do not have an intrinsic order

* Cabin
* Embarked (port embarkation)

# Dichotomous : Nominal variable with only two categories

* Sex (male or female)

# Ordinal (variables that have two or more categories just like nominal variables. Only the categories can also be ordered or ranked)

* Pclass (A proxy for socio-economic status (SES))

### Numerical variables

# Discrete

* PAssenger Id
* SibSp
* Parch
* Survived

# Continuous

* Age
* Fare (ticket price)

### Text variable

* Ticket (ticket number for passenger)
* Name (Name of passenger)
"""

# Some info about the train data

train_df.info()

"""
It looks like, the features have unequal amount of data entries for every column and they have many different types of variables. 
This can happen for the following reasons...

We may have missing values in our features.
We may have categorical features.
We may have alphanumerical or/and text features.
"""

################################################################################
#### Missing values ####
################################################################################

# Let's write a functin to print the total percentage of the missing values.(this can be a good exercise for beginners to try to write simple functions like this.)
def missing_percentage(df):
    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])
  

missing_percentage(train_df)
missing_percentage(test_df)

"""
We see that in both train, and test dataset have missing values. 
Let's make an effort to fill these missing values starting with "Embarked" feature.
"""

################################################################################
#### Fill data ####
################################################################################

def percent_value_counts(df, feature):
    """This function takes in a dataframe and a column and finds the percentage of the value_counts"""
    percent = pd.DataFrame(round(df.loc[:,feature].value_counts(dropna=False, normalize=True)*100,2))
    ## creating a df with th
    total = pd.DataFrame(df.loc[:,feature].value_counts(dropna=False))
    ## concating percent and total dataframe

    total.columns = ["Total"]
    percent.columns = ['Percent']
    return pd.concat([total, percent], axis = 1)
  
# Embarked feature
percent_value_counts(train_df, "Embarked")
percent_value_counts(test_df, "Embarked")  

"""
It looks like there are only two null values( ~ 0.22 %) in the Embarked feature, we can replace these with the mode value "S". 
However, let's dig a little deeper.
"""

test = train_df[train_df.Embarked.isnull()]


"""
We may be able to solve these two missing values by looking at other independent variables of the two raws. 
Both passengers paid a fare of $80, are of Pclass 1 and female Sex. 
Let's see how the Fare is distributed among all Pclass and Embarked feature values
"""

fig, ax = plt.subplots(figsize=(16,12), ncols=2)

ax1 = sns.boxplot(data=train_df, x="Embarked", y="Fare", hue="Pclass", ax = ax[0]);
ax2 = sns.boxplot(data=test_df, x="Embarked", y="Fare", hue="Pclass", ax = ax[1]);
ax1.set_title("Training Set", fontsize = 18)
ax2.set_title('Test Set',  fontsize = 18)

fig.show()

"""
Here, in both training set and test set, the average fare closest to $80 are in the C Embarked values where pclass is 1. 
So, let's fill in the missing values as "C"
"""


train_df.Embarked.fillna("C", inplace=True)
train_df.Embarked.value_counts()

# Cabin feature
percent_value_counts(train_df, "Cabin")
percent_value_counts(test_df, "Cabin")  

"""
Approximately 77% of Cabin feature is missing in the training data and 78% missing on the test data. We have two choices,

we can either get rid of the whole feature, or
we can brainstorm a little and find an appropriate way to put them in use. 
For example, We may say passengers with cabin record had a higher socio-economic-status then others. 
We may also say passengers with cabin record were more likely to be taken into consideration when loading into the boat.
Let's combine train and test data first and for now, will assign all the null values as "N"
"""


train_df.shape
test_df.shape

"""
Let's combine train and test data first and for now, will assign all the null values as "N"
"""

survivers = train_df.Survived

train_df.drop(["Survived"], axis=1, inplace=True)

train_df.shape

## Concat train and test into a variable "all_data"
all_data = pd.concat([train_df, test_df], ignore_index=False)

## Assign all the null values to N
all_data.Cabin.fillna("N", inplace=True)

"""
All the cabin names start with an English alphabet following by multiple digits. 
It seems like there are some passengers that had booked multiple cabin rooms in their name. 
This is because many of them travelled with family. However, they all seem to book under the same letter followed by different numbers.
It seems like there is a significance with the letters rather than the numbers. 
Therefore, we can group these cabins according to the letter of the cabin name.
"""

all_data.Cabin.value_counts()

all_data.Cabin = [i[0] for i in all_data.Cabin]

percent_value_counts(all_data, "Cabin")

""""
So, We still haven't done any effective work to replace the null values. 
Let's stop for a second here and think through how we can take advantage of some of the other features here.
We can use the average of the fare column We can use pythons groupby function to get the mean fare of each cabin letter.
"""

all_data.groupby(by=["Cabin"])["Fare"].mean().sort_values()

"""
Now, these means can help us determine the unknown cabins, if we compare each unknown cabin rows with the given mean's above. 
Let's write a simple function so that we can give cabin names based on the means.
"""

def cabin_estimator(i):
    """Grouping cabin feature by the first letter"""
    a = 0
    if i<16:
        a = "G"
    elif i>=16 and i<27:
        a = "F"
    elif i>=27 and i<38:
        a = "T"
    elif i>=38 and i<47:
        a = "A"
    elif i>= 47 and i<53:
        a = "E"
    elif i>= 53 and i<54:
        a = "D"
    elif i>=54 and i<116:
        a = 'C'
    else:
        a = "B"
    return a

"""
Let's apply cabin_estimator function in each unknown cabins(cabin with null values). 
Once that is done we will separate our train and test to continue towards machine learning modeling.
"""

with_N = all_data[all_data.Cabin == "N"]

without_N = all_data[all_data.Cabin != "N"]

## Applying cabin estimator function
with_N["Cabin"] = with_N.Fare.apply(lambda x: cabin_estimator(x))

all_data = pd.concat([with_N, without_N], axis=0)

## PassengerId helps us separate train and test. 
all_data.sort_values(by = 'PassengerId', inplace=True)

## Separating train and test from all_data. 
train = all_data[:891]

test = all_data[891:]

# adding saved target variable with train. 
train['Survived'] = survivers

# Fare feature

"""
Here, We can take the average of the Fare column to fill in the NaN value. 
However, for the sake of learning and practicing, we will try something else. 
We can take the average of the values where Pclass is 3, Sex is male and Embarked is S
"""
test[test.Fare.isnull()]

missing_value = test[(test.Pclass == 3) & (test.Sex == "male") & (test.Embarked == "S")].Fare.mean()

test.Fare.fillna(missing_value, inplace=True)

################################################################################
#### Visualize data ####
################################################################################

"""
Before we dive into finding relations between independent variables and our dependent variable(survivor), 
let us create some assumptions about how the relations may turn-out among features.

Assumptions:

Gender: More female survived than male
Pclass: Higher socio-economic status passenger survived more than others.
Age: Younger passenger survived more than other passengers.
Fare: Passenger with higher fare survived more that other passengers. 
This can be quite correlated with Pclass.
Now, let's see how the features are related to each other by creating some visualizations.
"""

## Gender and survived

# Survivor by sex
pal = {'male':"green", 'female':"Pink"}
sns.set(style="darkgrid")
plt.subplots(figsize=(15,8))
ax = sns.barplot(
    x="Sex", 
    y="Survived", 
    data=train, 
    palette=pal,
    linewidth=5,
    order=['female', 'male'],
    capsize=.05,
)
plt.title("Survived/Non-Survived Passenger by Gender Distribution", fontsize=25,loc='center', pad=40)
plt.ylabel("% of passenger survived", fontsize=15, )
plt.xlabel("Sex", fontsize=15);


"""
This bar plot above shows the distribution of female and male survived. 
The x_label represents Sex feature while the y_label represents the % of passenger survived. 
This bar plot shows that ~74% female passenger survived while only ~19% male passenger survived.
"""

# Survivor by sex
pal = {1:"seagreen", 0:"gray"}
sns.set(style="darkgrid")
plt.subplots(figsize = (15,8))
ax = sns.countplot(
    x="Sex", 
    hue="Survived",
    data=train, 
    palette = pal,
    linewidth=5,
)

## Fixing title, xlabel and ylabel
plt.title("Passenger Gender Distribution - Survived vs Not-survived", fontsize = 25, pad=40)
plt.xlabel("Sex", fontsize = 15);
plt.ylabel("# of Passenger Survived", fontsize = 15)

## Fixing xticks
#labels = ['Female', 'Male']
#plt.xticks(sorted(train.Sex.unique()), labels)

## Fixing legends
leg = ax.get_legend()
leg.set_title("Survived")
legs = leg.texts
legs[0].set_text("No")
legs[1].set_text("Yes")
plt.show()

"""
This count plot shows the actual distribution of male and female passengers that survived and did not survive. 
It shows that among all the females ~ 230 survived and ~ 70 did not survive. 
While among male passengers ~110 survived and ~480 did not survive.

Summary
#######

As we suspected, female passengers have survived at a much better rate than male passengers.
It seems about right since females and children were the priority.
"""

## Pclass and survived

# Survivor by sex
pal = {1:"green", 2:"pink", 3:"red"}
sns.set(style="darkgrid")
plt.subplots(figsize=(15,8))
ax = sns.barplot(
    x="Pclass", 
    y="Survived", 
    data=train, 
    palette=pal,
    linewidth=5,
    order=[1, 2, 3],
    capsize=.05,
)
plt.title("Survived/Non-Survived Passenger by Pclass Distribution", fontsize=25,loc='center', pad=40)
plt.ylabel("% of passenger survived", fontsize=15)
plt.xlabel("Pclass",fontsize=15);

# Survivor by Pclass
pal = {1:"seagreen", 0:"gray"}
sns.set(style="darkgrid")
plt.subplots(figsize = (15,8))
ax = sns.countplot(
    x="Pclass", 
    hue="Survived",
    data=train, 
    palette = pal,
    linewidth=5,
)

## Fixing title, xlabel and ylabel
plt.title("Passenger Pclass- Survived vs Not-survived", fontsize = 25, pad=40)
plt.xlabel("Pclass", fontsize = 15);
plt.ylabel("# of Passenger Survived", fontsize = 15)

## Fixing xticks
#labels = ['Female', 'Male']
#plt.xticks(sorted(train.Sex.unique()), labels)

## Fixing legends
leg = ax.get_legend()
leg.set_title("Survived")
legs = leg.texts
legs[0].set_text("No")
legs[1].set_text("Yes")
plt.show()