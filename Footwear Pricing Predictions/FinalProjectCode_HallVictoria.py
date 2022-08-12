#Victoria Hall
#DSC530
#Final Project
#11-20-2021




#libraries needed for analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mstats
from scipy import stats
pd.set_option('display.max_columns', None)
import os
from collections import Counter
import scipy
import dataframe_image as dfi
import thinkstats2
import thinkplot
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix


#Checking working directory and reading in data
os.getcwd()
path= r"C:\Users\hallt\Documents\GitHub\DSC530\Final Project"
os.chdir(path)
os.getcwd()
shoedata= pd.read_csv("Datafiniti_Womens_Shoes.csv")
shoedata.head().T


#Reviewing, subsetting and cleaning data
shoedata1 = shoedata[["id","brand","prices.isSale","prices.amountMin","prices.amountMax","prices.currency","prices.color"]]
shoedata1.head().T
##renaming columns as lower case values
shoedata1 = shoedata1.rename(columns={"prices.isSale" : "onsale","prices.amountMin": "minprice", "prices.amountMax": "maxprice", "prices.currency": "currency","prices.color" : "color"})
shoedata1.head().T
print(shoedata1.isnull().sum())
shoedata1["brand"] = shoedata1.brand.str.lower()
shoedata1["color"] = shoedata1.color.str.lower()
shoedata1.brand.unique()
shoedata1.onsale.unique()
shoedata1.currency.unique()
shoedata1.minprice.unique()
shoedata1.maxprice.unique()
shoedata1.minprice.min()
shoedata1.maxprice.max()

#There are over 500 different colors. For this analysis I am only going to look at colors where there are more than 50 instances of the color 
#Only including unique colors with more than 50 instances to prevent outlier effects
shoedata1.color.unique()
shoedata1.color.value_counts()
colorfreq = shoedata1.color.value_counts()>50
colorfreq = colorfreq[colorfreq] 
colorfreq = ["Black","Gray","Taupe","Navy","Brown","Nude","White","Blue","Stone","Tan","Pewter","Red","BlackWhite","Chestnut","Mushroom","Olive","DarkTan","Silver","Gold","Cognac","Wine","Pink","BlackFabric","Natural","BlackMicro","Slate","DarkGray","Blush","Sand","Whiskey","BlackPatent","DarkBrown"]
#Accidentally capitalized - converting back to lowercase
for i in range(len(colorfreq)):
    colorfreq[i] = colorfreq[i].lower()
boolean = shoedata1.color.isin(colorfreq)
shoedata1 = shoedata1[boolean]
shoedata1.head().T
shoedata1.color.value_counts()


#There are over 126 unique brands. I'm only going to look at brands where there is more than 10 instances.This is also going to take down the number of colors
#Only including unique brands with more than 10 instances to prevent outlier effects
shoedata1.brand.unique()
shoedata1.brand.value_counts()
brandfreq = shoedata1.brand.value_counts()>10
brandfreq = brandfreq[brandfreq]
brandfreq
brandfreq = ["2lipstoo","a2byaerosoles","adidas","apt.9","brinleyCo.","brinleyCo.Collection","candies","clarks","croftbarrow","dolcebymojomoxy","dr.scholls","eastland","easystreet","jenniferlopez","journeecollection","keds","kissesby2lipstoo","koolaburrabyugg","lclaurenconrad","lifestride","maddennyc","naturalsoulbynaturalizer","newbalance","nike","rocky4eursole]","simplyveraverawang","skechers","so","softstylebyhushpuppies","sonomagoodsforlife","springstep","stylecharlesbycharlesdavid","unionbay"]
boolean1 = shoedata.brand.isin(brandfreq)
shoedata1 = shoedata1[boolean1]
shoedata1.brand.value_counts()


#I'm also going to find the average price from the min and max prices
shoedata1["avgprice"] = (shoedata1.minprice + shoedata1.maxprice)/2
shoedata1.head().T
shoedata1.avgprice.min()
#Finding the new length of our dataframe
len(shoedata1)

#Finding frequencies of variables of all variables
list(shoedata1.columns)
brandfreq = shoedata1.brand.value_counts()
onsalefreq = shoedata1.onsale.value_counts()
minpricefreq = shoedata1.minprice.value_counts()
maxpricefreq = shoedata1.maxprice.value_counts()
currencyfreq = shoedata1.currency.value_counts()
colorfreq = shoedata1.color.value_counts()
avgpricefreq = shoedata1.avgprice.value_counts()
brandfreq
onsalefreq
minpricefreq
maxpricefreq
currencyfreq
colorfreq
avgpricefreq


#Making histograms. I will not create a histogram for onsale or currencies because there is only one value for these.
pd.Series(brandfreq).plot(kind='bar',title="Brand Histogram")
pd.Series(minpricefreq).plot(kind='hist',title="Minimum Price Histogram")
pd.Series(maxpricefreq).plot(kind='hist',title="Maximum Price Histogram")
pd.Series(colorfreq).plot(kind='bar',title="Color Histogram")
pd.Series(avgpricefreq).plot(kind='hist',title="Average Price Histogram")


#Checking for additional outliers in pricing and removing outliers with Z Scores greater than 3.
shoedata1.minprice.max()
shoedata1.maxprice.min()
shoedata1[["minprice"]].describe()
shoedata1["minz"] = np.abs(stats.zscore(shoedata1.minprice))
shoedata1["maxz"] = np.abs(stats.zscore(shoedata1.maxprice))
len(shoedata1)
shoedata1 = shoedata1[shoedata1.minz < 3]
len(shoedata1)
shoedata1 = shoedata1[shoedata1.maxz < 3]
len(shoedata1)


#Finding summary statistics for our numerical data and grouping by categorical data for more insight. 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 2)
pricesummarystats = pd.DataFrame(shoedata1[["minprice","maxprice","avgprice"]].describe())
dfi.export(pricesummarystats,'dataframe.png')
branddf=pd.DataFrame(shoedata1.groupby(["brand"])["maxprice"].describe())
branddf= branddf.style.set_caption("Average Price Summary Statistics by Brand")
dfi.export(branddf,"brandframe.png")
pd.DataFrame(shoedata1.groupby(["brand"])["avgprice"].describe())
colordf = shoedata1.groupby(["color"])["avgprice"].describe()
colordf= colordf.style.set_caption("Average Price Summary Statistics by Color")
dfi.export(colordf,"colordf.png")
median = np.median(shoedata1.avgprice)
mode = stats.mode(shoedata1.avgprice)
print("Avg Price Median:",median)
print("Avg Price Mode", mode)

#Psuedo Summary statistics for categorical data. Finding 'mode' and proportions
proportioncolor=shoedata1.color.value_counts()/shoedata1.color.value_counts().sum()
pd.set_option('display.float_format', lambda x: '%.2f' % x)
proportioncolor=pd.DataFrame(proportioncolor)
proportioncolor= proportioncolor.style.set_caption("Color Proportions")
dfi.export(proportioncolor,"colors.png")
proportionbrand=shoedata1.brand.value_counts()/shoedata1.brand.value_counts().sum()
proportionbrand=pd.DataFrame(proportionbrand)
proportionbrand= proportionbrand.style.set_caption("Brand Proportions")
dfi.export(proportionbrand,"brands.png")
shoedata1.brand.value_counts()
shoedata1.color.value_counts()

#PMF I am going to plot two PMFs. The first will be lifestride vs the balance of other brands avg price. The second is black vs other color. 
avgpricepmf = avgpricefreq.sort_index()/len(avgpricefreq)
lifestridedata = shoedata1[shoedata1.brand == "lifestride"]
otherbranddata = shoedata1[shoedata1.brand != "lifestride"]
otherbranddata.brand.unique()
lifestridedata.head()
lifestridefreq = lifestridedata.avgprice.value_counts()
lifestrideavgpricepmf = lifestridefreq.sort_index()/len(lifestridefreq)
avgpricepmf = np.abs(avgpricepmf)
lifestrideavgpricepmf = np.abs(lifestrideavgpricepmf)


#Brand PMF plot
plt.figure(1)
sns.histplot(lifestridedata.avgprice, stat="probability", bins=20, color = "blue",alpha = .3,label="Lifestride")
sns.histplot(otherbranddata.avgprice, stat="probability", bins=20, color = "red",alpha =.1, label="Other Brands")
plt.xlabel("Average Price")
plt.ylabel("PMF")
plt.title("Probability LifeStride vs. Other Brands")
plt.legend(loc="upper right")

#color PMF plot
shoedata1.color.unique()
blackdata = shoedata1[shoedata1.color == "black"]
blackdata
othercolordata = shoedata1[shoedata1.color != "black"]
blackfreq = blackdata.avgprice.value_counts()
blackfreq = np.abs(blackfreq)
plt.figure(2)
sns.histplot(blackdata.avgprice, stat="probability", bins=20, color = "black",alpha = .5,label="Black")
sns.histplot(othercolordata.avgprice, stat="probability", bins=20, color = "red",alpha =.1, label="Other Colors")
plt.xlabel("Average Price")
plt.ylabel("PMF")
plt.title("Probability Black vs. Other Other Colors")
plt.legend(loc="upper right")


#CDF I am going to create a CDF plot of the averageprice of product. 
#Finding the number of datapoints
N = len(shoedata1.avgprice)
count, bins_count = np.histogram(shoedata1.avgprice, bins=10)
#Finding PDF to calculate the CDF
pdf = count/ sum(count)
cdf = np.cumsum(pdf)

#Plotting CDF
plt.figure(2)
plt.plot(bins_count[1:], cdf, label="Average Price")
plt.legend()
plt.xlabel("Price")
plt.ylabel("CDF")
plt.title("CDF Average Prices")


#Analytical Distribution - Normal Distribution

#Plotting the normal distribution model
mu, var = thinkstats2.MeanVar(shoedata1.avgprice,ddof=0)
sigma = np.sqrt(var)
#choosing low and high values of avg price range
xs,ps = thinkstats2.RenderNormalCdf(mu,sigma, low = 20, high = 120)
cdf = thinkstats2.Cdf(shoedata1.avgprice, label = "Data")

#Plotting the normal distribution
plt.figure(3)
thinkplot.Plot(xs,ps, label = "model", color = "0.6")
thinkplot.PrePlot(1)
thinkplot.Cdf(cdf)
thinkplot.Config(title= "Average Prices", xlabel="Average Prices", ylabel="CDF")


#Normal Probability Plot
#range for STD
xs = [-4,4]
fxs, fys = thinkstats2.FitLine(xs, mu, sigma)
plt.figure(4)
thinkplot.Plot(fxs, fys, linewidth=4, color = "0.6")
xs, ys = thinkstats2.NormalProbability(shoedata1.avgprice)
thinkplot.Plot(xs,ys, label = "Data")
thinkplot.Config(title="Normal Probability Plot", xlabel="Standard Deviations from Mean", ylabel="Average Price")



#Scatter Plots
#Adding jitter to prevent overlap
plt.figure(6)
sns.stripplot(x="brand",y="avgprice", data=shoedata1, jitter=1)
plt.xticks(rotation = 90)
plt.title("Brand vs. Average Prices")

plt.figure(7)
sns.stripplot(x="color", y="avgprice", data=shoedata1, jitter=1)
plt.xticks(rotation=90)
plt.title("Color vs. Average Prices")

#Characterizing the relationships Min and Max price as an example. 
#Plotting min vs max price
plt.figure(8)
pricelist = shoedata1.maxprice.unique()
sns.stripplot(x="minprice", y="maxprice",data=shoedata1, jitter= .5)
plt.xticks(rotation=90)
plt.xticks([9.99,19.99,29.99, 39.99, 49.99, 59.99, 69.99, 79.99, 89.99],["9.99","19.99","29.99","39.99","49.99","59.99","69.99", "79.99", "89.99"])
plt.title("Min Price Vs. Max Price")

#Correlation, Covariance
corr,p = scipy.stats.pearsonr(shoedata1.minprice,shoedata1.maxprice)
pvalue = f"{p:.5}"
print(pvalue)
df1= shoedata1[["minprice","maxprice"]]

scorr, sp = scipy.stats.spearmanr(shoedata1.minprice,shoedata1.maxprice)
scorr
sp

#Covariance Array
cov1 = np.cov(df1)
cov1


#Hypothesis Testing 
#Difference in Means Black vs balance of colors. Using Student's T Test
blackdata.color.unique()
othercolordata.color.unique()
blackdata.avgprice.mean()
blackdata.avgprice.std()
othercolordata.avgprice.mean()
othercolordata.avgprice.std()
stat,p= scipy.stats.ttest_ind(blackdata.avgprice,othercolordata.avgprice)
alpha = 0.05
if p>alpha:
    print("Same distributions: Failed to Reject null hypothesis")
else:
    print("Different distributions: Reject null hypothesis")
print(stat)
print(p)



#Difference in Means Lifestride vs. balance of brands. Using Students T Test
lifestridedata.brand.unique()
otherbranddata.brand.unique()
lifestridedata.avgprice.mean()
lifestridedata.avgprice.std()
otherbranddata.avgprice.mean()
otherbranddata.avgprice.std()
statb, pb= scipy.stats.ttest_ind(lifestridedata.avgprice,otherbranddata.avgprice)
if pb>alpha:
    print("Same distributions: Failed to Reject null hypothesis")
else:
    print("Different distributions: Reject null hypothesis")
print(statb)
print(pb)

#Creating function for T Test
def studentttest(x,y):
    stat, p= scipy.stats.ttest_ind(x,y)
    alpha= 0.05
    if p>alpha:
        print("Same distributions: Failed to Reject null hypothesis")
    else:
        print("Different distributions: Reject null hypothesis")
    print(stat)
    print(p)


#Checking another brand
sodata = shoedata1[shoedata1.brand == "so"]
otherbranddata1 = shoedata1[shoedata1.brand != "so"]
studentttest(sodata.avgprice,otherbranddata1.avgprice)


#Regression Analysis
#Creating training and test sets
part_70 = shoedata1.sample(frac = 0.70)
part_30 = shoedata1.drop(part_70.index)
part_70.head()
type(part_70)
x = part_70[["brand", "color"]]

#Model formula - average price explained by brand and color. 
#Coding so that the model knows I'm using categorical data
est = smf.ols(formula = 'avgprice ~ C(brand)+C(color)', data=part_70).fit()
est.summary()
#Creating an exportable image of the regression results
plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(est.summary()), {'fontsize': 10}, fontproperties = 'monospace')
plt.axis('off')
plt.tight_layout()
plt.savefig('output.png')


#Checking accuracy
predictions = est.predict(part_30).tolist()
actuals = part_30["avgprice"].tolist()
len(predictions)
len(actuals)
differencepa = []

zip_object = zip(predictions, actuals)
for predictions_i, actuals_i in zip_object:
    differencepa.append(predictions_i-actuals_i)
    
differencepa = np.abs(differencepa)
differencepa.max()
differencepa.min()

#Checking to see how far off the predictions are. I end up check multiple values for k (the difference between the actual price and the predicted price)
k = 10
count = 0
for i in differencepa:
    if i <= k :
        count = count +1

count
len(differencepa)
k1 = 81/514
k5 = 292/514
k10 = 433/514
