##############################################################
# Rating Product & Sorting Reviews and Keywords in Amazon
##############################################################

##############################################################
# 1. Business Problem
##############################################################

# Accurate product ratings and review sorting are crucial challenges in e-commerce. This project aims to enhance
# customer satisfaction and boost product visibility by implementing precise rating systems and effective review
# sorting mechanisms. By building trust and fostering a healthy trading environment, it benefits both customers and
# sellers. Additionally, providing users with a quick overview and easy review scanning improves the overall shopping
# experience.

# Variables
# reviewerID: User ID
# asin: Product ID
# reviewerName: Username
# helpful: Helpful rating score [helpful_yes,total_vote]
# reviewText: Review text
# overall: Product rating
# summary: Review summary
# unixReviewTime: Review time (in Unix format)
# reviewTime: Review time (raw format)
# day_diff: Number of days since the review was posted
# helpful_yes: Number of times the review was found helpful
# total_vote: Total number of votes received for the review

###############################################################
# 2. Data Preparation
###############################################################

# Importing libraries
##############################################
import pandas as pd
import math
import scipy.stats as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv('/Users/handeatasagun/Documents/Github/Amazon_Rating_Review_Keyword/amazon_review.csv')


# Data understanding
##############################################
def check_df(dataframe, head=5):
    print('################# Shape ################# ')
    print(dataframe.columns)
    print('################# Types  ################# ')
    print(dataframe.dtypes)
    print('##################  Head ################# ')
    print(dataframe.head(head))
    print('#################  Shape ################# ')
    print(dataframe.shape)
    print('#################  NA ################# ')
    print(dataframe.isnull().sum())
    print('#################  Quantiles ################# ')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99]).T)
    print('')


check_df(df)

# Remove rows with missing reviewer ID and reviewText
df = df.dropna(subset=['reviewerID', 'reviewText'])

#####################################################
# 1. Calculating Rating of Product
#####################################################

# Average rating
##############################################
df['overall'].value_counts()
df['overall'].mean()


# Weighted Average Rating by Date
##############################################
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w3 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 180), "overall"].mean() * w4 / 100


time_based_weighted_average(df)

#####################################################
# 2. Sorting Reviews
#####################################################

# Up-Down Diff Score = (up ratings) − (down ratings)
#####################################################

df['helpful_no'] = df['total_vote'] - df['helpful_yes']


def score_pos_neg_diff(pos, neg):
    df['score_pos_neg_diff'] = df[pos] - df[neg]
    return df


score_pos_neg_diff('helpful_yes', 'helpful_no').head(5)


# Score = Average rating = (up ratings) / (all ratings)
########################################################
def score_average_rating(pos, neg):
    df['score_average_rating'] = df[pos] / (df[pos] + df[neg])
    df['score_average_rating'] = df['score_average_rating'].fillna(0)  # for ZeroDivisionError
    return df


score_average_rating('helpful_yes', 'helpful_no').head(5)


# Wilson lower bound
########################################################
def wilson_lower_bound(pos, neg, confidence=0.95):
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

# Sort the DataFrame by the Wilson Lower Bound in descending order
###################################################################
top20_reviews = df.sort_values("wilson_lower_bound", ascending=False).head(20)
print(top20_reviews)

#####################################################
# 3. Keywords of the Product
#####################################################
# The top 5 adjectives that occur most frequently in customer reviews of the product have been identified.

# Loading English language model using Spacy
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

# Combining all the reviews into a single string
reviews = ' '.join(df['reviewText'].values)


# Defining the function to count the occurrences of adjectives in a string
def count_adjectives(text):
    doc = nlp(text)
    adjectives = [token.lemma_.lower() for token in doc if
                  token.pos_ == 'ADJ' and token.lemma_.lower() not in STOP_WORDS]
    number_adj = Counter(adjectives)
    return number_adj


# Counting adjectives
number_adj = count_adjectives(reviews)

# Top 5 adjectives in the string
top_5_adjectives = number_adj.most_common(5)
