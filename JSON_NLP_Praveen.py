
#<-----------Documentation---------->
#Text preprocessing
#top 20 ups in the comments
#Text Categorization
#Extracting Key phrases
#sentiment analysis --> positive and negative
#extracting top 50 words out of whole data.
#bargraph and word cloud visual representation of top 50 words.
#Creating a Classification ML model that can perform sentiment analysis on new data along with accuracy.
#Topic modeling- Categorise data on the basis of categories.


#export PATH=/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin:$PATH

#if you get hive related run time exception" ==>  rm -rf metastore_db/




#===============================================================================
#Import all the required libraries
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, array_contains, when
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import StanfordNERTagger
import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
import re
import string

#Create spark session--> starting point of Spark program execution
spark = SparkSession.builder.master('yarn-client').appName('Json_processing').getOrCreate()
data = spark.read.json('file:///Users/nageshsinghchauhan/Documents/projects/praveenProject/sparkNLP/reddit.json')
data.createOrReplaceTempView("data_table")
data.printSchema()
#convert to Parquet
#data = data_ini.write.parquet('file:///Users/nageshsinghchauhan/Documents/projects/praveenProject/sparkNLP/reddit.parquet')
#data.createOrReplaceTempView("data_table")


#Select author from the table
userDF = spark.sql("SELECT author from data_table where author != '[deleted]' and body is not null")
userDF.createOrReplaceTempView("user_table")

#remove deleted columns
df1 = spark.sql("SELECT parent_id, author,created_utc, body,score, ups from data_table where author != '[deleted]' and body is not null")
df1.createOrReplaceTempView("filtered_table")

#top 20 ups in the comments
df2 = spark.sql("SELECT parent_id, author,created_utc, body, ups from filtered_table where ups >= 3 ORDER BY ups ASC limit 250")
df2.createOrReplaceTempView("most_ups")

#extract Body part
#body_rdd = df1.select("body").rdd.flatMap(lambda x: x)

#remove header
#header = body_rdd.first()
#data_rmvCol = body_rdd.filter(lambda row: row != header)

#convert to lowercase
lowerCase_sent = body_rdd.map(lambda x : x.lower())

#sentence tokenizer
def sent_tokenize1(x):
	import nltk
	return nltk.sent_tokenize(x)

sentenceTokenizeRDD = lowerCase_sent.map(sent_tokenize1)

#word tokenizer
def word_tokenize1(x):
    splitted = [word for line in x for word in line.split()]
    return splitted

words1 = sentenceTokenizeRDD.map(word_tokenize1)

#Remove stopWords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def removeStopWords(x):
    from nltk.corpus import stopwords
    filteredSentence = [w for w in x if not w in stop_words]
    return filteredSentence

stopWRDD = words1.map(removeStopWords)

#Remove punctuations and empty spaces
def removePunctuations(x):
    list_punct=list(string.punctuation)
    filtered = [''.join(c for c in s if c not in list_punct) for s in x] #remove punctuations
    filtered_space = [s for s in filtered if s] #remove empty space
    return filtered

rmvPunctRDD = stopWRDD.map(removePunctuations)

"""
#Remove tags
def removeTags(x):
	text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",x)
	return text

rmvTagsRDD = rmvPunctRDD.map(removeTags)
"""

#Lemmatization
def lemmatizationFunct(x):
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    finalLem = [lemmatizer.lemmatize(s,pos='v') for s in x]
    return finalLem

lem_wordsRDD = rmvPunctRDD.flatMap(lemmatizationFunct)

#Named Entity Recognition using Stanford NER library
"""
Download Stanford NER library
Go to https://nlp.stanford.edu/software/CRF-NER.html#Download
and download the latest version, I am using Stanford Named Entity Recognizer version 3.9.2.
I get a zip file which is called “stanford-ner-2018–10–16.zip” which needs to be unzipped and I renamed it to stanford_ner and placed it in the home folder.
"""
def namedEntityRecog(x):
	stanford_ner_tagger = StanfordNERTagger(
	'Documents/projects/praveenProject/sparkNLP/stanford_ner/classifiers/english.muc.7class.distsim.crf.ser.gz',
	'Documents/projects/praveenProject/sparkNLP/stanford_ner/stanford-ner-3.9.2.jar')
	results = stanford_ner_tagger.tag(x)
	#print('Original Sentence: %s' % (article))
	list1 = []
	for result in results:
		tag_value = result[0]
		tag_type = result[1]
		if tag_type != 'O':
			list1.append('Type: %s, Value: %s' % (tag_type, tag_value))

	filtered_list = [s for s in list1 if s!=""]
	return filtered_list

namedEntityRDD = lem_wordsRDD.map(namedEntityRecog)
namedEntityRDD.take(50)

#join tokens
def joinTokensFunct(x):
    joinedTokens_list = []
    x = " ".join(x)
    return x

joinedTokens = lem_wordsRDD.map(joinTokensFunct)

#feature exraction using CountVectorizer and extracting key phrases
#here we are extracting key phrases with 2 and 3 terms
from sklearn.feature_extraction.text import CountVectorizer
import re
def featureExtDef(param):
	#param=[param] #since param originally a string
	SW = set(stopwords.words('english'))
	#CV is same as TF from TF-IDF
	cv=CountVectorizer(stop_words=SW, max_features=50000, ngram_range=(2,2))
	X=cv.fit(param)
	bag_of_words = X.transform(param)
	sum_words = bag_of_words.sum(axis=0)
	words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
	words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
	return words_freq

key_phrase_RDD = joinedTokens.map(featureExtDef)
key_phrase_RDD.take(20)

#sentiment analysis
#Through my previous attempt at sentiment analysis with Pandas and Scikit-Learn,
# I learned that TF-IDF with Logistic Regression is quite a strong combination,
#and showed robust performance, as high as Word2Vec + Convolutional Neural Network model.

"""
#extracting the keywords
def extractphraseFunct(x):
    from nltk.corpus import stopwords
    stop_words=set(stopwords.words('english'))

    def leaves(tree):
        #Finds NP (nounphrase) leaf nodes of a chunk tree
        for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
            yield subtree.leaves()

    def acceptable_word(word):
        """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""
        accepted = bool(3 < len(word) <= 20
            and 'https' not in word.lower()
            and 'http' not in word.lower()
            and '#' not in word.lower()
            )
        yield accepted

    def get_terms(tree):
        for leaf in leaves(tree):
            term = [w for w,t in leaf if not w in stop_words if acceptable_word(w)]
            yield term

    sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
    grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """
    chunker = nltk.RegexpParser(grammar)
    tokens = nltk.regexp_tokenize(x,sentence_re)
    postoks = nltk.tag.pos_tag(tokens) #Part of speech tagging
    tree = chunker.parse(postoks) #chunking
    terms = get_terms(tree)
    temp_phrases = []
    for term in terms:
        if len(term):
            temp_phrases.append(' '.join(term))

    finalPhrase = [w for w in temp_phrases if w] #remove empty lists
    return finalPhrase

extractphraseRDD = joinedTokens.map(extractphraseFunct)

# get feature names
feature_names=cv.get_feature_names()
 """

#sentiment of each key phrase
def sentimentWords(x):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    senti_list_temp = []
    for i in x:
        i = str(i).encode('utf-8')
        y = ''.join(i)
        vs = analyzer.polarity_scores(y)
        senti_list_temp.append((y, vs))
        senti_list_temp = [w for w in senti_list_temp if w]
    sentiment_list  = []
    for j in senti_list_temp:
        first = j[0]
        second = j[1]
        for (k,v) in second.items():
            if k == 'compound':
                if v < 0.0:
                    sentiment_list.append((first, "Negative"))
                elif v == 0.0:
                    sentiment_list.append((first, "Neutral"))
                else:
                    sentiment_list.append((first, "Positive"))
    return sentiment_list

sentimentRDD = key_phrase_RDD.flatMap(sentimentWords)

#extract positive sentiments :
def positiveTokensFunct(x):
    positive_list = []
    for i in x:
        first = i[0]
        second = i[1]
        if second == "Positive":
            positive_list.append(first)
    return positive_list

pos_sentimentRDD = sentimentRDD.map(positiveTokensFunct).filter(lambda row:row != [])

#extract Negative sentiments
def negativeTokensFunct(x):
    negative_list = []
    for i in x:
        first = i[0]
        second = i[1]
        if second == "Negative":
            negative_list.append(first)
    return negative_list

neg_sentimentRDD = sentimentRDD.map(negativeTokensFunct).filter(lambda row:row != [])

#=========================
# Extract top 50 most used words
freqDistRDD = extractphraseRDD.flatMap(lambda x : nltk.FreqDist(x).most_common()).map(lambda x: x).reduceByKey(lambda x,y : x+y).sortBy(lambda x: x[1], ascending = False)
#Plot bargraph top 50 most used words
df_fDist = freqDistRDD.toDF() #converting RDD to sprk DF
df_fDist.createOrReplaceTempView("myTable")
df2 = spark.sql("SELECT _1 AS Keywords, _2 as Frequency from myTable limit 20")#renaming columns
pandD = df2.toPandas() #converting spark dataframes to pandas daraframes
pandD.plot.barh(x='Keywords', y='Frequency', rot=1, figsize=(10,8))

#==========================
#Plot wordCloud for positive phrases
from wordcloud import WordCloud
xyRDD = pandD.set_index('Keywords').T.to_dict('records')
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=100, relative_scaling=0.5, colormap='Dark2').generate_from_frequencies(dict(*xyRDD))
plt.figure(figsize=(14, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#================================================================================
#creating a logistic regression model that can perform sentiment Analysis
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
#from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import NaiveBayes

from pyspark.ml.evaluation import BinaryClassificationEvaluator


#create column sentiment based on score column
body_df = df1.select("body", "score").withColumn("sentiment", when(col("score") <= 0, 0).otherwise(1))
corpus_df = body_df.select("body", "sentiment")
text_df = corpus_df.select("body")
#split the final corpus
(train_set, val_set, test_set) = body_df.randomSplit([0.98, 0.01, 0.01], seed = 1000)

tokenizer = Tokenizer(inputCol="body", outputCol="words")
hashtf = HashingTF(numFeatures=10000, inputCol="words", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "score", outputCol = "label")

#creating SparkML pipeline
pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])
pipelineFit = pipeline.fit(train_set)
train_df = pipelineFit.transform(train_set)
val_df = pipelineFit.transform(val_set)
train_df.show(5)

#Fitting a Logistic Regression model
#lr = LogisticRegression()
lr = NaiveBayes(smoothing=1.0, modelType="binomial")
lrModel = lr.fit(train_df)
predictions = lrModel.transform(val_df)

#Evaluating model accuracy
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)


================================================================================
#Topic Modeling
from pyspark.ml.clustering import LDA

tokenizer1 = Tokenizer(inputCol="body", outputCol="words")
hashtf1 = HashingTF(numFeatures=10000, inputCol="words", outputCol='tf')
idf1 = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
#label_stringIdx1 = StringIndexer(inputCol = "score", outputCol = "label")

pipeline = Pipeline(stages=[tokenizer1, hashtf1, idf1])
pipelineFit = pipeline.fit(text_df)
train_dataframe = pipelineFit.transform(text_df)

# Trains a LDA model.
lda = LDA(k=10, maxIter=10)
model = lda.fit(train_dataframe)

ll = model.logLikelihood(train_dataframe)
lp = model.logPerplexity(train_dataframe)
print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
print("The upper bound on perplexity: " + str(lp))

# Describe topics.
topics = model.describeTopics(3)
print("The topics described by their top-weighted terms:")
topics.show(truncate=False)

# Shows the result
transformed = model.transform(train_dataframe)
transformed.show(truncate=False)
