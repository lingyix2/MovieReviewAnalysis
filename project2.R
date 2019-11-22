#STAT 542 Project2
#Lingyi Xu
if (!require("pacman")) 
  install.packages("pacman")
pacman::p_load(
  "text2vec",
  "glmnet",
  "pROC",
  "MASS",
  "slam",
  "xgboost",
  "corrplot"
  
)
#functions
prep_fun = tolower
tok_fun = word_tokenizer
#stopwords
stop_words = c("i", "me", "my", "myself", 
               "we", "our", "ours", "ourselves", 
               "you", "your", "yours", 
               "their", "they", "his", "her", 
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were", 
               "him", "himself", "has", "have", 
               "it", "its", "of", "one", "for", 
               "the", "us", "this")

#Clean html tags
all = read.table("Project2_data.tsv",stringsAsFactors = F,header = T)
all$review = gsub('<.*?>', ' ', all$review)
splits = read.table("Project2_splits.csv", header = T)

auc_lasso<- c()
auc_lda <-c()
auc_xgboost <-c()

# model apply
for (s in 1:3)
{
#load data
train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]
ytrain = train$sentiment
ytest = test$sentiment

#built vocabulary and construct DT matrix (maximum 4-grams)
it_train = itoken(train$review,
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun)
it_test = itoken(test$review,
                 preprocessor = prep_fun, 
                 tokenizer = tok_fun)
vocab = create_vocabulary(it_train,ngram = c(1L,4L), stopwords = stop_words)
pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 5, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)
bigram_vectorizer = vocab_vectorizer(pruned_vocab)
dtm_train = create_dtm(it_train, bigram_vectorizer)
dtm_test = create_dtm(it_test, bigram_vectorizer)


##lasso
cv.out <- cv.glmnet(dtm_train, train$sentiment, alpha = 1,family="binomial") #glmnt
tmp_lasso <-predict(cv.out, s = cv.out$lambda.min, newx = dtm_test, type="response")
auc_lasso[s]=auc(ytest,tmp_lasso)



#two-sample t-statistic
v.size = dim(dtm_train)[2]
summ = matrix(0, nrow=v.size, ncol=4)
summ[,1] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==1, ]), mean)
summ[,2] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==1, ]), var)
summ[,3] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==0, ]), mean)
summ[,4] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==0, ]), var)

n1=sum(ytrain); 
n=length(ytrain)
n0 = n-n1
myp = (summ[,1] - summ[,3])/sqrt(summ[,2]/n1 + summ[,4]/n0)

#order the words by the magnitude of their t-statistics
words = colnames(dtm_train)
id = order(abs(myp), decreasing=TRUE)[1:800]
pos.list = words[id[myp[id]>0]]
neg.list = words[id[myp[id]<0]]

write(words[id], file="myvocab.txt")
myvocab = scan(file = "myvocab.txt", what = character())
pruned_vocab2 = vocab[vocab$term %in% myvocab, ]
bigram_vectorizer2 = vocab_vectorizer(pruned_vocab2)
dtm_train2 = create_dtm(it_train, bigram_vectorizer2)
dtm_test2 = create_dtm(it_test, bigram_vectorizer2)

#Naive Bayes

#DF <- as.data.frame(as.matrix(dtm_train2))
#DF$y = as.factor(ytrain)

#NBfit = NaiveBayes(y~., data=DF)
#y.class = predict(NBfit, newdata = as.matrix(dtm_test2))$class
#auc(testy,y.class)

#library(Rfast)
#DF1=DF[,colVars(DF !=0) >0.004]
#ProcessRemoveVars <- function(x1, x2){
#  x1 = x1[, which(! colnames(x1) %in% x2)]
#}

#remove.var=c('watching_bad', 'reason_gave', 'worse_movie', 'worst_movie_ever_made', 'highly_recommend_film', 'film_so_bad', 'gave_2', 'not_worth_time', 'doubt_worst', 'definitely_worth_seeing', 'don.t_waste_time_with', 'definitely_best', 'waste_time_or_money', 'no_tension', 'fails_on', 'not_even_worth', 'X1_star', 'half_time', 'worst_movies_i.ve_ever', 'avoid_like_plague', 'don.t_waste_money', 'badly_made', 'unless_want_to', 'worst_film_i.ve', 'worst_i.ve', 'unless_want', 'easily_worst', 'total_waste', 'only_good_thing_about', 'worst_movie_i.ve_ever', 'movie_worst', 'awful_acting', 'wasted_time', 'waste_time_with', 'worst_movies_i.ve', 'avoid_like', 'worst_movies_ever_seen', 'uwe_boll', 'what_waste', 'awful_film', 'avoid_at_all_costs', 'avoid_at_all', 'X3_out_10', 'uwe', 'avoid_at', 'boll')
#DF <- ProcessRemoveVars(DF, remove.var)
#sum(DF[,'bored_to'])
#Zero variances for at least one class in variables: what_waste, awful_film, avoid_at_all_costs, avoid_at_all, X3_out_10, avoid_at


##lda
#corr.matrix <- cor(as.matrix(dtm_train2))
#corrplot(corr.matrix,method = "circle",tl.pos="n", title='Correlation Matrix')
model_lda = lda(dtm_train2,ytrain)
pre_lda = predict(model_lda,dtm_test2)$posterior
tmp_lda = pre_lda[,2]
auc_lda[s]=auc(ytest,tmp_lda)

##Xgboost
xgb_model <- xgboost(booster="gblinear",data = dtm_train, label = ytrain, max.depth = 18, 
                     nthread = 10, nrounds = 10, eta =0.03, seed = 3,
                     colsample_bytree = 0.2, subsample = 0.9 )
tmp_boost <- predict(xgb_model, dtm_test)
auc_xgboost[s]=auc(ytest,tmp_boost)

}

auc_lasso
auc_lda
auc_xgboost
