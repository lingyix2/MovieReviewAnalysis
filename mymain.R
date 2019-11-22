if (!require("pacman")) 
  install.packages("pacman")
pacman::p_load(
  "text2vec",
  "glmnet",
  "pROC",
  "MASS",
  "xgboost",
  "slam"
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
auc_lasso=auc(ytest,tmp_lasso)
  
  
  
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

  
##lda
model_lda = lda(dtm_train2,ytrain)
pre_lda = predict(model_lda,dtm_test2)$posterior
tmp_lda = pre_lda[,2]
auc_lda=auc(ytest,tmp_lda)
  
##Xgboost
xgb_model <- xgboost(booster="gblinear",data = dtm_train, label = ytrain, max.depth = 18, 
                      nthread = 10, nrounds = 10, eta =0.03, seed = 3,
                      colsample_bytree = 0.2, subsample = 0.9 )
tmp_boost <- predict(xgb_model, dtm_test)
auc_xgboost=auc(ytest,tmp_boost)

tmp_lasso=as.vector(tmp_lasso)
prob_lasso <- cbind(new_id=test$new_id, prob=tmp_lasso)
prob_lda <- cbind(new_id=test$new_id, prob=tmp_lda)
prob_xgboost <- cbind(new_id=test$new_id, prob=tmp_boost)


write.csv(prob_lasso, file = "mysubmission1.txt", row.names = FALSE, col.names = TRUE)
write.csv(prob_lda, file = "mysubmission2.txt", row.names = FALSE, col.names = TRUE)
write.csv(prob_xgboost, file = "mysubmission3.txt", row.names = FALSE, col.names = TRUE)
