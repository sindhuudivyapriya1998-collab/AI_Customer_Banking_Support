import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.svm import LinearSVC 
from sklearn.metrics import accuracy_score,classification_report 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score

#-----------------------------------------------------------------------------------------
# 1. Load Data
#-----------------------------------------------------------------------------------------
train = pd.read_csv(r"\Users\DIVYA PRIYA\Downloads\FInal_project1_AI_Banking\Data\Raw\train.csv")
print("Train shape:",train.shape)
test = pd.read_csv(r"\Users\DIVYA PRIYA\Downloads\FInal_project1_AI_Banking\Data\Raw\test.csv")
print("Test shape:",test.shape)
#-----------------------------------------------------------------------------------------
# 2.Basic preprocessing
#-----------------------------------------------------------------------------------------
train = train.dropna(subset=["text","category"]).copy()
test = test.dropna(subset=["text","category"]).copy()

train["text"] = train["text"].astype(str)
test["text"] = test["text"].astype(str)

train["text"] = train["text"].str.lower()
test["text"] = test["text"].str.lower()

train["text"] = train["text"].str.replace(r"https?://\S+|www\.\S+", "",regex=True)
test["text"] = test["text"].str.replace(r"https?://\S+|www\.\S+", "",regex=True)

train["text"] = train["text"].str.replace(r"<.*?>", "",regex=True)
test["text"] = test["text"].str.replace(r"<.*?>", "",regex=True)

train["text"] = train["text"].str.replace(r"[^a-z0-9\s.,!'-]", "",regex=True)
test["text"] = test["text"].str.replace(r"[^a-z0-9\s.,!'-]", "",regex=True)

train["text"] = train["text"].str.replace(r"\s+", " ",regex=True).str.strip()
test["text"] = test["text"].str.replace(r"\s+", " ",regex=True).str.strip()
#----------------------------------------------------------------------------------------------------
# 3.Train/validation split
#----------------------------------------------------------------------------------------------------
X = train["text"]
y = train["category"]

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y) 
print("\nTraining samples:",len(X_train))
print("Validation samples:",len(X_val))
#------------------------------------------------------------------------------------------------------
# 4.TF-IDF feature extraction
#------------------------------------------------------------------------------------------------------

tfidf = TfidfVectorizer(max_features=5000,ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf =tfidf.transform(X_val)
print("\nTF-IDF training shape:",X_train_tfidf.shape)
print("\nTF-IDF Validation shape:",X_val_tfidf.shape)
#------------------------------------------------------------------------------------------------------
# 5.Linear SVM model
#------------------------------------------------------------------------------------------------------
svm_model = LinearSVC(random_state=42)
svm_model.fit(X_train_tfidf,y_train)
#-------------------------------------------------------------------------------------------------------
# 6. Validation prediction
#-------------------------------------------------------------------------------------------------------
svm_val_pred = svm_model.predict(X_val_tfidf)

#-------------------------------------------------------------------------------------------------------
# 7.Evaluation
#--------------------------------------------------------------------------------------------------------
accuracy = accuracy_score(y_val,svm_val_pred) 
print("\n ============= Linera SVM Model RESULTS ==============") 
print("Validation Accuracy:",accuracy)
print("\n Classification Report:")
print(classification_report(y_val,svm_val_pred))

#=================================================================================================
# 6.Random forest model
#=================================================================================================
rf_model = RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)
rf_model.fit(X_train_tfidf,y_train)

rf_val_pred = rf_model.predict(X_val_tfidf)

rf_accuracy = accuracy_score(y_val,rf_val_pred) 
print("\n ============= Random Forest RESULTS ==============") 
print("Validation Accuracy:",rf_accuracy)
print("\n Classification Report:")
print(classification_report(y_val,rf_val_pred))

#==================================================================================================
#7. XG Boost
#===================================================================================================

xgb_model = XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,random_state=42,n_jobs=-1,eval_metric="mlogloss")
xgb_label_encoder = LabelEncoder()
y_train_xgb = xgb_label_encoder.fit_transform(y_train)
y_val_xgb = xgb_label_encoder.transform(y_val)
xgb_model.fit(X_train_tfidf,y_train_xgb)

xgb_val_pred = xgb_model.predict(X_val_tfidf)

xgb_accuracy = accuracy_score(y_val_xgb,xgb_val_pred) 
xgb_f1 = f1_score(y_val_xgb,xgb_val_pred,average="weighted",zero_division=0)
print("\n ============= XGBOOST RESULTS ==============") 
print("Validation Accuracy:",xgb_accuracy)
print("\n Classification Report:")
print(classification_report(y_val_xgb,xgb_val_pred))

#compare models
#===========================================================================================================
print("\n" + "=" * 60)
print("ML model comparison")
print("=" * 60)
print(f"Linear SVM Accuracy : {accuracy:.4f}")
print(f"Random forest : {rf_accuracy:.4f}")
print(f"XGboost Accuracy : {xgb_accuracy:.4f}")

svm_f1 = f1_score(y_val,svm_val_pred,average="weighted")
rf_f1 = f1_score(y_val,rf_val_pred,average="weighted")
xgb_f1 = f1_score(y_val_xgb,xgb_val_pred,average="weighted")

print("\n weighted f1 comparison")
print(f"Linera SVM     :{svm_f1:.4f}")
print(f"Random forest     :{rf_f1:.4f}")
print(f"Xgboost     :{xgb_f1:.4f}")

#=============================================================================================================
#choose best model

ml_results = {"Linera svm": svm_f1,
              "Random forest":rf_f1,
              "xgb"    :xgb_f1
}
best_ml_model = max(ml_results,key=ml_results.get)
print("\n" + "=" * 60)
print("Best ML model")
print("=" * 60)

print("Best Model:", best_ml_model)
print("weighted f1:",round(ml_results[best_ml_model],4))


import joblib

#save model
joblib.dump(svm_model,"ml_baseline_model.pkl")
#save TF-IDF vectorizer
joblib.dump(tfidf,"tfidf_vectorizer.pkl") 

print("Model and TF-IDF vectorizer saved successfully.") 
#--------------------------------------------------------------------------------------------------------
# 8.confusion matrix
#--------------------------------------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_val,svm_val_pred,labels=svm_model.classes_)
plt.figure(figsize=(18,15))

sns.heatmap(cm,cmap="Blues",xticklabels=svm_model.classes_,yticklabels=svm_model.classes_)
plt.title("confusion_matrix - Linear SVM")
plt.xlabel("Predicted category")
plt.ylabel("Acutual category")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("Confusion_matrix.png",dpi=300,bbox_inches="tight")
print("\n Confusion matrix saved successfully")
#----------------------------------------------------------------------------------------------------
# 9.Performance analysis
#-----------------------------------------------------------------------------------------------------
from sklearn.metrics import precision_score,recall_score,f1_score 

accuracy = accuracy_score(y_val,svm_val_pred)
precision = precision_score(y_val,svm_val_pred,average='weighted',zero_division=0)
recall = recall_score(y_val,svm_val_pred,average='weighted',zero_division=0)
f1 = f1_score(y_val,svm_val_pred,average='weighted',zero_division=0)

print("\n =============Performance Analysis================")
print(f"Accuracy  :{accuracy:.4f}")
print(f"Precision  :{precision:.4f}")
print(f"Recall  :{recall:.4f}")
print(f"F1-score  :{f1:.4f}")

plt.show()