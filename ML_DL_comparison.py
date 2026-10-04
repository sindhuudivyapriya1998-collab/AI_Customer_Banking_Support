import pandas as pd
import numpy as np
import re 
import joblib 
import torch 
import matplotlib.pyplot as plt 

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from transformers import (AutoTokenizer, AutoModelForSequenceClassification)
#==========================================================================================================================================
# 1. Load test data
# =========================================================================================================================================

print("\nLoading test data...")
test_data = pd.read_csv(r"\Users\DIVYA PRIYA\Downloads\FInal_project1_AI_Banking\Data\Raw\test.csv")
X_test = test_data["text"].astype(str)
y_test = test_data["category"].astype(str)

test = test_data.dropna(subset=["text","category"]).copy()
test["text"] = test["text"].astype(str)
test["text"] = test["text"].str.lower()
test["text"] = test["text"].str.replace(r"https?://\S+|www\.\S+", "",regex=True)
test["text"] = test["text"].str.replace(r"<.*?>", "",regex=True)
test["text"] = test["text"].str.replace(r"[^a-z0-9\s.,!'-]", "",regex=True)
test["text"] = test["text"].str.replace(r"\s+", " ",regex=True).str.strip()

ml_model = joblib.load("ml_baseline_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl") 
print("Model is loaded")

# ml_predictions
X_test_tfidf = tfidf.transform(X_test)
ml_predictions = ml_model.predict(X_test_tfidf)
print("ML predictions completed")

#===========================================================================================================
# ML evaluation
#===========================================================================================================
ml_accuracy = accuracy_score(y_test,ml_predictions)
ml_precision = precision_score(y_test,ml_predictions,average="weighted",zero_division=0)
ml_recall = recall_score(y_test,ml_predictions,average="weighted",zero_division=0)
ml_f1 = f1_score(y_test,ml_predictions,average="weighted",zero_division=0)

print("\n========================ML results============================")
print(f"Accuracy     :{ml_accuracy:.4f}")
print(f"Precision    :{ml_precision:.4f}")
print(f"Recall       :{ml_recall:.4f}")
print(f"weighted f1  :{ml_f1:.4f}")

#============================================================================================================
#DL evaluation
#============================================================================================================
model_path ="saved_distilbert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
dl_model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
dl_model.to(device)
dl_model.eval()

print("Distilbert model loaded successfully")
print("Device:",device)

#============================================================================================================
#DL predictions
#============================================================================================================
dl_predictions =[]
dl_confidences =[]
for text in X_test:
    inputs = tokenizer(text,
                       return_tensors='pt',
                       truncation=True,
                       padding=True,
                       max_length=128
                       )
    inputs ={
        key:value.to(device)
        for key,value in inputs.items()
        }
    with torch.no_grad():
        outputs = dl_model(**inputs)
        probabilities =torch.softmax(outputs.logits,dim=1)
        prediction = torch.argmax(outputs.logits,dim=1).item()
        confidence = probabilities[0][prediction].item()
    dl_predictions.append(prediction)
    dl_confidences.append(confidence)
print("distilbert predictions completed")

print("number of test samples:",len(X_test))
print("number of dl predictions:",len(dl_predictions))
print("\n first 10 DL confidence scores:")
for i in range(min(10,len(dl_confidences))):
    print("Prediction:",dl_predictions[i],"| confidence:",round(dl_confidences[i],4))

#==================================================================================
# Decode distilbert 
#==================================================================================
label_encoder = joblib.load("models/label_encoder.pkl")
dl_predictions = label_encoder.inverse_transform(np.array(dl_predictions))
print("Distilbert predictions decoded")
print("number of decoded predictions:",len(dl_predictions))

#==================================================================================================
#DL evaluation
#==================================================================================================
dl_accuracy = accuracy_score(y_test,dl_predictions)
dl_precision = precision_score(y_test,dl_predictions,average="weighted",zero_division=0)
dl_recall = recall_score(y_test,dl_predictions,average="weighted",zero_division=0)
dl_f1 = f1_score(y_test,dl_predictions,average="weighted",zero_division=0)

print("\n========================DL results=========================")
print(f"Accuracy     :{dl_accuracy:.4f}")
print(f"Precision    :{dl_precision:.4f}")
print(f"Recall       :{dl_recall:.4f}")
print(f"weighted f1  :{dl_f1:.4f}")

#================================================================================================================
#ML and DL comparison
#=================================================================================================================
comparison = pd.DataFrame({"Metric" :["Accuracy",
                                      "Precision",
                                      "Recall",
                                      "Weighted f1"],
                            "ML"    :[ml_accuracy,
                                      ml_precision,
                                      ml_recall,
                                      ml_f1],
                            "DL"    :[dl_accuracy,
                                      dl_precision,
                                      dl_recall,
                                      dl_f1]
})

print("\n===========================ML vs DL comparison===========================")
print(comparison.to_string(index=False))

#==============================================================================================================================
# Performance improvement
#==============================================================================================================================
print("\n" + "=" * 70)
print("1.Performance improvement")
print("\n" + "=" * 70)

comparison = pd.DataFrame({"Metric" :["Accuracy",
                                      "Precision",
                                      "Recall",
                                      "Weighted f1"],
                            "Traditional ML"  :[ml_accuracy,
                                                ml_precision,
                                                ml_recall,
                                                ml_f1],
                            "Distilbert"  :[dl_accuracy,
                                            dl_precision,
                                            dl_recall,
                                            dl_f1]
})
print(comparison.to_string(index=False))

print("\nPerformance difference (?Distilbert - ML)")
print("-" * 50)

accuracy_diff = dl_accuracy - ml_accuracy
precision_diff = dl_precision - ml_precision
recall_diff = dl_recall - ml_recall 
f1_diff = dl_f1 - ml_f1 

print(f"Accuracy difference   : {accuracy_diff:+.4f}")
print(f"Precision difference  :{precision_diff:+.4f}")
print(f"Recall difference  :{recall_diff:+.4f}")
print(f"F1 difference  :{f1_diff:+.4f}")

#===================================================================================================================================
# classification errors
#==================================================================================================================================
from sklearn.metrics import confusion_matrix
print("\n" + "=" * 70)
print(" Classification errors")

actual=np.array(y_test)
ml_pred = np.array(ml_predictions)
dl_pred = np.array(dl_predictions)

ml_error_count = np.sum(actual != ml_pred)
dl_error_count = np.sum(actual != dl_pred)

print("ML classification errors :", ml_error_count)
print("Distilbert classification errors:", dl_error_count)

labels = sorted(np.unique(y_test))

ml_cm = confusion_matrix(y_test,ml_predictions,labels=labels)
plt.figure(figsize=(14,12))
plt.imshow(ml_cm)
plt.title("ML confusion matrix")
plt.xlabel("Predicted label")
plt.ylabel("actual label")

plt.xticks(range(len(labels)),labels,rotation=90)
plt.yticks(range(len(labels)),labels)

for i in range(len(labels)):
    for  j in range(len(labels)):
        if ml_cm[i,j] > 0:
            plt.text(j,i,ml_cm[i,j],ha="center",va="center")
plt.tight_layout()
plt.savefig("ML_confusion_matrix.png")
plt.show()

dl_cm = confusion_matrix(y_test,dl_predictions,labels=labels)
plt.figure(figsize=(14,12))
plt.imshow(dl_cm)

plt.title("Distilbert - confusion matrix")
plt.xlabel("Predicted label")
plt.ylabel("actual label")

plt.xticks(range(len(labels)),labels,rotation=90)
plt.yticks(range(len(labels)),labels)

for i in range(len(labels)):
    for  j in range(len(labels)):
        if dl_cm[i,j] > 0:
            plt.text(j,i,dl_cm[i,j],ha="center",va="center")
plt.tight_layout()
plt.savefig("Dl_confusion_matrix.png")
plt.show()

#=======================================================================================================================
#Classification error comparison
#=======================================================================================================================
ml_errors = sum(y_test != ml_predictions)
dl_errors = sum(y_test != dl_predictions)

print("\n" + "=" * 50)
print("classification error comapriosn")

print(f"ML classification errors   :{ml_errors}")
print(f"Distilbert classification errors :{dl_errors}")

print(f"ML error rate       :{ml_errors / len(y_test):.4f}")
print(f"DL error rate       :{dl_errors / len(y_test):.4f}")

#===========================================================================================================================
# Confidensce and escalation
#=============================================================================================================================
high_threshold = 0.80
medium_threshold = 0.50 

def confidence_decision(confidence):
    if confidence >= high_threshold:
        return "High confidence","Automated response"
    elif confidence >= medium_threshold:
        return "Medium confidence","Additional verification"
    else:
        return "Low confidence","Human agent escalation"

confidence_levels =[]
decisions = []
for confidence in dl_confidences:
    level,decision = confidence_decision(confidence)
    confidence_levels.append(level)
    decisions.append(decision)

#================================================================================================
high_count = confidence_levels.count("High confidence")
medium_count = confidence_levels.count("Medium confidence")
low_count = confidence_levels.count("Low confidence")

print("\n" + "=" * 60)
print("Confidence and Escalation")
print("=" * 60)

print("High confidence  :",high_count)
print("Medium confidence :",medium_count)
print("Low confidence  :",low_count)

#calculate percentages
#=================================================================================================
total_predictions = len(confidence_levels)
print("\nconfidence distribution:")
print("High confidence   :",f"{high_count / total_predictions * 100:.2f}%")
print("Medium confidence   :",f"{medium_count / total_predictions * 100:.2f}%")
print("Low confidence   :",f"{low_count / total_predictions * 100:.2f}%")

#====================================================================================================
#show individual decision
#=====================================================================================================
print("\nSample confidence decision:")
for i in range(min(10,len(X_test))):
    level, decision = confidence_decision(dl_confidences[i])

    print("\nCustomer Query:",X_test.iloc[i] if hasattr(X_test,"iloc")else X_test[i])
    print("Predicted class :",dl_predictions[i])
    print("Confidence  :",round(dl_confidences[i],4))
    print("Confidence level :",level)
    print("Decision  :",decision)
