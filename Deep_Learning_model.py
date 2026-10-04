#Libraries
#---------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np

import torch 
from torch.utils.data import Dataset
from transformers import(AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer)
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import(accuracy_score,precision_recall_fscore_support,classification_report,confusion_matrix)
print("Libraries imported successfully")

# 1.Load Data
#===========================================================================================================================
train_data = pd.read_csv(r"\Users\DIVYA PRIYA\Downloads\FInal_project1_AI_Banking\Data\Raw\train.csv")
print("Train shape:",train_data.shape)
test_data = pd.read_csv(r"\Users\DIVYA PRIYA\Downloads\FInal_project1_AI_Banking\Data\Raw\test.csv")
print("Test shape:",test_data.shape)
print("\nTrain columns:")
print(train_data.columns.tolist())
print("\nTest columns:")
print(test_data.columns.tolist())

# 2.Select text and target
#===================================================================================================================================
text_column = "text"
target_column = "category"

X = train_data[text_column].astype(str)
y = train_data[target_column].astype(str)

X_test = test_data[text_column].astype(str)
y_test = test_data[target_column].astype(str)
print("Training data prepared")
print("Testing data prepared")

#=================================================================================================================================
# 3.Label encoding
#=================================================================================================================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_test_encoded = label_encoder.transform(y_test)
num_labels = len(label_encoder.classes_)

print("Number of categories:",num_labels)
print("Label encoding completed")

import joblib
import os
os.makedirs("models",exist_ok=True)
joblib.dump(label_encoder,"models/label_encoder.pkl")
print("label encoder saved")

#===================================================================================================================================
# 4.split train and validation
#====================================================================================================================================
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X,y_encoded,test_size=0.20,random_state=42,stratify=y_encoded)
print("Training samples:",len(X_train))
print("Validation samples:",len(X_val))
print("Final test samples:",len(X_test))

#====================================================================================================================================
# 5.Load DistilBERT tokenizer
#====================================================================================================================================
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded successfully")

# tokenize training data
train_tokens = tokenizer(X_train.tolist(),padding=True,truncation=True,max_length=128)
print("Training data tokenized")

# tokenize validation data
val_tokens = tokenizer(X_val.tolist(),padding=True,truncation=True,max_length=128)
print("validation data tokenized")

# tokenize test data
test_tokens = tokenizer(X_test.tolist(),padding=True,truncation=True,max_length=128)
print("Test data tokenized")

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=num_labels)
print("DistilBERT model loaded successfully")

#=======================================================================================================================================
# 6. create datasets
#=======================================================================================================================================
from datasets import Dataset 

train_dataset = Dataset.from_dict({"input_ids":train_tokens["input_ids"],"attention_mask":
                                   train_tokens["attention_mask"],"labels":y_train.tolist()})
val_dataset = Dataset.from_dict({"input_ids":val_tokens["input_ids"],"attention_mask":
                                   val_tokens["attention_mask"],"labels":y_val.tolist()})
test_dataset = Dataset.from_dict({"input_ids":test_tokens["input_ids"],"attention_mask":
                                   test_tokens["attention_mask"],"labels":y_test_encoded.tolist()})

print("Training dataset:",len(train_dataset))
print("Validation dataset:",len(val_dataset))
print("Test dataset:",len(test_dataset))

#===================================================================================================================================
# 7.Evaluation metrics
#===================================================================================================================================
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score)
def compute_metrics(eval_pred):
    logits,labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels,predictions)
    precision = precision_score(labels,predictions,average="weighted",zero_division=0)
    recall = recall_score(labels,predictions,average="weighted",zero_division=0)
    f1 = f1_score(labels,predictions,average="weighted",zero_division=0)

    return{ "accuracy": accuracy,
           "precision": precision,
           "recall": recall,
           "weighted_f1":f1 
           }
print("Evaluation metrics defined successfully")
#========================================================================================================================================
# 8.Training arguments
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="./models/distilbert_training",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
per_device_train_batch_size=8,
per_device_eval_batch_size=8,
    num_train_epochs = 3,
    weight_decay = 0.01,
    logging_steps=50,
    load_best_model_at_end=True,
metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none"
)
print("Training arguments created successfully")

#==================================================================================================================================
# 9. create trainer
#==================================================================================================================================
from transformers import Trainer

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset = train_dataset,
                  eval_dataset = val_dataset,
compute_metrics = compute_metrics )

print("Trainer created successfully")
#====================================================================================================================================
# Train/fine tune distilbert model
#====================================================================================================================================

print("Starting DistilBERT training")
train_result = trainer.train()
print("DistilBERT training completed")

#=====================================================================================================================================
# Save the model
#=====================================================================================================================================
model_path = "./saved_distilbert_model"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
print("Model saved successfully!!!")
print("Tokenizer saved successfully")

#======================================================================================================================================
# 10.Validation
#======================================================================================================================================
validation_results = trainer.evaluate(eval_dataset=val_dataset)
print("\nValidation results:")

for key,value in validation_results.items():
    print(key, ":",value)

#======================================================================================================================================
# 11.Testing
##======================================================================================================================================
test_results = trainer.predict(test_dataset)
test_logits = test_results.predictions 
test_true_labels = (test_results.label_ids)
test_predictions = np.argmax(test_logits,axis=1)

print("Testing completed")
#=========================================================================================================================================
# 12.Performance evaluation
#=========================================================================================================================================
test_accuracy = accuracy_score(test_true_labels,test_predictions)
test_precision = precision_score(test_true_labels,test_predictions,average="weighted",zero_division=0)
test_recall = recall_score(test_true_labels,test_predictions,average="weighted",zero_division=0)
test_f1 = f1_score(test_true_labels,test_predictions,average="weighted",zero_division=0)

print("\n=======================Test Performance========================")
print("Accuracy    :",round(test_accuracy,4))
print("Precision   :",round(test_precision,4))
print("Recall      :",round(test_recall,4))
print("Weighted f1 :",round(test_f1,4))
#======================================================================================================================================
# 13.classification report
#======================================================================================================================================
print(classification_report(
    test_true_labels,
    test_predictions,
target_names = label_encoder.classes_,
    zero_division=0 
)
)
                        
#======================================================================================================================================
# 14. Confusion matrix
#======================================================================================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_true_labels,test_predictions)
plt.figure(figsize=(18,15))
sns.heatmap(cm,cmap="Blues",
xticklabels=label_encoder.classes_,
yticklabels=label_encoder.classes_)
plt.xlabel("Predicted category")
plt.ylabel("Actual category")
plt.title("DistilBert confusion matrix")

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

#=================================================================================================================================
# 15.Error Analysis
#=================================================================================================================================
error_df = pd.DataFrame({"text":X_test.values,
                         "true_label":test_true_labels,
                         "predicted_label":test_predictions
})
error_df["true_category"] = label_encoder.inverse_transform(error_df["true_label"])
error_df["predicted_category"] = label_encoder.inverse_transform(error_df["predicted_label"])
errors = error_df[error_df["true_label"] != error_df["predicted_label"]].copy()

print("Total incorrect predictions:",len(errors))
print(errors[["text","true_category","predicted_category"]].head(20))
#==================================================================================================================================
# 16.prediction confidence
#==================================================================================================================================
import torch
import numpy as np 
probabilities = torch.softmax(torch.tensor(test_logits),dim=1).numpy()
confidence = probabilities.max(axis=1)
print("Prediction confidence calculated")
print("\n Sample confidence scores:")
print(confidence[:10])
#==================================================================================================================================
# 17. Add confidence/prediction table 
#==================================================================================================================================
error_df["confidence"] = confidence 
print(error_df[["text","true_category","predicted_category","confidence"]].head(10))

# save prediction results for gen ai
error_df[["text","true_category","predicted_category","confidence"]].to_csv("DL_predictions.csv",index=False)
print("\nDL prediction saved successfully.")

# 18.Top 3 probable categories
#==================================================================================================================================
top_3_indices = np.argsort(probabilities,axis=1)[:,-3:][:,::-1]

top_3_results = []
for i in range(len(X_test)):
    indices = top_3_indices[i]
    categories = label_encoder.inverse_transform(indices)
    scores = probabilities[i,indices]
    top_3_results.append({
        "text":X_test.iloc[i],
        "top_1_category":categories[0],
        "top_1_confidence":scores[0],
        "top_2_category":categories[1],
        "top_2_confidence":scores[1],
        "top_3_category":categories[2],
        "top_3_confidence":scores[2],
     })
top_3_df = pd.DataFrame(top_3_results)

print("\nTop 3 probable categories:")
print(top_3_df.head(10))
