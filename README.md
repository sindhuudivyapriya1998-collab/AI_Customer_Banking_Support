# Project goal
An end-to-end AI banking support system that takes a customer query and automatically classifies it, measures confidence, determine priority/escalation, generates an AI response, summarizes the conversation, and provides business insights.

# Workflow
Customer Query -> NLP -> ML -> DistilBERT -> Confidence -> Priority -> Escalation -> Generative AI -> Support Action -> Business Metrics -> Streamlit Dashboard

# Technologies:
- Python
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- TensorFlow/Pytorch
- Hugging Face
- Transformers
- DistilBERT
- Groq API
- Streamlit
- SQL

# Main Phases:
# 1. Data preparation & NLP 
   - validation, missing/duplicate checks, category imbalance, text analysis, and preprocessing.
# 2. Machine Learning Baseline
    - Linear SVM + TF-IDF
    - Random Forest + TF-IDF
    - XGBoost + TF-IDF
  compared these models using classification metrics such as Accuracy, Precision, Recall, Weighted F1-score, and Confusion Matrix, and then used the comparison to identify the appropriate baseline model.
# 3. Deep Learning 
     DistilBERT is fine-tuned for banking intent classification with confidence and error analysis.
# 4. Confidence & Escalation
     - >=80% -> High
     - 60%-79% -> Medium 
     - <60% -> Low 
     Low confidence or high priority can trigger escalation.
# 5. Generative AI
     Produces customer responses, conversation summaries, and recommended support actions using the Groq API 
# 6. Business Metrics
     response time, confidence, escalation status and resolution mode.
# 7. Business Intelligence
     Combines query, category, confidence, priority, escalation, AI response, summary, and recommended action into the dashboard.

# Streamlit Apllication
The final streamlit application provides:
- Customer query input
- Predicted category
- Confidence & confidence level
- Priority
- Escalation recommendation
- AI-generated customer response
- Conversation summary
- Recommended support action
- Business metrics
- Business intelligence

We observed that AI can automate a significant part of the banking customer-support workflow while using confidence and priority to identify cases that require human intervention.

     
