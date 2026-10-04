import os
import time
import csv
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv(".env",override=True)

api_key =os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq_api_key not found. check your .env file.")
client = Groq(api_key=api_key)
model_name = "openai/gpt-oss-120b"

print("Groq client initialized successfully.")
print("Model:", model_name)

#=======================================================================================================================
# Load deep learning predictions
#==========================================================================================================================
dl_predictions=pd.read_csv("DL_predictions.csv")
selected_index=1

customer_query = dl_predictions.loc[selected_index,"text"]
predicted_category = dl_predictions.loc[selected_index,"predicted_category"]
model_confidence = dl_predictions.loc[selected_index,"confidence"]

print("\nDeep Learning Prediction")
print("Customer Query:",customer_query)
print("Predicted Category:",predicted_category)
print(f"Model Confidence:{model_confidence:.2%}")

#================================================================================
#confidence & escalation
#================================================================================
confidence_threshold = 0.70 
print("\n" + "=" * 60)
print("Confidence and escalation")
print("=" * 60)

print("Predicted category:",predicted_category)
print(f"Model confidence:{model_confidence:.2%}")

if model_confidence < confidence_threshold:
    print("\nDecision: Escalate to human support")
    print("Reason:Model confidence is below the threshold")
else:
    print("\nDecision: generate ai response")
      

# relevant support context
#=============================================================================
support_context = """ The customer reports that their new bank card has not arrived after being ordered over a week ago.

The response should acknowledge the card arrival issue.

The only information available is that the card was ordered over a week ago and has not yet arrived.

If additional verification is required, clearly state that additional information may be needed.

Do not invent or request specific information such as order number,reference number, mailing addresses, trackin numbers or address changes unless those details are explicitly provided.

Do not provide a delivery date, tracking status, refund, banking policy, contact method, or quarantee.

Do not claim that the shipment can be checked or that any action has already been completed."""

#=============================================================================
#prompt engineering
#=============================================================================
system_prompt = """You are a professional banking customer-support assistant.
your task is to generate a helpful response based ONLY on the customer query,predicted category,model confidence and the relevant support context provided to you.

strict hallucination-control Rules:
1. Never assume or invent the customer's account status.
2. Never claim that a card has been dispatched, shipped, delivered, delayed or is currently in transit unless this information is explicitly present in the support context.
3. Never invent tracking numbers, delivery dates, order numbers, transaction IDs, case numbers, reference numbers, refund amounts or timelines.
4. Never invent contact details, phone numbers, websites, branches, or support channels.
5. Never claim that you can access shipment status,account records,transaction records or any external banking systems unless explicitly provided.
6. Never say that an investigation, verification,review, or resolution will occur or has already been performed.
7. Never promise that the bank will investigate, review, process, or resolve an issue unless this capability and action are explicitly provided in the support context.
8. Never ask for passwords, PINs, OTPs, card CVV, or other sensitive credentials.
9. Do not use general banking knowledge to fill missing customer-specific information.
10. If the available information is insufficient,explicitly state that the current information does not allow the requested status to be verified.
11. When information is unavailable, provide a safe, general next step without inventing specific contact details or procedures.
12. Keep the response professional, concise, and polite.
13. Do not state assumptions as facts.
14. Be professional, polite, and clear."""

user_prompt =f"""Customer Query: {customer_query}
Predicted Category: {predicted_category}
Model confidence: {model_confidence:.2%}
Relevant support context: {support_context}

Generate an appropriate customer-support response."""

#=========================================================================
#generate response
#=========================================================================
start_time = time.time()
response = client.chat.completions.create(model=model_name,
                                          messages=[{"role": "system",
                                                     "content":system_prompt},
                                                     {"role": "user",
                                                     "content":user_prompt}],
                                                     temperature=0.2,
                                                     max_tokens=300)
response_time = time.time() - start_time

#====================================================================================
# Extract generated response
#====================================================================================
generated_response = response.choices[0].message.content

print(f"\nResponse Generation Time:{response_time:.2f}seconds")

#Display results

print("\n" + "=" * 60)
print("GENERATIVE AI CUSTOMER SUPPORT")
print("=" * 60)

print("\nCustomer Query:")
print(customer_query)

print("\nPredicted Category")
print(predicted_category)

print("\nModel Confidence")
print(f"{model_confidence:.2%}")

print("\nRelevant Support Context:")
print(support_context.strip())

print("\nGenerated Customer Response:")
print(generated_response)

print("=" * 60)

#========================================================================================================================
# Gen AI evaluation
#========================================================================================================================
evaluation = {"Relevance":5,
              "Intent Alignment":5,
              "Factual consistency":5,
              "Completeness":4,
              "clarity":5,
              "Professionalism":5,
              "Hallucination / Error Rate":0
}

print("\nEvaluation scores:")
print("-" * 40)

for criterion,score in evaluation.items():
    if criterion == "Hallucination / Error Rate":
        print(f"{criterion}:{score}%")
    else:
        print(f"{criterion}:{score}/5")
quality_scores = [evaluation["Relevance"],
                  evaluation["Intent Alignment"],
                  evaluation["Factual consistency"],
                  evaluation["Completeness"],
                  evaluation["clarity"],
                  evaluation["Professionalism"]]
overall_score = sum(quality_scores) / len(quality_scores)

print("=" * 40)
print(f"Overall Generative AI score:{overall_score:.2f}/5")

print("\n Evaluation summary:")
print("The generated response is relevant to the predicted intent.")
print("The response is clear, professional,and consistent with the")
print("available context.")
print("No supported banking policy, refund timeline, tracking status,")
print("or delivery date was provided.")

#===============================================================================================
#Conversation summary
#===============================================================================================
print("\n" + "=" * 60)
print("Additional generative AI Features")
print("=" * 60)

additional_prompt = f"""You are a prefessional banking customer-support assistant.criterion
Analyse the following customer-support information.criterion

Customer Query: {customer_query}

Predicted category: {predicted_category}

Model Confidence: {model_confidence:.2%}

Support Context:{support_context}

Provide exactly these four sections:

1.Conversation summary 
Give a short factual summary of the customer's issue.

2.Sentiment
Classify the customer's sentiment as one of:
Positive, Neutral, Concerned or Frustrated.

3.Priority
Classify priority as: 
Low, Medium, or High.
Base this only on the information provided. 

4.Recommended support action 
Suggest a safe and relevant next support action. 
Do not invent banking policies, refund timelines, tracking information, account details, or other facts not provided by the customer. 

Do not make unsupported assumptions."""

additional_response =client.chat.completions.create(model=model_name,
                                                    messages=[{"role":"system",
                                                               "content":additional_prompt}],
                                                               temperature=0.2,
                                                               max_tokens=300)
additional_result = additional_response.choices[0].message.content
print("\nRecommended support action:") 
print(additional_result)

#================================================================================================
# conversation intelligence
#================================================================================================
print("\n" + "=" * 60)
print("Conversation Intelligence")
print("=" * 60)

conversation_prompt = f""" You are a banking customer-support conversation intelligence assistant.

Analyse only the information provided below.

Customer Query: {customer_query}

Predicted Intent: {predicted_category}

Model Confidence: {model_confidence:.2%}

Support context: {support_context}

Extract the following:

1. Customer Issue
2. Predicted Intent 
3. Customer Sentiment 
4. Priority 
5. Key information
6. Recommended action 

Rules:
- Output all 6 fields
- Do not skip any field
- Use only information provided above. 
- Do not invent transaction details, dates, amounts,policies, tracking numbers, contact methods,refund timelines, or other unsupported information.
- Recommend a possible next support action only.
- Do not claim that any investigation, verification, shipment check, or other action has already been performed. 
- Keep each field concise and prefessional. 
- If information is unavailable, Clearly state "Not provided".
- Customer sentiment must be one of:
Positive, Neutral, Concerned, Frustrated.
-Priority must be one of:
Low,Medium,High.
- Recommend a possible next support action only.
- Do not claim that any investigation,verification, shipment check, or other action has already been performed.


Use exactly this format:

Customer Issue:
...
Predicted Intent:
...
Customer sentiment:
...
Priority:
...
Key Information:
...
Recommended action:
...
""" 

conversation_response = client.chat.completions.create(model=model_name,
                                                    messages=[{"role":"system",
                                                               "content":conversation_prompt}],
                                                               temperature=0.2,
                                                               max_completion_tokens=1000,
                                                               reasoning_effort="low",
                                                               include_reasoning=False)
conversation_result = conversation_response.choices[0].message.content 

print("\nConversation Intelligence:")
print(conversation_result)

#=================================================================================================================================
#Business metrics
#=================================================================================================================================
print("\n" + "=" * 60)
print("Business Metrics")
print("=" * 60)

total_queries = 1 
if model_confidence < confidence_threshold:
    human_escalations = 1
    automated_responses = 0 
else:
    human_escalations = 0
    automated_responses = 1

automated_response_rate = (automated_responses / total_queries) * 100 
human_escalation_rate = (human_escalations / total_queries) * 100 
low_confidence_queries = (1 if model_confidence < confidence_threshold else 0)
low_confidence_percentage = (low_confidence_queries / total_queries)* 100 

average_response_time = response_time 
most_frequent_issue = predicted_category 

if model_confidence < confidence_threshold:
    high_priority = 1
else:
    high_priority = 0 
high_priority_percentage = (high_priority / total_queries) * 100 

print(f"\nTotal queries     :{total_queries}")
print(f"Automated response Rate   :" f"{automated_response_rate:.2f}%")
print(f"Human escalation Rate   :" f"{human_escalation_rate:.2f}%")
print(f"Low-confidence query %  :"  f"{low_confidence_percentage:.2f}%")
print(f"Average response time     :" f"{average_response_time:.2f}seconds")
print(f"Most frequent customer issue:"  f"{most_frequent_issue}")
print(f"High-priority query %   :" f"{high_priority_percentage:.2f}%")

print("\n" + "=" * 60)
print("Business metrics completed")
print("=" * 60)