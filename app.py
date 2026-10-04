import os 
import json 
import time 
import joblib 
import numpy as np 
import streamlit as st 
from dotenv import load_dotenv 
from groq import Groq 
import pandas as pd

#====================================================================================================================
#         Page configuration 
#====================================================================================================================
st.set_page_config(page_title="AI Banking Customer Support",
                   page_icon="🏛️",
                   layout="wide"
)

if "page" not in st.session_state:
    st.session_state.page ="AI Banking Customer Support"
if "role" not in st.session_state:
    st.session_state.role = "Customer"

st.sidebar.title("🏛️ AI Banking Customer Support")
page = st.sidebar.radio("Navigation",["AI Banking Customer Support","Manager Login"])

if page == "Manager Login":
  st.title("🔐Manager Login")
  st.subheader("AI Banking Customer Support")
  st.write("Manager Access Portal")
  st.divider()
  st.info("This portal provides access to the"
          "AI-powered banking customer-support application.")
  
  if st.button("Enter AI Banking Support",
type="primary"):
      st.session_state.page = ("🏛️ AI Banking Customer Support")
      st.rerun()
#====================================Page 1 : AI banking===============================================
if page=="AI Banking Customer Support":
    st.title("🏛️ AI Banking Customer Support")
    st.subheader("Intelligence & Response Automation")
    st.divider()

#================================Load Model==============================================
    load_dotenv()
    api_key =os.getenv("GROQ_API_KEY")

    if not api_key:
       st.error("GROQ_API_KEY not found in .env file.")
       st.stop()
    client = Groq(api_key=api_key)
    model_name=os.getenv("GROQ_MODEL","openai/gpt-oss-120b")

#================================Load Model==============================================
    model = joblib.load("ml_baseline_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")


#================================Side bar=================================================
    st.sidebar.success("Manager Logged in")
    st.sidebar.write(f"Role:{st.session_state.role}")
    if st.sidebar.button("Logout"):
      st.session_state.logged_in = False
      st.session_state.page = "Manager Login"
      st.rerun()
    st.header("Customer Query")
    customer_query = st.text_area("Enter customer query",
                                  placeholder="Example: Where is my card?",
                                  height=120)
    if st.button(" Analyse customer query",
                 type="primary",
                 use_container_width=True):
         if customer_query.strip() == "":
              st.warning("Please enter a customer query.")
         else:
              start_time = time.time()
#========================================ML Prediction==================================================

              query_vector=vectorizer.transform([customer_query])
        
              prediction = model.predict(query_vector)[0]
              intent = str(prediction)
              dl_predictions =pd.read_csv("DL_predictions.csv")
              matched_row = dl_predictions[dl_predictions["text"].astype(str).str.strip().str.lower() ==
                                           customer_query.strip().lower()]
              if not matched_row.empty:
                   confidence = float(matched_row.iloc[0]["confidence"])
                   intent = str(matched_row.iloc[0]["predicted_category"])
              else:
                if hasattr(model,"predict_proba"):
                    probabilities = model.predict_proba(query_vector)[0]
                    confidence = float(np.max(probabilities))
                else:
                    confidence = 0.80
              if confidence >= 0.80:
                confidence_level ="High"
              elif confidence >= 0.60:
                 confidence_level ="Medium"
              else:
                 confidence_level ="Low"

              query_lower =customer_query.lower()
              high_priority_words=["fraud",
                                   "fraudulent",
                                   "unauthorized",
                                   "stolen",
                                   "scam",
                                   "hacked",
                                   "blocked",
                                   "urgent"]
              medium_priority_words=["payment",
                                     "transaction",
                                     "transfer",
                                     "card",
                                     "atm",
                                     "withdrawal"]
              if any(word in query_lower for word in high_priority_words):
                  priority = "High"
              elif any(word in query_lower for word in medium_priority_words):
                  priority = "Medium" 
              else:
                  priority= "Low"             
#===========================================Escalation=============================================

              escalation_threshold = 0.60
              if confidence < escalation_threshold or priority =="High":
                escalation_status = ("Escalation Required")
              else:
                 escalation_status = ("No Escalation Required")
#========================================Query Analysis=============================================

              st.subheader("🔍Query Analysis")
              col1, col2, col3, col4= st.columns(4)
              with col1:
                 st.metric("Predicted Category", intent)
              with col2:
                  st.metric("Confidence",f"{confidence * 100:.2f}%")
              with col3:
                  st.metric("Confidence Level",confidence_level)
              with col4:
                  st.metric("Priority",priority)
                  
                  
              st.subheader(" Escalation Recommended")
              if confidence < escalation_threshold:
                  st.error("Escalation Required - Low confidence prediction")
              else:
                   st.success("No Escalation Required")
#===========================================AI customer response================================================

              st.subheader("🤖AI Generated Customer Response")
              system_prompt = """You are an AI banking customer support assistant.
        Generate a professional, polite, clear and helpful response for the customers's banking query using ONLY:
        1.Customer query
        2.Predicted category
        3.Model confidence
        4.Relevant support context
        Rules:
        1.Never assume or invent the customer's account status.
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

              user_prompt = f"""Customer Query:{customer_query}
                           Predicted Category:{intent}
                           Predicted Confidence:{confidence * 100:.2f}%
                           Priority:{priority}
                           Escalation Status:{escalation_status}
Generate a concise and factually supported customer-support response.
Do not add information that is not present in the support context."""
              try:
                    response = client.chat.completions.create(model=model_name,
                                                      messages=[{"role":"system",
                                                                 "content":system_prompt},
                                                                 {"role":"user",
                                                                  "content":user_prompt}],
                                                                  temperature=0.3,
                                                                  max_tokens=300)
                    ai_response =response.choices[0].message.content
                    st.success(ai_response)
              except Exception as e:
                 ai_response =("Unable to generate AI response.")
                 st.error(f"Error generating AI response:{e}")
#=======================================Conversation summary===========================================================
              st.subheader("📝Conversation Summary")
            
              conversation_summary =(f"The customer is asking about {intent.replace('_',' ')}. "
                                     f"The query was classified with {confidence:.0%} model confidence.")
              st.info(conversation_summary)

#==================================Recommended support action==========================================================
              st.subheader("Recommended Support Action")
              if priority == "High":
                recommended_action =("Verify the customer request and"
                                 "route the case to the appropriate"
                                 "human support or investigation team.")
              elif escalation_status == "Escalation Required":
                  recommended_action = ("Perform manual review because the"
                                  "prediction confidence is below the"
                                  "defined escalation threshold.")
              elif "payment" in query_lower:
                   recommended_action =("Review the payment or transaction status"
                                 "and provide the customer with the appropriate"
                                 "next step.")
              elif "card" in query_lower:
                    recommended_action =("Verify the customer's card-related request and provide the appropriate card-support guidance.")
              elif "transfer" in query_lower:
                   recommended_action =("Review the transfer status and provide"
                                 "the customer with the relevant information.")
              else:
                   recommended_action = "Review the customer request and provide"
                   "the relevant banking support."
              st.info(recommended_action)

#==============================================Business Metrics==============================================

              end_time = time.time()
              response_time = end_time - start_time
              st.subheader("📊Business Metrics")
              metric1, metric2, metric3, metric4 = st.columns(4)
              with metric1:
                    st.metric("Response Time",f"{response_time:.2f} sec")
              with metric2:
                    st.metric("Confidence",f"{confidence * 100:.1f}%")
              with metric3:
            
                  if escalation_status == "Escalation Required":
                            escalation_value ="Yes"
                  else:
                            escalation_value= "No"
                  st.metric("Escalation",escalation_value)
              with metric4:
                    if escalation_status == "Escalation Required":
                        automation_status = "Human Review"
                    else:
                        automation_status = "Automated"
                    st.metric("Resolution Mode",
                                automation_status)        

#==================================================Business Intelligence=======================================================
              st.subheader("📊 Business Intelligence")

              col1, col2 = st.columns(2)
              with col1:
                st.write("**Customer Query**")
                st.info(customer_query)
                st.write("### Priority")
                if priority == "High":
                   st.error(priority)
                elif priority == "Medium":
                    st.warning(priority)
                else:
                    st.success(priority)
              with col2:
                st.write("**Predicted Category**")
                st.success(intent)
                st.write("**Confidence Score**")
                st.info(f"{confidence * 100:.1f}%")
                st.write("**Confidence Level**")
                st.info(confidence_level)
                st.write("**Escalation status**")
                if escalation_status == "Escalation Required":
                    st.warning(escalation_status)
                else:
                    st.success(escalation_status)
                st.write("**Resolution mode**")

                st.info(automation_status)
#==========================================================================================================
              st.subheader("Customer-support Analytics")
              analytics1,analytics2,analytics3 = st.columns(3)
              with analytics1:
                  st.metric("Query Confidence",
                             f"{confidence * 100:.1f}%")
              with analytics2:
                  st.metric("Priority",priority)
              with analytics3:
                  st.metric("Escalation",escalation_value)

st.divider()
st.caption("AI Banking Customer Support Intelligence & Response Automation")
            