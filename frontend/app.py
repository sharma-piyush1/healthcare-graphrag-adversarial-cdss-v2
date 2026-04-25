import streamlit as st
import requests
import json

# The endpoint of your new microservice
API_URL = "http://localhost:8000/api/v1/clinical-query"

st.set_page_config(page_title="v2 Clinical API Client", page_icon="🛡️", layout="wide")
st.title("🛡️ CDSS API Interface")
st.markdown("This frontend is decoupled and communicates with the LangGraph engine via REST API.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "audit_logs" not in st.session_state:
    st.session_state.audit_logs = []

with st.sidebar:
    st.header("🛡️ API Audit Trace")
    if st.session_state.audit_logs:
        for idx, log in enumerate(reversed(st.session_state.audit_logs)):
            with st.expander(f"Query: {log['query'][:25]}...", expanded=(idx==0)):
                st.write(f"**Data Source:** {log['source']}")
                st.write(f"**Iterations:** {log['retries']}")
                st.write(f"**Status:** {'✅ Approved' if log['approved'] else '⚠️ Rejected'}")
                if not log['approved'] or log['retries'] > 1:
                    st.warning(f"**Critique:**\n{log['critique']}")
    else:
        st.write("No system audits performed yet.")
    
    st.markdown("---")
    st.header("📄 Upload Lab Report")
    uploaded_file = st.file_uploader("Upload an image of a clinical note or lab result", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        if st.button("Analyze Report"):
            with st.spinner("Analyzing Multimodal Document..."):
                try:
                    # Send file to the new endpoint
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL.replace('/clinical-query', '/analyze-report')}", files=files)
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    st.success("Analysis Complete!")
                    # Add the result directly to the chat
                    st.session_state.messages.append({"role": "assistant", "content": f"**[Document Analysis]**\n\n{result['generation']}"})
                    st.rerun()
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"API Error: {e}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter clinical query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Awaiting response from FastAPI Backend..."):
            try:
                # Prepare payload for the API
                payload = {
                    "query": prompt,
                    "chat_history": st.session_state.messages[-4:]
                }
                
                # Make the HTTP POST request
                response = requests.post(API_URL, json=payload)
                response.raise_for_status() # Raise exception for bad status codes
                
                result = response.json()
                
                res_text = result["generation"]
                source = result["source_type"]
                is_approved = result["is_approved"]
                critique = result["audit_report"]
                retries = result["retry_count"]

                st.caption(f"API Source: **{source}** | Audited by **120B Model** | Loops: **{retries}**")
                st.markdown(res_text)
                
                st.session_state.messages.append({"role": "assistant", "content": res_text})
                st.session_state.audit_logs.append({
                    "query": prompt,
                    "source": source,
                    "critique": critique,
                    "approved": is_approved,
                    "retries": retries
                })
                
                st.rerun()

            except requests.exceptions.ConnectionError:
                st.error("🚨 API Connection Error: Is the FastAPI server running on port 8000?")
            except Exception as e:
                st.error(f"Frontend Request Error: {str(e)}")