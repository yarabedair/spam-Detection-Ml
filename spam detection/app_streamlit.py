import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def preprocess(text):
    text = text.lower().strip()
    return text

st.title("📩 Spam Message Classifier")
st.write("أدخل نص الرسالة بالأسفل للتأكد إذا كانت رسالة مزعجة أم لا")

user_input = st.text_area("الرسالة:", placeholder="اكتب رسالتك هنا...")

if st.button("تحليل الرسالة"):
    if user_input:
        clean_text = preprocess(user_input)
        vectorized_text = vectorizer.transform([clean_text])
        
        prediction = model.predict(vectorized_text)
        
        if prediction[0] == 'spam':
            st.error("⚠️ تحذير: هذه الرسالة غالباً رسالة مزعجة (Spam)")
        else:
            st.success("✅ هذه الرسالة تبدو آمنة (Ham)")
    else:
        st.warning("من فضلك اكتب نصاً أولاً")