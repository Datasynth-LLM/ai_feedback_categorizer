# 🤖 AI Feedback Categorizer

An AI-powered feedback analysis system that classifies user reviews into actionable categories like bugs, feature requests, compliments, and more — with real-time sentiment, urgency, keyword trends, and business insights.

![Banner](https://user-images.githubusercontent.com/your-banner-image.png) <!-- Optional -->

---

## 🔍 Project Overview

**Domain:** Customer Intelligence / NLP  
**Tools:** Streamlit · Hugging Face Transformers · Scikit-learn · Pandas · Seaborn · Altair · WordCloud  
**Goal:** Automate the classification and analysis of user feedback using LLMs to save time, detect critical issues early, and extract business insights.

---

## 🚀 Features

- 🧠 **Zero-shot Classification** (Bug, Feature Request, Complaint, Compliment, General)
- 💬 **Sentiment Analysis** (Positive / Negative / Neutral with scores)
- ⚠️ **Urgency Scoring** (High / Medium / Low based on heuristics)
- 🏷️ **Keyword Extraction & WordCloud**
- 📈 **Real-time Dashboard** with interactive charts
- 📊 **Business Insights** (weekly summaries, counts, urgency %, sentiment dist.)
- 🧾 **CSV Upload + Download Processed Feedback**
- 🔔 **Alert-ready architecture** for spikes in urgent/negative feedback
- 🧵 **Multithreading for large datasets**
- ☁️ **Streamlit Cloud-Ready**

---

## 🛠️ Setup Instructions

### ⚙️ 1. Clone the Repository
```bash
git clone https://github.com/Datasynth-LLM/ai-feedback-categorizer.git
cd ai-feedback-categorizer
