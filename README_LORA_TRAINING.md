# 🚂 Railway Customer Care System - LoRA Fine-tuning Guide

## 📋 Overview

This system combines **Graph RAG** with **LoRA fine-tuned Mistral-7B** to create an intelligent railway customer care chatbot that provides:

- ✅ Context-aware delay predictions
- ✅ Weather-specific recommendations
- ✅ Time-based travel advice
- ✅ Cascading delay analysis
- ✅ Personalized customer responses

---

## 🎯 Key Features

### 1. **LoRA/QLoRA Fine-tuning**
- Fine-tune `bitext/Mistral-7B-Customer-Support` on your railway dataset
- Efficient 4-bit quantization (QLoRA) for consumer GPUs
- Only ~16M trainable parameters (vs 7B total)
- Train on single GPU in 2-4 hours

### 2. **Contextual Response Generation**
The model learns to provide varied responses based on:
- **Time of day** (morning fog vs evening rain)
- **Weather conditions** (fog, rain, storm, clear)
- **Delay severity** (on-time, minor, major, severe)
- **Cascading effects** (previous train delays)

### 3. **Customer Care Features**
- Empathetic, professional responses
- Real-time delay predictions
- Service recovery protocols
- Alternative recommendations
- Emergency contact information

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Core dependencies
pip install pandas numpy networkx matplotlib seaborn scikit-learn

# For LoRA fine-tuning
pip install torch transformers datasets peft accelerate bitsandbytes scipy trl

# For Streamlit app
pip install streamlit plotly
```

### Step 2: Generate Training Dataset

```bash
python train_customer_care_lora.py
```

This will:
1. Load your `data.csv` 
2. Generate 3,000+ context-aware training samples
3. Save to `railway_training_data.json`

**Sample Training Data:**
```
User: "Will morning rain affect my 8 AM train?"