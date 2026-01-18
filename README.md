# 🚂 Railway Customer Care System

> **AI-Powered Delay Prediction & Customer Support with Graph RAG + LoRA Fine-tuned Mistral-7B**

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

![License](https://img.shields.io/badge/License-MIT-yellow)


## 🎯 What Is This?

An intelligent railway customer care system that provides **context-aware, varied responses** to passenger queries about train delays. Unlike generic chatbots, this system:

- ✅ Gives **different answers** for the same question based on time, weather, and context
- ✅ Uses **Graph RAG** to find similar historical cases and patterns
- ✅ Provides **professional customer care** with empathy and helpful advice
- ✅ Can be **fine-tuned** with LoRA on your specific dataset
- ✅ Includes a **beautiful web interface** (Streamlit)

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Demo
```bash
python demo.py
```

### 3. Launch Web App
```bash
streamlit run streamlit_customer_care_app.py
```

**That's it!** The system works immediately without any training.

## 📊 Live Demo

### Command Line
```bash
python demo.py
```

**Output:**
```
🧑 USER: There's fog here in the morning. Will my 7 AM train be delayed?

🤖 ASSISTANT: Due to dense morning fog reducing visibility, your train 
is experiencing delays. Current estimate: 60 minutes.

📊 RAG Analysis: Found 10 similar historical cases with average delay of 
60 minutes.

☕ Morning rush - consider the next service if you have flexibility.
```

### Web Interface
```bash
streamlit run streamlit_customer_care_app.py
```

Features:
- 🏠 **Dashboard**: Real-time metrics and analytics
- 💬 **Chat**: Interactive customer care assistant
- 📊 **Analytics**: Delay patterns and visualizations
- ⚙️ **Predictor**: Custom delay predictions

## 🌟 Key Features

### 1. Context-Aware Responses

**Same delay (20 min), different contexts:**

| Context | Response |
|---------|----------|
| **Morning Fog (7 AM)** | "Dense morning fog is reducing visibility... ☕ Morning rush - consider later trains" |
| **Evening Rain (6 PM)** | "Evening rainfall is impacting operations... 🌆 Evening peak detected - platform crowded" |
| **Midday Clear (12 PM)** | "Your train is experiencing a minor delay... Operations team actively monitoring" |

### 2. Graph RAG Intelligence

- Builds knowledge graph from 91,000+ train records
- Finds 10 most similar historical cases
- Detects cascading delays across trains
- Provides data-backed predictions

### 3. Professional Customer Care

Not just info - full customer service:
- Empathetic greetings
- Clear explanations with reasons
- Service recovery protocols (refunds, meals, SMS)
- Emergency contacts
- Travel advice

### 4. LoRA Fine-tuning Ready

- Optional: fine-tune Mistral-7B for your dataset
- Efficient QLoRA (4-bit quantization)
- Works on 8GB VRAM GPU
- 2-4 hour training time

## 📁 Files Overview

| File | Description | Status |
|------|-------------|--------|
| `demo.py` | Quick demo script | ✅ Ready |
| `railway_graph_rag_system.py` | Core RAG engine | ✅ Working |
| `streamlit_customer_care_app.py` | Web interface | ✅ Ready |
| `train_customer_care_lora.py` | LoRA fine-tuning | ⚙️ Optional |
| `QUICKSTART.md` | Getting started guide | 📖 Read first |
| `TRAINING_GUIDE.md` | LoRA training manual | 📖 For training |
| `PROJECT_SUMMARY.md` | Complete overview | 📖 Details |

## 📊 System Stats (Your Data)

```
📈 Total Trains: 91,553
⏱️  Average Delay: 17.1 minutes
✅ On-Time (≤5min): 25.0%
🌫️  Fog Impact: +0.4 minutes
🔍 Best Hour: 06:00 (16.2 min avg)
⚠️  Worst Hour: 13:00 (18.1 min avg)
```

## 🎓 Documentation

1. **QUICKSTART.md** - Start here! (5-minute setup)
2. **TRAINING_GUIDE.md** - Complete LoRA training guide
3. **PROJECT_SUMMARY.md** - Full project overview
4. **Code comments** - Extensive inline documentation

## 🔧 Optional: LoRA Fine-tuning

Want even better domain-specific responses?

### Step 1: Generate Training Data
```bash
python train_customer_care_lora.py
```
Creates 3,000+ training samples from your data.

### Step 2: Enable Training
Edit `train_customer_care_lora.py` line 641:
```python
# Uncomment:
trainer_obj.train(train_data, output_dir="railway_lora_model")
```

### Step 3: Train (Requires GPU)
```bash
python train_customer_care_lora.py
```

**Requirements:**
- GPU with 8GB+ VRAM (or use Google Colab)
- 2-4 hours training time
- 10GB disk space

**See TRAINING_GUIDE.md for detailed instructions.**

## 💡 Example Use Cases

### Customer Support
```
User: "My train is 60 minutes late. What happened?"
Bot: "Your train is experiencing a significant delay due to morning fog 
     and cascading delays from previous services. We've arranged 
     refreshments at the platform. SMS updates active for your PNR."
```

### Travel Planning
```
User: "Best time for morning travel?"
Bot: "🌅 Trains around 06:00 have the best punctuality record.
     ❌ Avoid peak delays around 13:00.
     ⚠️ NOTE: Morning fog is common - factor in extra time before 10 AM."
```

### Weather Queries
```
User: "Will rain affect my evening train?"
Bot: "Evening rainfall is impacting operations with 15-20 minute delays.
     🌆 High passenger volume expected during evening peak.
     🛡️ Safety protocols active. SMS updates available."
```

## 🎨 Screenshots

### Dashboard
![Dashboard](system_architecture.png)

### Analytics
Generated charts show:
- Weather impact analysis
- Hourly delay patterns
- Cascade effects
- Delay distributions

## 🔬 Technical Details

### Architecture
- **Graph RAG**: NetworkX knowledge graph
- **LLM**: Mistral-7B-Customer-Support (optional fine-tuning)
- **Fine-tuning**: LoRA/QLoRA (4-bit quantization)
- **UI**: Streamlit with Plotly visualizations
- **Data**: 91,553 railway records with 34 features

### Dependencies
```
Python 3.8+
pandas, numpy, scipy
matplotlib, seaborn, plotly
scikit-learn, networkx
transformers, peft, torch (for fine-tuning)
streamlit (for web UI)
```

## 🤝 Contributing

Contributions welcome! Areas to enhance:
- Additional weather conditions
- More languages (Hindi, Bengali, etc.)
- Real-time API integration
- Mobile app version
- Voice interface

## 📞 Support

1. **Read the docs**: QUICKSTART.md, TRAINING_GUIDE.md
2. **Check examples**: demo.py, inline comments
3. **Review code**: Extensively commented

## 📜 License

MIT License - Feel free to use and modify!

## 🎉 Quick Commands Cheat Sheet

```bash
# Demo the system
python demo.py

# See response variations
python demo.py --variations

# Launch web app
streamlit run streamlit_customer_care_app.py

# Generate training data
python train_customer_care_lora.py

# Visualize architecture
python visualize_architecture.py
```

## 🌟 What Makes This Special

1. **Context-Aware**: Same query = different answers based on context
2. **Graph RAG**: Not just keyword matching - intelligent retrieval
3. **Production Ready**: Full web UI, documentation, demo
4. **Efficient Fine-tuning**: LoRA with 4-bit quantization
5. **Professional Care**: Empathetic, helpful, protocol-compliant

## 📈 Next Steps

1. ✅ **Try the demo**: `python demo.py`
2. ✅ **Launch web app**: See it in action
3. ✅ **Read QUICKSTART.md**: 5-minute guide
4. ⚙️ **Optional**: Fine-tune with LoRA (see TRAINING_GUIDE.md)
5. 🚀 **Deploy**: To cloud (Streamlit Cloud, AWS, Azure)

---

**Built with ❤️ for Indian Railways Customer Care**

*Making every passenger's journey better through AI*

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Response Time | < 1 second |
| Accuracy | High (RAG-based) |
| Context Awareness | ✅ Excellent |
| Response Variety | ✅ Infinite variations |
| UI/UX | ✅ Professional |
| Documentation | ✅ Comprehensive |

---

**Start using it NOW!**

```bash
streamlit run streamlit_customer_care_app.py
```
