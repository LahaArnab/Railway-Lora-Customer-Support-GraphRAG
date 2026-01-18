# 🚂 Railway Customer Care - Complete Training Guide

## 📋 Overview

This system fine-tunes **Mistral-7B-Customer-Support** using **LoRA/QLoRA** on your railway dataset to provide intelligent, context-aware customer care responses.

---

## ✨ Key Improvements

### 1. **Context-Aware Responses**
Unlike generic responses, the model provides varied answers based on:
- **Morning rain**: "Morning rain showers are affecting track conditions..."
- **Evening fog**: "Evening fog conditions require cautious operations..."
- **Storm at 6 PM**: "Severe weather during evening peak hours..."

### 2. **Time-Specific Advice**
- Morning fog warnings
- Evening rush hour alerts
- Best travel time recommendations

### 3. **Professional Customer Care**
- Empathetic greetings
- Clear delay explanations
- Service recovery options (refunds, meals, SMS updates)
- Emergency contact information

---

## 🚀 Installation

```bash
# Step 1: Core dependencies
pip install pandas numpy networkx matplotlib seaborn scikit-learn

# Step 2: LoRA fine-tuning (requires GPU)
pip install torch transformers datasets peft accelerate bitsandbytes scipy trl

# Step 3: Streamlit UI
pip install streamlit plotly
```

**GPU Requirements:**
- Minimum: 8GB VRAM (RTX 3060, RTX 4060)
- Recommended: 12GB+ VRAM (RTX 3080, RTX 4070)
- Cloud: Google Colab (free T4), Kaggle, AWS/Azure

---

## 📊 Step 1: Generate Training Dataset

```bash
python train_customer_care_lora.py
```

**What it does:**
1. Loads your `data.csv` (91,000+ railway records)
2. Generates 3,000+ training conversations
3. Creates varied user queries for each scenario
4. Generates context-aware responses
5. Saves to `railway_training_data.json`

**Sample Training Example:**

```
User Query:
"Will morning rain affect my 8 AM train?"

Model Response:
"Thank you for contacting Indian Railways Customer Care. Your train is experiencing 
a moderate delay of approximately 18 minutes. This is primarily due to morning rain 
showers affecting track conditions. Wet tracks require increased braking distances 
and speed restrictions.

☕ Morning Rush Advisory: Platform may be crowded. If you have flexibility, the 
next service might be a better option.

💡 Helpful tip: You can track live status via SMS by sending your PNR to 139 or 
through our mobile app.

We apologize for the inconvenience and appreciate your patience. Safe journey!"
```

**Dataset Features:**
- 3,000+ diverse conversations
- Multiple query variations per condition
- Context-aware weather descriptions
- Time-based recommendations
- Delay-specific service protocols

---

## 🔥 Step 2: Train with LoRA

### Option A: Full Training (Recommended)

**Edit `train_customer_care_lora.py`:**
```python
# Line 641: Uncomment this line
trainer_obj.train(train_data, output_dir="railway_lora_model")
```

**Run training:**
```bash
python train_customer_care_lora.py
```

**Training Configuration:**
- **Model**: Mistral-7B-Customer-Support (7B parameters)
- **Method**: QLoRA (4-bit quantization)
- **Trainable**: ~16M parameters (0.2% of model)
- **Epochs**: 3
- **Batch Size**: 4 (with gradient accumulation)
- **Learning Rate**: 2e-4
- **Time**: 2-4 hours on single GPU
- **VRAM**: ~8-12GB

**Expected Output:**
```
Step    Loss    Time
10      2.456   45s
50      1.823   3.2min
100     1.234   6.5min
...
Training complete! Model saved to railway_lora_model/
```

### Option B: Quick Test (CPU)

For testing without GPU:
```python
# Use smaller sample
samples = generator.generate_training_samples(n_samples=100)

# Skip training, use rule-based system
# The railway_graph_rag_system.py works without fine-tuning
```

---

## 🎯 Step 3: Run the System

### A. Command Line Interface

```bash
python railway_graph_rag_system.py
```

**Features:**
- 4-step operational system
- Fog impact analysis
- Hourly delay patterns
- Cascade detection
- Context-aware demo queries

### B. Streamlit Web App

```bash
streamlit run streamlit_customer_care_app.py
```

**Features:**
- 🏠 **Home Dashboard**: Network metrics, weather impact charts
- 💬 **Customer Care Chat**: Interactive chatbot with context awareness
- 📊 **Network Analytics**: Delay distributions, cascade analysis
- ⚙️ **Delay Predictor**: Custom condition prediction tool

**Chat Interface:**
- Real-time responses
- Quick action buttons
- Chat history
- Context extraction (time, weather)
- Professional formatting

---

## 📁 File Structure

```
TRY 4/
├── data.csv                          # Your railway dataset (91K records)
├── railway_graph_rag_system.py       # Core Graph RAG system
├── train_customer_care_lora.py       # LoRA fine-tuning script
├── streamlit_customer_care_app.py    # Web interface
├── TRAINING_GUIDE.md                 # This file
├── railway_training_data.json        # Generated training data
└── railway_lora_model/               # Fine-tuned model (after training)
    ├── adapter_config.json
    ├── adapter_model.bin
    └── ...
```

---

## 🎓 Understanding the System

### Graph RAG Architecture

1. **Graph Construction**
   - Nodes: Stations, Trains
   - Edges: Cascading delays
   - Attributes: Weather, time, delay

2. **Retrieval (RAG)**
   - Find similar historical cases
   - Extract relevant patterns
   - Calculate averages

3. **Generation**
   - Base prediction model
   - Context-aware adjustments
   - Human-like responses

### LoRA Fine-tuning Benefits

**Without LoRA (Base Model):**
```
User: "Morning fog delay?"
Bot: "Delays may occur due to weather conditions."
```

**With LoRA (Fine-tuned):**
```
User: "Morning fog delay?"
Bot: "Dense morning fog is affecting operations with ~38 min delays. 
Visibility typically improves after 10 AM. Safety is our priority. 
Consider later trains if possible.

📱 SMS alerts active for your PNR. Customer care: 139"
```

### Response Variations

The system generates **different responses** for the same condition based on:

**Morning Rain:**
- "Morning rain showers are affecting track conditions..."
- "Wet tracks require increased braking distances..."
- "☕ Morning rush - platform may be crowded..."

**Evening Rain:**
- "Evening rainfall is impacting operations..."
- "🌆 Evening peak detected - high passenger volume expected..."

**Storm:**
- "Severe weather conditions require enhanced safety protocols..."
- "Your safety is our top priority..."

---

## 🔧 Customization

### Adjust Training Parameters

```python
# train_customer_care_lora.py

# More training data
samples = generator.generate_training_samples(n_samples=5000)

# Longer training
training_args = TrainingArguments(
    num_train_epochs=5,  # Instead of 3
    per_device_train_batch_size=2,  # For less VRAM
)

# Higher LoRA rank (more capacity)
lora_config = LoraConfig(
    r=32,  # Instead of 16
    lora_alpha=64,
)
```

### Add Custom Responses

```python
# railway_graph_rag_system.py

def _generate_response(self, delay, weather, ...):
    # Add your custom logic
    if weather == 'Snow':
        response += "Snow conditions detected. Special advisory in effect..."
```

---

## 🚀 Performance Optimization

### For Faster Training

1. **Use QLoRA** (already enabled)
   ```python
   use_qlora=True  # 4-bit quantization
   ```

2. **Reduce batch size**
   ```python
   per_device_train_batch_size=2
   gradient_accumulation_steps=8
   ```

3. **Mixed precision**
   ```python
   fp16=True
   ```

### For Better Quality

1. **More training data**
   ```python
   n_samples=5000  # Instead of 3000
   ```

2. **More epochs**
   ```python
   num_train_epochs=5
   ```

3. **Higher LoRA rank**
   ```python
   r=32, lora_alpha=64
   ```

---

## 📊 Evaluation

### Test the Model

```python
# After training
bot = RailwayCustomerCareBot(lora_path="railway_lora_model")
bot.load_model()

# Test queries
queries = [
    "Morning fog impact?",
    "Will rain delay my 6 PM train?",
    "Train 60 minutes late - why?",
]

for q in queries:
    print(f"Q: {q}")
    print(f"A: {bot.chat(q)}\n")
```

### Metrics to Track

- **Perplexity**: Lower is better (< 2.0)
- **Loss**: Should decrease over training
- **User feedback**: Response quality ratings
- **Response variety**: Different responses for same conditions

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size
per_device_train_batch_size=1
gradient_accumulation_steps=16

# Or use CPU (much slower)
device_map="cpu"
```

### Model Not Loading

```bash
# Reinstall dependencies
pip install --upgrade transformers peft

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Poor Response Quality

1. Generate more training data
2. Increase training epochs
3. Check dataset quality
4. Adjust temperature in pipeline

---

## 🌟 Advanced Features

### Multi-GPU Training

```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
)
```

### Custom Evaluation

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Calculate your custom metrics
    return {"accuracy": acc, "f1": f1}

trainer = SFTTrainer(
    compute_metrics=compute_metrics,
    evaluation_strategy="steps",
    eval_steps=100,
)
```

---

## 📞 Support & Resources

### Documentation
- Hugging Face: https://huggingface.co/docs
- PEFT: https://huggingface.co/docs/peft
- Mistral: https://docs.mistral.ai

### Community
- GitHub Issues: Report bugs
- Discord: Real-time help
- Stack Overflow: Q&A

---

## ✅ Checklist

Before training:
- [ ] GPU available (8GB+ VRAM)
- [ ] Dependencies installed
- [ ] data.csv exists (91K+ records)
- [ ] Disk space (10GB+ free)

After training:
- [ ] Model saved to railway_lora_model/
- [ ] Test queries working
- [ ] Streamlit app running
- [ ] Responses context-aware

---

## 🎉 Next Steps

1. **Test the fine-tuned model** with various queries
2. **Deploy to production** (FastAPI, Flask, Streamlit Cloud)
3. **Collect user feedback** for continuous improvement
4. **Monitor performance** metrics
5. **Retrain periodically** with new data

---

## 📈 Expected Results

### Before Fine-tuning
- Generic responses
- No context awareness
- Same output for different conditions
- Limited railway knowledge

### After Fine-tuning
- Context-specific responses
- Time-aware recommendations
- Weather-based variations
- Railway domain expertise
- Professional customer care tone
- Service recovery protocols

---

**Happy Training! 🚂💨**

For questions or issues, check the troubleshooting section or review the code comments.
