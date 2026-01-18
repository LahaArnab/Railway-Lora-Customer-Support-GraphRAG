"""
===============================================================================
Railway Customer Care - LoRA/QLoRA Fine-tuning
Fine-tune Mistral-7B-Customer-Support for Railway Delay Queries
===============================================================================

Installation:
pip install torch transformers datasets peft accelerate bitsandbytes scipy trl

Features:
- QLoRA (4-bit quantization) for efficient training
- Custom railway delay dataset generation
- Context-aware customer care responses
- Weather-time-delay correlation training

Run:
python train_customer_care_lora.py
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import SFTTrainer
import warnings
warnings.filterwarnings('ignore')


class RailwayDatasetGenerator:
    """Generate training dataset from railway data"""
    
    def __init__(self, data_path='data.csv'):
        self.df = pd.read_csv(data_path)
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare data with context features"""
        self.df['WeatherCondition'] = self.df['WeatherCondition'].fillna('Clear')
        self.df['Departure_Hour'] = pd.to_datetime(
            self.df['Scheduled_Departure_Time'], 
            format='%H:%M', 
            errors='coerce'
        ).dt.hour
        self.df['Is_Morning'] = (self.df['Departure_Hour'].between(6, 11)).astype(int)
        self.df['Is_Evening'] = (self.df['Departure_Hour'].between(17, 21)).astype(int)
        self.df['Delay_Category'] = pd.cut(
            self.df['Arrival_Delay_min'],
            bins=[-1, 5, 15, 30, 60, 1000],
            labels=['OnTime', 'Minor', 'Moderate', 'Major', 'Severe']
        )
    
    def generate_training_samples(self, n_samples=5000):
        """Generate diverse training samples"""
        print(f"\n📊 Generating {n_samples} training samples...")
        
        samples = []
        
        # Sample from different delay categories and conditions
        for _, row in self.df.sample(n=min(n_samples, len(self.df))).iterrows():
            # Extract context
            delay = row['Arrival_Delay_min']
            weather = row['WeatherCondition']
            hour = row['Departure_Hour'] if not pd.isna(row['Departure_Hour']) else 12
            is_morning = bool(row['Is_Morning'])
            is_evening = bool(row['Is_Evening'])
            prev_delay = row['Previous_Train_Delay_min']
            day = row['Day_of_Week']
            
            # Generate varied user queries
            query_templates = self._get_query_templates(
                weather, hour, is_morning, is_evening, delay
            )
            
            # Generate appropriate assistant response
            response = self._generate_response(
                delay, weather, hour, is_morning, is_evening, 
                prev_delay, day, row
            )
            
            # Create conversation format
            for query_template in query_templates[:2]:  # Use 2 variations per sample
                conversation = self._format_conversation(query_template, response)
                samples.append(conversation)
        
        print(f"✅ Generated {len(samples)} training conversations")
        return samples
    
    def _get_query_templates(self, weather, hour, is_morning, is_evening, delay):
        """Generate varied user query templates"""
        queries = []
        
        # Time-based queries
        if is_morning:
            queries.extend([
                f"Will my {int(hour):02d}:00 morning train be delayed?",
                f"I have a train at {int(hour)} AM. What's the delay status?",
                f"Morning train schedule check for {int(hour):02d}:00",
                f"Is there any delay for early morning trains around {int(hour)}?"
            ])
        elif is_evening:
            queries.extend([
                f"Evening train delay info for {int(hour):02d}:00",
                f"My {int(hour)} PM train - will it be on time?",
                f"Check delay status for evening departure at {int(hour)}:00"
            ])
        else:
            queries.extend([
                f"Train delay prediction for {int(hour):02d}:00",
                f"What's the expected delay at {int(hour)}:00?",
                f"Delay status inquiry for {int(hour):02d}:00 train"
            ])
        
        # Weather-specific queries
        if weather == 'Fog':
            queries.extend([
                f"There's fog here. Will my {int(hour):02d}:00 train be affected?",
                f"Fog visibility issues - train delay expected?",
                f"Morning fog impact on {int(hour)} AM train?"
            ])
        elif weather == 'Rain':
            queries.extend([
                f"It's raining. How will this affect my train?",
                f"Rain delays for {int(hour):02d}:00 departure?",
                f"Heavy rainfall - train schedule status?"
            ])
        elif weather == 'Storm':
            queries.extend([
                f"Storm warning - is my train safe?",
                f"Will storm delay my {int(hour):02d}:00 train?",
                f"Severe weather impact on railway services?"
            ])
        
        # Delay-specific queries
        if delay > 60:
            queries.extend([
                "My train is very late. What's happening?",
                "Major delay - need information urgently",
                "When will my delayed train arrive?"
            ])
        elif delay > 30:
            queries.extend([
                "Train running late - how much delay?",
                "Moderate delay expected?",
                "Update on train arrival time?"
            ])
        else:
            queries.extend([
                "Is my train on time?",
                "Quick status check for my train",
                "Train punctuality status?"
            ])
        
        return queries
    
    def _generate_response(self, delay, weather, hour, is_morning, 
                          is_evening, prev_delay, day, row):
        """Generate context-aware customer care response"""
        
        # Greeting variations
        greetings = [
            "Thank you for contacting Indian Railways Customer Care.",
            "Hello! I'm here to help with your train inquiry.",
            "Greetings from Railway Customer Support.",
            "Welcome to Railway Assistance."
        ]
        
        import random
        greeting = random.choice(greetings)
        response = f"{greeting} "
        
        # Delay status
        if delay <= 5:
            status_msgs = [
                f"Great news! Your train is running on schedule.",
                f"Excellent! Your train is maintaining its timetable.",
                f"You're in luck - your train is on time.",
                f"Good news! Minimal delay of just {int(delay)} minutes expected."
            ]
            response += random.choice(status_msgs)
        
        elif delay <= 15:
            response += f"Your train is experiencing a minor delay of approximately {int(delay)} minutes. "
            
        elif delay <= 30:
            response += f"Your train has a moderate delay of about {int(delay)} minutes. "
            
        elif delay <= 60:
            response += f"Your train is delayed by approximately {int(delay)} minutes. "
            
        else:
            response += f"Your train is experiencing a significant delay of {int(delay)} minutes. "
        
        # Weather context
        if weather == 'Fog' and is_morning:
            response += "This is primarily due to dense morning fog affecting visibility across the route. "
            response += "Safety protocols require reduced speeds during low visibility conditions. "
            response += "Fog typically clears by late morning, and subsequent trains often recover lost time. "
        
        elif weather == 'Fog':
            response += "Fog conditions in the region are causing reduced visibility. "
            response += "Our priority is passenger safety, requiring cautious operations. "
        
        elif weather == 'Rain' and is_morning:
            response += "Morning rain showers are affecting track conditions. "
            response += "Wet tracks require increased braking distances and speed restrictions. "
        
        elif weather == 'Rain' and is_evening:
            response += "Evening rainfall is impacting operations. "
            response += "Weather-related safety measures are in effect. "
        
        elif weather == 'Rain':
            response += "Rainfall in the region is affecting schedule. "
            response += "Track conditions require cautious operation. "
        
        elif weather == 'Storm':
            response += "Severe weather conditions require enhanced safety protocols. "
            response += "Your safety is our top priority, and we're monitoring the situation closely. "
        
        # Cascade impact
        if prev_delay > 30:
            response += f"Additionally, cascading delays from previous services (affecting by {int(prev_delay)} minutes) are contributing to the current situation. "
        elif prev_delay > 15:
            response += "Some upstream delays are also affecting the schedule. "
        
        # Track issues
        if row.get('Track_Maintenance') == 'Yes':
            response += "Scheduled track maintenance is also in progress. "
        if row.get('Signal_Failure') == 'Yes':
            response += "Signal system issues are being addressed by our technical team. "
        
        # Time-specific advice
        if is_morning and delay > 20:
            response += "\n\n☕ Morning Rush Advisory: Platform may be crowded. If you have flexibility, the next service might be a better option. "
        elif is_evening and delay > 20:
            response += "\n\n🌆 Evening Peak: High passenger volume expected. Please plan accordingly. "
        
        # Service recovery measures
        if delay >= 240:
            response += "\n\n🚨 MAJOR DELAY ASSISTANCE:\n"
            response += "• Complimentary meals are being arranged at the station\n"
            response += "• Full refund or free rebooking available\n"
            response += "• SMS updates will be sent every 30 minutes to your registered number\n"
            response += "• For immediate assistance, please call our 24/7 helpline: 139\n"
            response += "• Our station staff is ready to help at the inquiry counter"
        
        elif delay >= 120:
            response += "\n\n⚠️ DELAY SUPPORT SERVICES:\n"
            response += "• Refreshments available at platform vendors\n"
            response += "• Live tracking updates via our mobile app\n"
            response += "• Alternative route options being evaluated\n"
            response += "• Customer care: 139 for real-time assistance"
        
        elif delay >= 60:
            response += "\n\n📱 STAY CONNECTED:\n"
            response += "• SMS alerts are active for your PNR\n"
            response += "• Download our app for live tracking\n"
            response += "• Platform displays will show updated timings"
        
        elif delay >= 30:
            response += "\n\n💡 Helpful tip: You can track live status via SMS by sending your PNR to 139 or through our mobile app."
        
        # Positive closing
        closings = [
            "\n\nWe apologize for the inconvenience and appreciate your patience. Safe journey!",
            "\n\nThank you for your understanding. We're working to minimize delays. Have a pleasant journey!",
            "\n\nWe regret the inconvenience. Your comfort and safety are our priorities. Travel safely!",
            "\n\nApologies for the delay. We value your patience. Wishing you a safe journey!",
            "\n\nThank you for traveling with us. We're committed to getting you to your destination safely!"
        ]
        
        if delay > 5:
            response += random.choice(closings)
        else:
            response += "\n\nHave a wonderful journey with Indian Railways!"
        
        return response
    
    def _format_conversation(self, user_query, assistant_response):
        """Format as instruction-following conversation"""
        return {
            "text": f"""<s>[INST] You are a helpful and empathetic Indian Railways Customer Care assistant. Provide accurate, context-aware information about train delays while being professional and supportive.

User: {user_query} [/INST]

{assistant_response}</s>"""
        }
    
    def save_dataset(self, samples, output_path='railway_training_data.json'):
        """Save dataset to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved dataset to {output_path}")
        return output_path


class RailwayLoRATrainer:
    """LoRA/QLoRA trainer for customer care model"""
    
    def __init__(self, model_name="bitext/Mistral-7B-Customer-Support", use_qlora=True):
        self.model_name = model_name
        self.use_qlora = use_qlora
        self.model = None
        self.tokenizer = None
        
    def setup_model(self):
        """Setup model with QLoRA configuration"""
        print(f"\n🤖 Loading model: {self.model_name}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        if self.use_qlora:
            # QLoRA: 4-bit quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # Prepare for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            print("✅ Model loaded with QLoRA (4-bit)")
        
        else:
            # Standard LoRA (16-bit)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            print("✅ Model loaded with LoRA (16-bit)")
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,  # Alpha scaling
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        print("\n📊 Model Parameters:")
        self.model.print_trainable_parameters()
        
        return self.model, self.tokenizer
    
    def train(self, train_dataset, output_dir="railway_lora_model"):
        """Train with LoRA"""
        print(f"\n🚀 Starting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            warmup_steps=50,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit" if self.use_qlora else "adamw_torch",
            report_to="none",  # Disable wandb
        )
        
        # SFT Trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=1024,
            packing=False,
        )
        
        # Train
        print("\n" + "="*80)
        print("TRAINING IN PROGRESS")
        print("="*80)
        trainer.train()
        
        # Save
        trainer.save_model(output_dir)
        print(f"\n✅ Model saved to {output_dir}")
        
        return trainer


class RailwayCustomerCareBot:
    """Inference with fine-tuned model"""
    
    def __init__(self, base_model="bitext/Mistral-7B-Customer-Support", 
                 lora_path="railway_lora_model"):
        self.base_model = base_model
        self.lora_path = lora_path
        self.pipeline = None
        
    def load_model(self):
        """Load fine-tuned model"""
        print(f"\n🔄 Loading fine-tuned model from {self.lora_path}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        # Load base model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(base, self.lora_path)
        
        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        
        print("✅ Model loaded successfully!")
        return self.pipeline
    
    def chat(self, user_query):
        """Generate response"""
        prompt = f"""<s>[INST] You are a helpful and empathetic Indian Railways Customer Care assistant. Provide accurate, context-aware information about train delays while being professional and supportive.

User: {user_query} [/INST]"""
        
        response = self.pipeline(prompt)[0]['generated_text']
        
        # Extract assistant response (after [/INST])
        assistant_response = response.split("[/INST]")[-1].strip()
        return assistant_response


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("🚂 RAILWAY CUSTOMER CARE - LoRA FINE-TUNING")
    print("   Mistral-7B-Customer-Support + Railway Context")
    print("="*80)
    
    # Step 1: Generate dataset
    print("\n" + "="*80)
    print("STEP 1: DATASET GENERATION")
    print("="*80)
    
    generator = RailwayDatasetGenerator('data.csv')
    samples = generator.generate_training_samples(n_samples=3000)
    dataset_path = generator.save_dataset(samples)
    
    # Convert to HuggingFace dataset
    train_data = Dataset.from_list(samples)
    print(f"✅ Training dataset: {len(train_data)} samples")
    
    # Step 2: Train with LoRA
    print("\n" + "="*80)
    print("STEP 2: LORA TRAINING")
    print("="*80)
    
    trainer_obj = RailwayLoRATrainer(use_qlora=True)
    model, tokenizer = trainer_obj.setup_model()
    
    # Uncomment to train (requires GPU)
    # trainer_obj.train(train_data, output_dir="railway_lora_model")
    
    print("\n" + "="*80)
    print("✅ TRAINING SETUP COMPLETE")
    print("="*80)
    print("\nTo start training:")
    print("1. Ensure GPU is available (CUDA)")
    print("2. Uncomment trainer_obj.train() line above")
    print("3. Run: python train_customer_care_lora.py")
    print("\nExpected training time: 2-4 hours on single GPU")
    print("Output: railway_lora_model/ directory with fine-tuned weights")
    
    # Step 3: Demo inference (if model exists)
    if os.path.exists("railway_lora_model"):
        print("\n" + "="*80)
        print("STEP 3: INFERENCE DEMO")
        print("="*80)
        
        bot = RailwayCustomerCareBot()
        bot.load_model()
        
        # Demo queries
        test_queries = [
            "Will morning rain affect my 8 AM train?",
            "There's fog here. When will my train arrive?",
            "My train is showing 60 minutes delay. What happened?",
            "Is evening train on time during storm?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"User: {query}")
            print(f"{'='*80}")
            response = bot.chat(query)
            print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()
