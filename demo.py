"""
Railway Customer Care System

"""

import sys
import os


sys.path.append(os.path.dirname(__file__))

from railway_graph_rag_system import GraphRAGBot
import pandas as pd
from datetime import datetime


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def demo_basic_system():
    """Demo the basic Graph RAG system"""
    print_header("🚂 RAILWAY CUSTOMER CARE DEMO")
    

    bot = GraphRAGBot('data.csv')
    bot.initialize()
    
   
    print_header("💬 DEMO 1: Morning Fog Query")
    print("\n🧑 USER: There's fog here in the morning. Will my 7 AM train be delayed?")
    response = bot.respond(
        'delay',
        conditions={
            'is_fog': True,
            'weather': 'Fog',
            'hour': 7,
            'previous_delay': 15,
            'track_maintenance': False,
            'signal_failure': False
        },
        train='Morning Express #12345'
    )
    
    print_header("💬 DEMO 2: Evening Rain Query")
    print("\n🧑 USER: It's raining heavily. My 6 PM train status?")
    response = bot.respond(
        'delay',
        conditions={
            'weather': 'Rain',
            'hour': 18,
            'previous_delay': 25,
            'track_maintenance': False,
            'signal_failure': True
        },
        train='Evening Commuter #54321'
    )
    
    print_header("💬 DEMO 3: Storm Warning Query")
    print("\n🧑 USER: Storm warning issued. Is my train safe?")
    response = bot.respond(
        'delay',
        conditions={
            'weather': 'Storm',
            'hour': 14,
            'previous_delay': 40,
            'track_maintenance': False,
            'signal_failure': False
        },
        train='Afternoon Express #98765'
    )
    
    print_header("💬 DEMO 4: Best Travel Time Query")
    print("\n🧑 USER: What's the best time for morning travel?")
    response = bot.respond('morning')
    
    print_header("💬 DEMO 5: Fog Status Check")
    print("\n🧑 USER: How is fog affecting trains right now?")
    response = bot.respond('fog', hour=8)
    
    print_header("💬 DEMO 6: Clear Weather - On Time")
    print("\n🧑 USER: My train is at 10 AM. Any delays?")
    response = bot.respond(
        'delay',
        conditions={
            'weather': 'Clear',
            'hour': 10,
            'previous_delay': 2,
            'track_maintenance': False,
            'signal_failure': False
        },
        train='Mid-Morning Service #11111'
    )
    
    # Show some analytics
    print_header("📊 NETWORK STATISTICS")
    
    total_trains = len(bot.df)
    avg_delay = bot.df['Arrival_Delay_min'].mean()
    ontime_pct = (bot.df['Arrival_Delay_min'] <= 5).sum() / len(bot.df) * 100
    fog_penalty = bot.knowledge['fog']['penalty']
    
    print(f"""
    📈 Total Trains Analyzed: {total_trains:,}
    ⏱️  Average Delay: {avg_delay:.1f} minutes
    ✅ On-Time Performance: {ontime_pct:.1f}%
    🌫️  Fog Impact: +{fog_penalty:.1f} minutes
    
    🔍 Best Departure Hour: {bot.knowledge['time']['best']:02d}:00
    ⚠️  Worst Departure Hour: {bot.knowledge['time']['worst']:02d}:00
    🔄 Cascade Correlation: {bot.knowledge['cascade']:.3f}
    """)
    
    # Show delay categories
    print_header("📊 DELAY CATEGORIES")
    
    delay_cats = pd.cut(
        bot.df['Arrival_Delay_min'],
        bins=[-1, 5, 15, 30, 60, 1000],
        labels=['On-Time (≤5min)', 'Minor (6-15min)', 'Moderate (16-30min)', 
                'Major (31-60min)', 'Severe (>60min)']
    ).value_counts()
    
    for cat, count in delay_cats.items():
        pct = count / len(bot.df) * 100
        bar = '█' * int(pct / 2)
        print(f"    {cat:20s} | {bar:50s} {pct:5.1f}% ({count:,})")
    
    # Weather analysis
    print_header("🌤️ WEATHER CONDITIONS IMPACT")
    
    weather_impact = bot.df.groupby('WeatherCondition')['Arrival_Delay_min'].agg(['mean', 'count'])
    weather_impact = weather_impact[weather_impact['count'] > 100].sort_values('mean', ascending=False).head(8)
    
    for weather, row in weather_impact.iterrows():
        avg = row['mean']
        count = int(row['count'])
        bar = '█' * int(avg / 2)
        print(f"    {weather:15s} | {bar:40s} {avg:6.1f} min (n={count:,})")
    
    


def demo_response_variations():
    """Show how responses vary by context"""
    print_header("🎭 RESPONSE VARIATION DEMO")
    print("\nSame delay amount (20 min) but different contexts:\n")
    
    bot = GraphRAGBot('data.csv')
    bot.initialize()
    
    contexts = [
        {
            'name': 'Morning Fog',
            'conditions': {'weather': 'Fog', 'hour': 7, 'previous_delay': 10, 'is_fog': True}
        },
        {
            'name': 'Evening Rain',
            'conditions': {'weather': 'Rain', 'hour': 18, 'previous_delay': 10, 'is_fog': False}
        },
        {
            'name': 'Midday Clear',
            'conditions': {'weather': 'Clear', 'hour': 12, 'previous_delay': 10, 'is_fog': False}
        }
    ]
    
    for ctx in contexts:
        print(f"\n{'─'*80}")
        print(f"Context: {ctx['name']}")
        print(f"{'─'*80}")
        
       
        ctx['conditions']['track_maintenance'] = False
        ctx['conditions']['signal_failure'] = False
        
        response = bot.respond('delay', conditions=ctx['conditions'], train='Test Train')
        print()  # Response already printed by bot.respond


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Railway Customer Care Demo')
    parser.add_argument('--variations', action='store_true', 
                       help='Show response variation demo')
    args = parser.parse_args()
    
    if args.variations:
        demo_response_variations()
    else:
        demo_basic_system()
