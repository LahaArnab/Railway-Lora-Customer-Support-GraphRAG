"""

Complete Railway Graph RAG System

Installation:
pip install pandas numpy networkx matplotlib seaborn scikit-learn transformers torch accelerate


"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import warnings
import json
warnings.filterwarnings('ignore')


try:
    from transformers import pipeline
    LLM_AVAILABLE = True
except:
    LLM_AVAILABLE = False


class RailwayNetworkGraph:
    """Graph structure for railway network"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.stations = {}
        self.trains = {}
        
    def build(self, df):
        """Build network graph from data"""
        print("\n🔗 Building Railway Network Graph...")
        
        # Create stations
        n_stations = 50
        for i in range(n_stations):
            sid = f"STN_{i:03d}"
            station_trains = df[df.index % n_stations == i]
            self.graph.add_node(sid, type='station',
                               avg_delay=station_trains['Arrival_Delay_min'].mean() if len(station_trains)>0 else 0)
            self.stations[sid] = len(station_trains)
        
        
        for idx, row in df.iterrows():
            tid = f"TRAIN_{idx}"
            self.graph.add_node(tid, type='train',
                               delay=row['Arrival_Delay_min'],
                               fog=1 if row['WeatherCondition']=='Fog' else 0,
                               hour=pd.to_datetime(row['Scheduled_Departure_Time'], format='%H:%M').hour)
            self.trains[tid] = idx
            
            
            if idx > 0 and row['Previous_Train_Delay_min'] > 0:
                self.graph.add_edge(f"TRAIN_{idx-1}", tid, 
                                   delay_transfer=row['Previous_Train_Delay_min'])
        
        print(f"✅ Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph
    
    def find_similar_cases(self, target_delay, prev_delay, is_fog):
        """RAG: Retrieve similar historical cases"""
        similar = []
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'train':
                n_delay = self.graph.nodes[node].get('delay', 0)
                n_fog = self.graph.nodes[node].get('fog', 0)
                
                if abs(n_delay - target_delay) < 20 and n_fog == is_fog:
                    similar.append({'node': node, 'delay': n_delay})
        
        return similar[:10]
    
    def predict_cascade(self, train_id, hops=3):
        """Predict multi-hop cascading impact"""
        if not self.graph.has_node(train_id):
            return []
        
        affected = []
        queue = [(train_id, 0, self.graph.nodes[train_id].get('delay', 0))]
        visited = set()
        
        while queue and len(affected) < hops:
            curr, depth, cum_delay = queue.pop(0)
            if depth >= hops: continue
            visited.add(curr)
            
            for neighbor in self.graph.successors(curr):
                if neighbor not in visited:
                    transfer = self.graph.edges[curr, neighbor].get('delay_transfer', 0)
                    new_cum = cum_delay * (0.7 ** (depth+1))
                    affected.append({'train': neighbor, 'hops': depth+1, 'delay': new_cum})
                    queue.append((neighbor, depth+1, new_cum))
        
        return affected


class DelayAnalyzer:
    """Analyze delay patterns"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.stats = {}
        self._prepare()
    
    def _prepare(self):
        """Prepare data"""
        self.df['WeatherCondition'] = self.df['WeatherCondition'].fillna('Clear')
        self.df['Departure_Hour'] = pd.to_datetime(self.df['Scheduled_Departure_Time'], format='%H:%M').dt.hour
        self.df['Is_Fog'] = (self.df['WeatherCondition'] == 'Fog').astype(int)
        self.df['Is_Morning'] = (self.df['Departure_Hour'].between(6, 11)).astype(int)
    
    def analyze_fog(self):
        """Fog impact analysis"""
        print("\n🌫️ Fog Impact Analysis...")
        fog = self.df[self.df['Is_Fog'] == 1]
        clear = self.df[self.df['Is_Fog'] == 0]
        
        fog_avg = fog['Arrival_Delay_min'].mean()
        clear_avg = clear['Arrival_Delay_min'].mean()
        penalty = fog_avg - clear_avg
        
        print(f"   Fog: {fog_avg:.1f}min | Clear: {clear_avg:.1f}min | Penalty: +{penalty:.1f}min")
        
        self.stats['fog'] = {'penalty': penalty, 'fog_avg': fog_avg, 'clear_avg': clear_avg}
        return self.stats['fog']
    
    def analyze_time(self):
        """Time pattern analysis"""
        print("\n⏰ Time Pattern Analysis...")
        hourly = self.df.groupby('Departure_Hour')['Arrival_Delay_min'].mean()
        best = hourly.idxmin()
        worst = hourly.idxmax()
        
        print(f"   Best hour: {best:02d}:00 ({hourly[best]:.1f}min)")
        print(f"   Worst hour: {worst:02d}:00 ({hourly[worst]:.1f}min)")
        
        self.stats['time'] = {'best': best, 'worst': worst, 'hourly': hourly.to_dict()}
        return self.stats['time']
    
    def analyze_cascade(self):
        """Cascade analysis"""
        print("\n🔄 Cascade Analysis...")
        corr = self.df['Previous_Train_Delay_min'].corr(self.df['Arrival_Delay_min'])
        print(f"   Correlation: {corr:.4f}")
        
        self.stats['cascade'] = corr
        return corr


class GraphRAGBot:
    """Main chatbot with Graph RAG"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.graph = RailwayNetworkGraph()
        self.analyzer = None
        self.knowledge = {}
        self.llm = None
        self._init_llm()
    
    def _init_llm(self):
        """Initialize small LLM (optional)"""
        if LLM_AVAILABLE:
            try:
                # Use a small, fast model
                self.llm = pipeline("text-generation", model="gpt2", max_new_tokens=100)
                print("✅ LLM loaded (GPT-2 for responses)")
            except:
                print("⚠️ LLM unavailable, using rule-based responses")
        else:
            print("⚠️ LLM unavailable, using rule-based responses")
    
    def initialize(self):
        """Initialize system"""
        print("\n" + "="*80)
        print("🚂 INDIAN RAILWAYS GRAPH RAG SYSTEM")
        print("   Graph Network | Cascade Detection | Fog Analysis")
        print("="*80)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"\n✅ Loaded {len(self.df):,} records")
        
        
        self.analyzer = DelayAnalyzer(self.df)
        self.graph.build(self.analyzer.df)
        
        
        fog = self.analyzer.analyze_fog()
        time = self.analyzer.analyze_time()
        cascade = self.analyzer.analyze_cascade()
        
        
        self.knowledge = {
            'fog': fog,
            'time': time,
            'cascade': cascade,
            'avg_delay': self.df['Arrival_Delay_min'].mean()
        }
        
        print("\n✅ System Ready!")
    
    def predict_delay(self, conditions):
        """Predict delay using Graph RAG"""
        
        base = 10
        
        # Fog
        if conditions.get('is_fog'):
            base += self.knowledge['fog']['penalty']
        
        # Weather
        weather_map = {'Clear': 0, 'Rain': 20, 'Fog': 45, 'Storm': 35}
        base += weather_map.get(conditions.get('weather', 'Clear'), 0)
        
        # Cascade
        prev = conditions.get('previous_delay', 0)
        base += prev * self.knowledge['cascade'] * 0.5
        
        # Time
        hour = conditions.get('hour', 12)
        if hour in [7, 8, 18, 19]:
            base *= 1.15
        
        # Operations
        if conditions.get('track_maintenance'): base += 30
        if conditions.get('signal_failure'): base += 25
        
        # RAG: Adjust with similar cases
        similar = self.graph.find_similar_cases(base, prev, conditions.get('is_fog', False))
        if similar:
            hist_delays = [c['delay'] for c in similar]
            base = np.mean([base] + hist_delays)
        
        # Scenarios
        scenarios = [
            (0.70, base, "network analysis"),
            (0.20, base * 0.65, "if improved"),
            (0.10, base * 1.35, "if worsens")
        ]
        
        return scenarios, base, similar
    
    def respond(self, query_type, **kwargs):
        """Generate context-aware response using RAG"""
        print("\n" + "="*80)
        print("💬 RESPONSE")
        print("="*80)
        
        if query_type == "delay":
            cond = kwargs.get('conditions', {})
            train = kwargs.get('train', 'your train')
            
            scenarios, base, similar = self.predict_delay(cond)
            p1, d1 = int(scenarios[0][0]*100), int(scenarios[0][1])
            p2, d2 = int(scenarios[1][0]*100), int(scenarios[1][1])
            
            # Extract context
            hour = cond.get('hour', 12)
            is_morning = 6 <= hour <= 11
            is_evening = 17 <= hour <= 21
            weather = cond.get('weather', 'Clear')
            is_fog = cond.get('is_fog', False)
            prev_delay = cond.get('previous_delay', 0)
            
            # Build context-aware reasons
            reasons = []
            weather_context = ""
            
            if weather == 'Rain':
                if is_morning:
                    weather_context = "morning rain showers are affecting track conditions"
                    reasons.append("reduced visibility and wet tracks")
                elif is_evening:
                    weather_context = "evening rainfall is impacting operations"
                    reasons.append("heavy evening downpour")
                else:
                    weather_context = "rainfall in the region"
                    reasons.append("wet track conditions")
            elif is_fog:
                if is_morning:
                    weather_context = "dense morning fog is reducing visibility"
                    reasons.append("low visibility in morning fog")
                else:
                    weather_context = "fog conditions across the route"
                    reasons.append("foggy weather")
            elif weather == 'Storm':
                weather_context = "storm conditions requiring safety protocols"
                reasons.append("severe weather safety measures")
            
            if prev_delay > 30:
                reasons.append("cascading delays from previous services")
            elif prev_delay > 15:
                reasons.append("minor upstream delays")
            
            if cond.get('track_maintenance'):
                reasons.append("scheduled track maintenance")
            if cond.get('signal_failure'):
                reasons.append("signal system issues")
            
            reason = ", ".join(reasons) if reasons else "current network conditions"
            
            # Generate varied responses based on context
            if d1 <= 10:
                greetings = [
                    f"Great news! {train} is running smoothly",
                    f"You're in luck! {train} is on schedule",
                    f"Excellent! {train} is maintaining its timetable"
                ]
                import random
                msg = f"{random.choice(greetings)}, expected within {d1} minutes."
                if similar:
                    msg += f" Based on {len(similar)} similar trips, conditions are favorable today."
            else:
                # Context-aware delay messaging
                if weather_context:
                    intro = f"Due to {weather_context}, "
                elif is_morning:
                    intro = f"For this morning departure, "
                elif is_evening:
                    intro = f"During evening peak hours, "
                else:
                    intro = ""
                
                msg = f"{intro}{train} is experiencing delays. "
                msg += f"Current estimate: {d1} minutes (probability: {p1}%). "
                
                if p2 < 100:
                    msg += f"Alternative scenario: {d2} minutes ({p2}% chance) if conditions worsen."
                
                if similar:
                    avg_similar = np.mean([s['delay'] for s in similar])
                    msg += f"\n\n📊 RAG Analysis: Found {len(similar)} similar historical cases with average delay of {avg_similar:.0f} minutes."
                
                # Time-specific advice
                if is_morning and d1 > 20:
                    msg += "\n\n☕ Morning rush - consider the next service if you have flexibility."
                elif is_evening and d1 > 20:
                    msg += "\n\n🌆 Evening peak detected - platform may be crowded, please plan accordingly."
            
            # Enhanced contingency
            if d1 >= 240:
                msg += "\n\n🚨 MAJOR DELAY PROTOCOL:\n   • Complimentary meals being arranged\n   • Full/partial refunds available\n   • SMS updates every 30 minutes\n   • Customer care: 139"
            elif d1 >= 120:
                msg += "\n\n⚠️ DELAY ASSISTANCE:\n   • Refreshments available at platform\n   • Live updates via SMS\n   • Alternative routes being evaluated"
            elif d1 >= 60:
                msg += "\n\n📱 Stay updated: SMS alerts active for your PNR"
            
            print(f"\n{msg}\n")
            return msg
        
        elif query_type == "fog":
            penalty = self.knowledge['fog']['penalty']
            hour = kwargs.get('hour', datetime.now().hour)
            is_morning = 6 <= hour <= 11
            
            if is_morning:
                if penalty < 20:
                    msg = "Morning fog is present but visibility is improving. Minimal impact expected (~{:.0f} min delays). Early trains may face more delays.".format(penalty)
                elif penalty < 40:
                    msg = f"Dense morning fog is affecting operations with ~{penalty:.0f} min delays. Visibility typically improves after 10 AM. Consider later trains if possible."
                else:
                    msg = f"Severe morning fog causing significant delays (~{penalty:.0f} min). Safety is our priority. Delays expected until fog clears around noon."
            else:
                if penalty < 20:
                    msg = "Fog conditions detected with minimal schedule impact."
                elif penalty < 40:
                    msg = f"Moderate fog causing ~{penalty:.0f} min delays. Operations teams actively monitoring."
                else:
                    msg = f"Heavy fog impacting network (~{penalty:.0f} min delays). Enhanced safety protocols in effect."
            
            print(f"\n{msg}\n")
            return msg
        
        elif query_type == "morning":
            best = self.knowledge['time']['best']
            worst = self.knowledge['time']['worst']
            msg = f"🌅 Morning Travel Recommendations:\n\n"
            msg += f"✅ BEST TIME: Trains around {best:02d}:00 have the best punctuality record.\n"
            msg += f"❌ AVOID: Peak delays typically occur around {worst:02d}:00.\n\n"
            
            # Check fog conditions
            fog_penalty = self.knowledge['fog']['penalty']
            if fog_penalty > 30:
                msg += "⚠️ NOTE: Morning fog is common - factor in extra time for departures before 10 AM."
            
            print(f"\n{msg}\n")
            return msg
        
        elif query_type == "chat":
            # General chat handler
            user_msg = kwargs.get('message', '').lower()
            
            if 'rain' in user_msg:
                return self.respond('weather_rain', **kwargs)
            elif 'fog' in user_msg:
                return self.respond('fog', **kwargs)
            elif 'morning' in user_msg or 'early' in user_msg:
                return self.respond('morning', **kwargs)
            elif 'delay' in user_msg or 'late' in user_msg:
                return self.respond('delay', **kwargs)
            else:
                return "Hello! I'm your Railway Customer Care assistant. I can help with:\n• Train delay predictions\n• Weather impact analysis\n• Best travel times\n• Real-time network status\n\nHow may I assist you today?"
        
        elif query_type == "weather_rain":
            hour = kwargs.get('hour', datetime.now().hour)
            is_morning = 6 <= hour <= 11
            
            if is_morning:
                msg = "🌧️ Morning rain typically causes 15-25 minute delays due to:"
                msg += "\n• Reduced track adhesion during morning rush"
                msg += "\n• Increased braking distances"
                msg += "\n• Cautious speed restrictions"
                msg += "\n\n💡 Tip: Delays usually improve by late morning as rain subsides."
            else:
                msg = "🌧️ Rainfall is affecting operations. Typical impact: 10-20 minute delays."
                msg += "\n\n🛡️ Safety protocols active. Real-time updates available via SMS."
            
            return msg
        
        return "How may I help you today? Ask about delays, weather, or travel recommendations!"
    
    def run_4step(self):
        """4-step operational system"""
        print("\n" + "="*80)
        print("🔄 4-STEP OPERATIONAL SYSTEM")
        print("="*80)
        
        # Step 1: Identify late trains
        late = self.analyzer.df[self.analyzer.df['Arrival_Delay_min'] > 15].copy()
        late['Priority'] = (late['Arrival_Delay_min'] * 0.3 + 
                           late['Passenger_Load_pct'] * 0.25 +
                           late['Previous_Train_Delay_min'] * 0.25)
        late = late.sort_values('Priority', ascending=False)
        
        print(f"\n📊 STEP 1: {len(late):,} late trains identified")
        
        # Step 2: Prioritize
        print(f"\n🔥 STEP 2: TOP 5 PRIORITY")
        for i, (idx, row) in enumerate(late.head(5).iterrows(), 1):
            impact = self.graph.predict_cascade(f"TRAIN_{idx}", 3)
            print(f"   {i}. {row['Arrival_Delay_min']:.0f}min delay, Priority={row['Priority']:.0f}, Affects {len(impact)} trains")
        
        # Step 3: Update app
        print(f"\n📱 STEP 3: APP UPDATE")
        if len(late) > 0:
            sample = late.iloc[0]
            cond = {'is_fog': bool(sample['Is_Fog']), 
                   'previous_delay': sample['Previous_Train_Delay_min']}
            self.respond('delay', conditions=cond, train=f"Train {late.index[0]}")
        
        # Step 4: Recovery
        print(f"\n🛠️ STEP 4: RECOVERY ACTIONS")
        if len(late) > 0:
            d = late.iloc[0]['Arrival_Delay_min']
            if d > 30: print("   🚉 Platform swap")
            if d > 45: print("   👥 Backup crew")
            if d > 60: print("   🔄 Route optimize")
    
    def visualize(self):
        """Create visualizations"""
        print("\n📊 Generating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Railway Graph RAG Analytics', fontsize=14, fontweight='bold')
        
        # Fog impact
        fog_data = self.analyzer.df.groupby('Is_Fog')['Arrival_Delay_min'].mean()
        axes[0,0].bar(['Clear', 'Fog'], fog_data.values, color=['skyblue', 'gray'])
        axes[0,0].set_title('Fog Impact')
        axes[0,0].set_ylabel('Avg Delay (min)')
        
        # Hourly
        hourly = self.analyzer.df.groupby('Departure_Hour')['Arrival_Delay_min'].mean()
        axes[0,1].plot(hourly.index, hourly.values, 'o-')
        axes[0,1].set_title('Hourly Pattern')
        axes[0,1].set_xlabel('Hour')
        
        # Weather
        weather = self.analyzer.df.groupby('WeatherCondition')['Arrival_Delay_min'].mean().sort_values(ascending=False).head(5)
        axes[1,0].barh(weather.index, weather.values, color='coral')
        axes[1,0].set_title('Weather Impact')
        
        # Cascade
        axes[1,1].scatter(self.analyzer.df['Previous_Train_Delay_min'],
                         self.analyzer.df['Arrival_Delay_min'], alpha=0.1, s=2)
        axes[1,1].set_title('Cascading Effect')
        axes[1,1].set_xlabel('Prev Delay')
        
        plt.tight_layout()
        plt.savefig('railway_analytics.png', dpi=200)
        print("✅ Saved: railway_analytics.png")
        plt.show()


def main():
    """Main execution"""
    bot = GraphRAGBot('data.csv')
    bot.initialize()
    bot.run_4step()
    
    # Demo with varied responses
    print("\n" + "="*80)
    print("💬 DEMO QUERIES - Context-Aware Responses")
    print("="*80)
    
    print("\n🔹 Query 1: 'Will morning rain affect my 8 AM train?'")
    bot.respond('delay', conditions={'weather': 'Rain', 'hour': 8, 'previous_delay': 10}, train='Train #12345')
    
    print("\n🔹 Query 2: 'Fog here in the morning. When will train arrive?'")
    bot.respond('delay', conditions={'is_fog': True, 'hour': 7, 'previous_delay': 20}, train='Morning Express')
    
    print("\n🔹 Query 3: 'What about evening trains during rain?'")
    bot.respond('delay', conditions={'weather': 'Rain', 'hour': 18, 'previous_delay': 5}, train='Evening Commuter')
    
    print("\n🔹 Query 4: 'Best time for morning travel?'")
    bot.respond('morning')
    
    print("\n🔹 Query 5: 'Check fog status'")
    bot.respond('fog', hour=8)
    
    bot.visualize()


if __name__ == "__main__":
    main()

