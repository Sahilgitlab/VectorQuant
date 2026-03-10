"""
Experiment Tracking & Lifecycle
"""
import json
import os
from datetime import datetime

class ExperimentTracker:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file = os.path.join(log_dir, "experiment_log.json")
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump([], f)
                
    def log_experiment(self, name, parameters, result_metrics):
        """
        Appends an experiment record to the JSON log.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": name,
            "parameters": parameters,
            "results": result_metrics
        }
        
        with open(self.log_file, "r") as f:
            data = json.load(f)
            
        data.append(record)
        
        with open(self.log_file, "w") as f:
            json.dump(data, f, indent=4)
            
        return len(data)

def display_leaderboard(log_dir="logs", sort_by="sharpe"):
    """
    Returns experiments sorted by a specific metric.
    """
    log_file = os.path.join(log_dir, "experiment_log.json")
    if not os.path.exists(log_file):
        return []
        
    with open(log_file, "r") as f:
        data = json.load(f)
        
    data.sort(key=lambda x: x["results"].get(sort_by, float('-inf')), reverse=True)
    return data
