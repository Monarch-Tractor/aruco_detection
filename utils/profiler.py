"""
This script contains the implementation of the
Profiler class which is used to profile the code.
"""


import time
import json

class Profiler:
    """
    This class defines the Profiler class which is used
    to profile the code.
    """
    def __init__(self):
        self.start_timings = {}
        self.total_timings = {}
        self.counts = {}
        self.start_times = {}
        self.averages = {"timings": {}}

    def start(self, component_name):
        """
        This method starts timing for a given component.
        """
        if component_name not in self.total_timings:
            self.total_timings[component_name] = 0.0
            self.counts[component_name] = 0
            self.averages["timings"][component_name] = 0.0
        self.start_timings[component_name] = time.time()
        return

    def stop(self, component_name):
        """
        This method stops timing for a given component and 
        record the elapsed time.
        """
        if component_name in self.start_timings:
            start_time = self.start_timings.pop(component_name)
            end_time = time.time()
            duration = end_time - start_time
            self.total_timings[component_name] += duration
            self.counts[component_name] += 1
            self.averages["timings"][component_name] = (
                self.total_timings[component_name] / \
                self.counts[component_name]
            ) * 1000  # Unit as milli seconds.
        else:
            raise ValueError(
                f"Component '{component_name}' was not started."
            )
        return

    def get_averages(self):
        """
        This method calculates and returns the average 
        times for each component.
        """
        return self.averages

    def save_stats(self, filename):
        """
        This method saves the timing statistics to a file.
        """
        with open(filename, 'w') as f:
            json.dump(self.total_timings, f)
        return

    def load_stats(self, filename):
        """
        This load the timing statistics from a file.
        """
        with open(filename, "r") as f:
            self.total_timings = json.load(f)
        return
