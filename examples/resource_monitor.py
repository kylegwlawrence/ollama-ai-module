"""Example demonstrating how to use the ResourceMonitor class.

This script monitors CPU and memory usage of a system process
using the ResourceMonitor class.
"""
import sys
import threading
import time
from pathlib import Path

# Add parent directory to path so we can import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utilities import ResourceMonitor

if __name__ == "__main__":
    # Monitor the init process (PID 1) which always runs on Linux
    pid = 1

    monitor = ResourceMonitor(pid, interval=0.5)

    # Start monitoring in a background thread
    monitor_thread = threading.Thread(target=monitor.monitor, daemon=True)
    monitor_thread.start()

    # Let it collect samples for 5 seconds
    print(f"Monitoring process {pid} for 5 seconds...")
    time.sleep(5)

    # Stop monitoring
    monitor.stop()
    monitor_thread.join()

    # Display the statistics
    stats = monitor.get_statistics()
    print("\nResource Statistics:")
    print(f"  CPU Average: {stats['cpu_avg_percent']}%")
    print(f"  CPU Peak: {stats['cpu_peak_percent']}%")
    print(f"  Memory Average: {stats['memory_avg_mb']} MB")
    print(f"  Memory Peak: {stats['memory_peak_mb']} MB")
