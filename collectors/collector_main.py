"""Main entry point for hot topic collection"""
import os
from datetime import datetime
from collectors.config import WORKSPACE, DATA_DIR
from collectors.collector_manager import TopicCollectorManager
from collectors.utils_date import get_mmddhhmi_timestamp

# For backward compatibility, import TopicCollectorManager as HotTopicsCollector
# This allows old code to still import HotTopicsCollector from collector_main
HotTopicsCollector = TopicCollectorManager

def run_topic_collection():
    """
    Runs the hot topic collection process
    """
    # Use a timestamped filename - format MMDDHHMI (MonthDayHourMinute)
    timestamp = get_mmddhhmi_timestamp()
    output_file = os.path.join(DATA_DIR, f"hot_topics_{timestamp}.json")

    # Create and configure the collector manager
    collector = TopicCollectorManager()
    collector.setup()

    print("Starting hot topic collection...")
    start_time = datetime.now()

    # Run the collection process
    topics = collector.collect_all_topics()

    # Save results
    collector.save_to_file(output_file, topics)
    collector.log_collection_result(output_file, start_time)

    print(f"Collection complete! Data saved to {output_file}")
    return output_file

if __name__ == "__main__":
    run_topic_collection()
