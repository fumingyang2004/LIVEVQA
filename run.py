import os
import sys
import argparse
from datetime import datetime
from collectors.config import DATA_DIR
from collectors.utils_date import get_current_timestamp
from collectors.utils_display import print_header, print_error, print_success
from collectors.collector_manager import TopicCollectorManager

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Hot Topic Collection Tool")
    
    # Basic parameters
    parser.add_argument("-o", "--output", type=str,
                        help="Specify the output file path")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="Show detailed output")
    parser.add_argument("-q", "--quiet", action="store_true", 
                        help="Reduce output verbosity")
    
    # Advanced parameters
    advanced = parser.add_argument_group('Advanced Options')
    advanced.add_argument("--no-color", action="store_true",
                          help="Disable colored output")
    advanced.add_argument("--debug", action="store_true",
                          help="Enable debug mode")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Set environment variables
    if args.no_color:
        os.environ['NO_COLOR'] = '1'
    
    # Display welcome message
    print_header("Hot Topic Collection Tool")
    
    try:
        # Create collector
        collector = TopicCollectorManager()
        collector.setup(verbose=args.verbose, quiet=args.quiet)
        
        # Determine output file
        if args.output:
            output_file = args.output
        else:
            # Always use a fixed filename, no timestamp
            output_file = os.path.join(DATA_DIR, "hot_topics.json")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Run collection
        topics = collector.collect_all_topics()
        
        # Save results
        collector.save_to_file(output_file, topics)
        
        # Log results
        collector.log_collection_result(output_file, datetime.now())
        
        print_success(f"Collection complete! Data saved to {output_file}")
        return 0
    
    except KeyboardInterrupt:
        print_error("Operation interrupted by user")
        return 1
        
    except Exception as e:
        print_error(f"An error occurred during execution: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())