"""Command line display tools module - provides unified command line output formatting"""
import os
import time
import sys
from datetime import datetime

# Color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def terminal_supports_color():
    """Checks if the terminal supports colored output"""
    # Windows environment check
    if os.name == 'nt':
        return False
        
    # Check if redirected to a file
    if not sys.stdout.isatty():
        return False
        
    # Check NO_COLOR environment variable
    if os.environ.get('NO_COLOR'):
        return False
        
    return True

# Set color usage based on terminal color support
USE_COLORS = terminal_supports_color()

def colorize(text, color):
    """Adds color to text based on terminal support"""
    if not USE_COLORS:
        return text
    return f"{color}{text}{Colors.ENDC}"

def print_header(message):
    """Prints a header with a unified style"""
    header = colorize(f"{'='*80}", Colors.BLUE)
    print(f"\n{header}")
    print(colorize(f" {message}", Colors.BOLD + Colors.CYAN))
    print(f"{header}\n")

def print_subheader(message):
    """Prints a subheader with a unified style"""
    print(colorize(f"\n{'-'*40}", Colors.BLUE))
    print(colorize(f" {message}", Colors.CYAN))
    print(colorize(f"{'-'*40}", Colors.BLUE))

def print_success(message):
    """Prints a success message"""
    print(colorize(f"✓ {message}", Colors.GREEN))

def print_warning(message):
    """Prints a warning message"""
    print(colorize(f"⚠ {message}", Colors.YELLOW))

def print_error(message):
    """Prints an error message"""
    print(colorize(f"✗ {message}", Colors.RED))

def print_info(message):
    """Prints an information message"""
    print(colorize(f"ℹ {message}", Colors.BLUE))

def print_progress(current, total, prefix='', suffix='', length=50):
    """Prints a progress bar
    
    Args:
        current: Current progress
        total: Total number
        prefix: Prefix text
        suffix: Suffix text
        length: Length of the progress bar
    """
    if not sys.stdout.isatty():  # If output is redirected, do not display progress bar
        return
        
    percent = float(current) / float(total)
    filled_length = int(length * percent)
    bar = '█' * filled_length + '░' * (length - filled_length)
    
    # Calculate percentage
    percent_str = f"{100 * percent:.1f}%"
    
    # Construct formatted progress bar
    if USE_COLORS:
        progress_bar = f"{Colors.BLUE}{prefix}{Colors.ENDC} |{Colors.GREEN}{bar}{Colors.ENDC}| {Colors.CYAN}{percent_str}{Colors.ENDC} {suffix}"
    else:
        progress_bar = f"{prefix} |{bar}| {percent_str} {suffix}"
    
    # Clear current line and print
    sys.stdout.write('\r' + progress_bar)
    sys.stdout.flush()
    
    if current == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

def print_collection_summary(topics, title="Collection Summary", show_categories=True, show_sources=True):
    """Prints a summary of topic collection results
    
    Args:
        topics: List of collected topics
        title: Summary title
        show_categories: Whether to show category statistics
        show_sources: Whether to show source statistics
    """
    if not topics:
        print_warning("No topics collected")
        return
    
    print_subheader(title)
    
    # Total count
    print(f"Total topics collected: {len(topics)} articles")
    
    # Count articles with images
    image_count = sum(1 for t in topics if t.get('has_image', False))
    print(f"Articles with images: {image_count} articles ({image_count/len(topics)*100:.1f}%)")
    
    # Display source statistics
    if show_sources:
        print_subheader("Source Statistics")
        sources = {}
        for t in topics:
            url = t.get('url', '')
            if 'cnn.com' in url:
                sources['CNN'] = sources.get('CNN', 0) + 1
            elif 'bbc.' in url:
                sources['BBC'] = sources.get('BBC', 0) + 1
            elif 'variety.com' in url:
                sources['Variety'] = sources.get('Variety', 0) + 1
            elif 'forbes.com' in url:
                sources['Forbes'] = sources.get('Forbes', 0) + 1
            elif 'apnews.com' in url:
                sources['AP News'] = sources.get('AP News', 0) + 1
            elif 'yahoo.com' in url:
                sources['Yahoo'] = sources.get('Yahoo', 0) + 1
            else:
                sources['Other'] = sources.get('Other', 0) + 1
                
        # Sort by article count
        sorted_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)
        for source, count in sorted_sources:
            percent = count / len(topics) * 100
            print(f"   {source}: {count} articles ({percent:.1f}%)")
    
    # Display category statistics
    if show_categories:
        print_subheader("Category Statistics")
        categories = {}
        for t in topics:
            cat = t.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
        # Sort by article count
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        for cat, count in sorted_categories:
            percent = count / len(topics) * 100
            print(f"   {cat}: {count} articles ({percent:.1f}%)")

def print_time_stats(start_time):
    """Prints time statistics"""
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print_subheader("Time Statistics")
    print(f"Start Time: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}")
    print(f"End Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total Run Time: {int(minutes)}m {int(seconds)}s")

def create_spinner():
    """Creates a simple loading animation generator"""
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    i = 0
    while True:
        yield spinner_chars[i]
        i = (i + 1) % len(spinner_chars)

class Spinner:
    """Loading animation class"""
    def __init__(self, message="Processing"):
        self.message = message
        self.spinner = create_spinner()
        self.running = False
        self.thread = None
    
    def start(self):
        """Starts the loading animation"""
        if not sys.stdout.isatty():  # If output is redirected, do not display animation
            return self
            
        self.running = True
        
        def spin():
            while self.running:
                if USE_COLORS:
                    sys.stdout.write(f"\r{Colors.BLUE}{next(self.spinner)}{Colors.ENDC} {self.message}")
                else:
                    sys.stdout.write(f"\r{next(self.spinner)} {self.message}")
                sys.stdout.flush()
                time.sleep(0.1)
                
        import threading
        self.thread = threading.Thread(target=spin)
        self.thread.daemon = True
        self.thread.start()
        return self
    
    def stop(self, message=None):
        """Stops the loading animation"""
        self.running = False
        if self.thread:
            self.thread.join(0.1)
        
        if sys.stdout.isatty():  # Only clear current line in terminal
            sys.stdout.write("\r" + " " * (len(self.message) + 10))  # Clear current line
            if message:
                if USE_COLORS:
                    sys.stdout.write(f"\r{Colors.GREEN}✓{Colors.ENDC} {message}\n")
                else:
                    sys.stdout.write(f"\r✓ {message}\n")
            else:
                sys.stdout.write("\r")
            sys.stdout.flush()
