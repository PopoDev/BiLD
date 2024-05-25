import re
import matplotlib.pyplot as plt
from collections import Counter

# Function to extract numbers from "Num FB:" lines
def extract_num_fb(log_content):
    pattern = r"Num FB: (\d+)"
    numbers = re.findall(pattern, log_content)
    return list(map(int, numbers))

# Function to read the log file
def read_log_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to plot the bar graph of frequencies
def plot_frequency_bar_graph(numbers):
    frequency = Counter(numbers)
    keys = list(frequency.keys())
    values = list(frequency.values())
    
    plt.bar(keys, values)
    plt.xlabel('Fallback Value')
    plt.ylabel('Frequency of Samples')
    plt.title('Frequency of FB values (XSUM)')
    plt.xticks(keys)  # Ensure all keys are shown on x-axis
    plt.savefig("bargraph")

# Main function
def main():
    log_file_path = '/local1/hfs/CSE481N_Project/test.log'
    log_content = read_log_file(log_file_path)
    num_fb_values = extract_num_fb(log_content)
    plot_frequency_bar_graph(num_fb_values)

if __name__ == "__main__":
    main()