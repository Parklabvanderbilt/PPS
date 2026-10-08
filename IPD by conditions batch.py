import re
import numpy as np
import statistics
import os
import csv
from datetime import datetime

folder_path = '/Users/hagrid/Desktop/ML/PPS/PPS3 raw data'

def extract(file):
    with open(file, 'r', errors='ignore') as file:
        content = file.read()

    numbers = re.findall(r'\d+\.\d*', content)
    numbers = [float(num) for num in numbers]

    return numbers


def get_stats(file):
    # If this trial file doesn't exist, return NaN
    if not os.path.exists(file):
        return np.nan, np.nan

    times = extract(file)

    # If the file exists but contains no usable values
    if len(times) == 0:
        return np.nan, np.nan

    mean = np.average(times)

    # stdev requires at least 2 values
    if len(times) >= 2:
        std = statistics.stdev(times)
    else:
        std = np.nan

    return mean, std

results = []

# Find all participants who have ANY of the four trial files
participants = set()

for filename in os.listdir(folder_path):

    if filename.endswith(('AF.txt', 'AT.txt', 'PF.txt', 'PT.txt')):
        # Remove AF.txt, AT.txt, PF.txt, or PT.txt
        base = filename[:-6]
        participants.add(base)


for participant in sorted(participants):

    participant_id = participant.split('.')[0]

    print(participant)

    # Construct the four possible filenames
    af_file = os.path.join(folder_path, participant + 'AF.txt')
    at_file = os.path.join(folder_path, participant + 'AT.txt')
    pf_file = os.path.join(folder_path, participant + 'PF.txt')
    pt_file = os.path.join(folder_path, participant + 'PT.txt')

    # Calculate stats; missing files become NaN
    mean_af, std_af = get_stats(af_file)
    mean_at, std_at = get_stats(at_file)
    mean_pf, std_pf = get_stats(pf_file)
    mean_pt, std_pt = get_stats(pt_file)

    results.append([
        participant_id,
        participant,
        mean_af,
        std_af,
        mean_at,
        std_at,
        mean_pf,
        std_pf,
        mean_pt,
        std_pt
    ])

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d %H-%M")
csv_file_path = f'/Users/hagrid/Desktop/ML/PPS/PPS3 results {timestamp}.csv'

with open(csv_file_path, mode='w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow([
        'participant_id',
        'Filename',
        'mean_af',
        'sd_af',
        'mean_at',
        'sd_at',
        'mean_pf',
        'sd_pf',
        'mean_pt',
        'sd_pt'
    ])

    writer.writerows(results)