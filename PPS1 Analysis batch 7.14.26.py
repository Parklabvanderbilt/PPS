import re
import numpy as np
from scipy.optimize import minimize
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime

folder_path = '/Users/hagrid/Desktop/ML/PPS/PPS1 raw data'

min_reaction_time = 0.01
max_reaction_time = 3
min_anticipatory_time = 0.1
max_anticipatory_time = 10

def mean_calculate(extract, condition, distance):
    new_list = extract[extract[:, 3] == condition]
    new_list = new_list[new_list[:, 1] == distance]
    error = np.mean(new_list[:, 2]) + np.std(new_list[:, 2]) * 3

    if error > max_reaction_time:
        error = max_reaction_time

    anticipatory_errors=0
    for i in new_list[:, 2]:
        if i > max_anticipatory_time or i < min_anticipatory_time:
            print("*** Potential Anticipatory Error ***")
            anticipatory_errors += 1

    new_list = new_list[new_list[:, 2] >= min_reaction_time]
    new_list = new_list[new_list[:, 2] <= error]

    if len(new_list) == 0:
        print (f"No valid observations for condition={condition}, distance={distance}")

    return np.average(new_list[:, 2]),anticipatory_errors

def sigmoid(x, xc, b, ymin, ymax):
    return ((ymin + ymax * np.exp((x - xc) / b)) / (1 + np.exp((x - xc) / b)))

# Define the RMSE objective function
def objective(params, ymin, ymax, x, y):
    xc, b = params
    y_pred = sigmoid(x, xc, b, ymin, ymax)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))  # Mean Squared Error
    return rmse

results = [['Subject','Anticipatory errors','Social xc','Social b','Social RMSE', "SocialMean_03", "SocialMean_06", "SocialMean_1", "SocialMean_13", "SocialMean_16", 'Non-social Xc','Non-social b', 'Non-social RMSE',"NonSocialMean_03", "NonSocialMean_06", "NonSocialMean_1", "NonSocialMean_13", "NonSocialMean_16"]]

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a Python file or any other type you need to process
    # Change this based on file type you need
    if filename == '.DS_Store':
        continue
    else:
        file_path = os.path.join(folder_path, filename)
        # Open and process the file
        print(filename)
        with open(file_path, 'r', errors='ignore') as file:
            content = file.read()

        keywords = ['Trial Number:', 'Trial', 'pulse:', 'time:']
        pattern = r"(?:" + "|".join(keywords) + r")\s+([-+]?\d*\.?\d+)"
        extract = np.array([0, 0, 0, 0])
        matches = re.findall(pattern, content)
        bracketed_text = re.findall(r'\[(.*?)\]', content) + re.findall(r'::(.*?)::', content)
        i = 1
        x = 0
        initial_list = []
        for match in matches:
            if i % 3 == 1:
                initial_list.append(int(match))
            elif i % 3 == 2:
                initial_list.append(float(match))
            elif i % 3 == 0:
                initial_list.append(float(match))
                if bracketed_text[x] == 'MALE THROWER' or bracketed_text[x] == 'Spiked Ball':
                    initial_list.append(1)
                else:
                    initial_list.append(2)
                x += 1
                extract = np.vstack((extract, np.array(initial_list)))
                initial_list = []
            i += 1

        try:
            if extract[-1][0] == 24:
                last_index = np.where(extract == 24)[0][3] + 1
            elif extract[-1][0] > 15:
                last_index = len(extract)
            else:
                last_index = np.where(extract == 24)[0][3] + 1
        except IndexError:
            last_index = np.where(extract == 24)[0][2] + 1

        if np.where(extract == 0)[0][-3] - np.where(extract == 0)[0][-4] < 10:
            first_index = np.where(extract == 0)[0][-3]
        else:
            first_index = np.where(extract == 0)[0][-4]
        extract = extract[first_index:last_index, :]

        mean1_03,anticipatory_error1_03 = mean_calculate(extract, 1, 0.3, )
        mean1_06,anticipatory_error1_06 = mean_calculate(extract, 1, 0.6)
        mean1_1,anticipatory_error1_1 = mean_calculate(extract, 1, 1)
        mean1_13,anticipatory_error1_13 = mean_calculate(extract, 1, 1.3)
        mean1_16,anticipatory_error1_16 = mean_calculate(extract, 1, 1.6)

        mean2_03,anticipatory_error2_03 = mean_calculate(extract, 2, 0.3)
        mean2_06,anticipatory_error2_06 = mean_calculate(extract, 2, 0.6)
        mean2_1,anticipatory_error2_1 = mean_calculate(extract, 2, 1)
        mean2_13,anticipatory_error2_13 = mean_calculate(extract, 2, 1.3)
        mean2_16,anticipatory_error2_16 = mean_calculate(extract, 2, 1.6)

        x_data = np.array([0.3, 0.6, 1, 1.3, 1.6])  # Distances (D1 to D5)
        y1_data = np.array([mean1_03, mean1_06, mean1_1, mean1_13, mean1_16])  # Reaction Times
        y2_data = np.array([mean2_03, mean2_06, mean2_1, mean2_13, mean2_16])

        # Initial guess for parameters: [xc, b, ymin, ymax]
        initial_guess = [[0.01, 0.01], [0.5, 0.5], [0.5, 0.1], [1, 0.5], [1, 0.1]]
        bounds = [(0, 2), (0, 1)]
        y1min = min(y1_data)
        y1max = max(y1_data)
        y2min = min(y2_data)
        y2max = max(y2_data)
        # Perform the optimization to minimize the RMSE
        best_rmse1 = 0.1
        best_rmse2 = 0.1
        best_guess1 = 0
        best_guess2 = 0
        xc1 = np.nan
        b1 = np.nan
        xc2 = np.nan
        b2 = np.nan
        for guess in initial_guess:
            result1 = minimize(objective, guess, args=(y1min, y1max, x_data, y1_data), method='nelder-mead', bounds=bounds)
            result2 = minimize(objective, guess, args=(y2min, y2max, x_data, y2_data), method='nelder-mead', bounds=bounds)

            if result1.success and result1.fun <= best_rmse1:
                best_rmse1 = result1.fun
                xc1, b1 = result1.x

            if result2.success and result2.fun <= best_rmse2:
                best_rmse2 = result2.fun
                xc2, b2 = result2.x

        results.append([filename, anticipatory_error1_03+anticipatory_error1_06+anticipatory_error1_1+anticipatory_error1_13+anticipatory_error1_16+anticipatory_error2_03+anticipatory_error2_06+anticipatory_error2_1+anticipatory_error2_13+anticipatory_error2_16,
                        xc1, b1, best_rmse1,mean1_03, mean1_06, mean1_1, mean1_13, mean1_16, xc2, b2, best_rmse2,mean2_03, mean2_06, mean2_1, mean2_13, mean2_16, ])
        file.close()

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d %H-%M")
csv_file_path = f'/Users/hagrid/Desktop/ML/PPS/PPS1 results {timestamp}.csv'

# Write the results to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(results)

rows = results[1:]  # drop header
social = np.array([r[5:10] for r in rows], dtype=float)  # subjects x 5 distances
nonsocial = np.array([r[13:18] for r in rows], dtype=float)


def group_stats(data):
    n = np.sum(~np.isnan(data), axis=0)
    mean = np.nanmean(data, axis=0)
    sem = np.nanstd(data, axis=0, ddof=1) / np.sqrt(n)
    return mean, sem


def fit_sigmoid(x, y):
    ymin, ymax = y.min(), y.max()
    best = None
    for g in initial_guess:
        res = minimize(objective, g, args=(ymin, ymax, x, y), method='nelder-mead', bounds=bounds)
        if res.success and (best is None or res.fun < best.fun):
            best = res
    return best.x, ymin, ymax


soc_mean, soc_sem = group_stats(social)
non_mean, non_sem = group_stats(nonsocial)

fig, ax = plt.subplots(figsize=(7, 5))
x_fine = np.linspace(x_data.min(), x_data.max(), 200)

for mean, sem, label, color in [(soc_mean, soc_sem, 'Social', 'tab:red'),
                                (non_mean, non_sem, 'Non-social', 'tab:blue')]:
    ax.errorbar(x_data, mean, yerr=sem, fmt='o', color=color, capsize=4, label=f'{label} (mean ± SEM)')
    (xc, b), ymin, ymax = fit_sigmoid(x_data, mean)  # sigmoid fit to group means
    ax.plot(x_fine, sigmoid(x_fine, xc, b, ymin, ymax), '--', color=color, alpha=0.7)

ax.set_xlabel('Distance')
ax.set_ylabel('Reaction time (s)')
ax.set_title(f'Group PPS (N = {len(rows)})')
ax.set_xticks(x_data)
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()