import re
import numpy as np
from scipy.optimize import minimize
import os
import csv
import matplotlib.pyplot as plt

folder_path = '/Users/hagrid/Desktop/ML/PPS/test'

min_reaction_time=0.01
max_reaction_time=3
min_anticipatory_time=0.1
max_anticipatory_time=10

def mean_calculate (extract,condition,distance):
    new_list = extract[extract[:,3] == condition]
    new_list=new_list[new_list[:,1]==distance]
    error=np.mean(new_list[:,2])+np.std(new_list[:,2])*3

    if error>max_reaction_time:
        error=max_reaction_time
    print (condition,distance,error)
    for i in new_list[:,2]:
        if i>max_anticipatory_time or i<min_anticipatory_time:
            print("*** Potential Anticipatory Error ***")

    new_list=new_list[new_list[:,2]>=min_reaction_time]
    new_list=new_list[new_list[:,2]<=error]
    print(new_list)
    if len(new_list) == 0:
        print (f"No valid observations for condition={condition}, distance={distance}")
    return np.average(new_list[:,2])

def sigmoid(x, xc, b, ymin, ymax):
    return ((ymin + ymax * np.exp((x - xc) / b)) / (1 + np.exp((x - xc) / b)))


# Define the MSE objective function
def objective(params, ymin, ymax, x, y):
    xc, b = params
    y_pred = sigmoid(x, xc, b, ymin, ymax)
    mse = np.sqrt(np.mean((y - y_pred) ** 2))  # Mean Squared Error
    return mse

results=[]

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
        with open(file_path, 'r',errors='ignore') as file:
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
                if bracketed_text[x] == 'MALE THROWER' or bracketed_text[x]=='Spiked Ball':
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

        mean1_03 = mean_calculate(extract, 1, 0.3,)
        mean1_06 = mean_calculate(extract, 1, 0.6)
        mean1_1 = mean_calculate(extract, 1, 1)
        mean1_13 = mean_calculate(extract, 1, 1.3)
        mean1_16 = mean_calculate(extract, 1, 1.6)
        print (mean1_03, mean1_06, mean1_1, mean1_13, mean1_16)

        mean2_03 = mean_calculate(extract, 2, 0.3)
        mean2_06 = mean_calculate(extract, 2, 0.6)
        mean2_1 = mean_calculate(extract, 2, 1)
        mean2_13 = mean_calculate(extract, 2, 1.3)
        mean2_16 = mean_calculate(extract, 2, 1.6)
        print (mean2_03, mean2_06, mean2_1, mean2_13, mean2_16)

        x_data = np.array([0.3, 0.6, 1, 1.3, 1.6])  # Distances (D1 to D5)
        y1_data = np.array([mean1_03, mean1_06, mean1_1, mean1_13, mean1_16])  # Reaction Times
        y2_data = np.array([mean2_03, mean2_06, mean2_1, mean2_13, mean2_16])

        # Initial guess for parameters: [xc, b, ymin, ymax]
        initial_guess = [[0.01,0.01],[0.5,0.5],[0.5,0.1],[1,0.5],[1, 0.1]]
        bounds = [(1e-6,2),(1e-6,1)]
        y1min = min(y1_data)
        y1max = max(y1_data)
        y2min = min(y2_data)
        y2max = max(y2_data)
        # Perform the optimization to minimize the MSE
        best_mse1=0.1
        best_mse2=0.1

        xc1 = np.nan
        b1 = np.nan
        xc2 = np.nan
        b2 = np.nan

        for guess in initial_guess:
            result1 = minimize(objective, guess, args=(y1min, y1max, x_data, y1_data), method='nelder-mead',bounds=bounds)
            result2 = minimize(objective, guess, args=(y2min, y2max, x_data, y2_data), method='nelder-mead',bounds=bounds)

            if result1.success and result1.fun <= best_mse1:
                best_mse1 = result1.fun
                xc1, b1 = result1.x

            if result2.success and result2.fun <= best_mse2:
                best_mse2 = result2.fun

                xc2,b2 = result2.x

                # Extract optimal parameters
        results.append([filename,xc1,b1,xc2,b2])
        file.close()

print("Social Optimal parameters:")
print("xc =", xc1)
print("b =", b1)
print("mse =",best_mse1)

print("Non-social Optimal parameters:")
print("xc =", xc2)
print("b =", b2)
print("mse =",best_mse2)

plt.scatter(x_data,y1_data)
xsigmoid_fit = np.linspace(min(x_data), max(x_data), 1000)
ysigmoid1_fit = sigmoid(xsigmoid_fit,xc1,b1,y1min,y1max)
plt.plot(xsigmoid_fit, ysigmoid1_fit,label="Social")

plt.scatter(x_data,y2_data)
ysigmoid2_fit = sigmoid(xsigmoid_fit,xc2,b2,y2min,y2max)
plt.plot(xsigmoid_fit, ysigmoid2_fit,label="Non-Social")
plt.legend()
plt.show()