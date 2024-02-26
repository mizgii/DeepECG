import matplotlib.pyplot as plt
import re
import numpy as np
def extract_acc(filename):
    accuracy_array = []
    with open(filename, 'r') as file:
        for line in file:
            match = re.search(r'Accuracy: ([\d.]+)', line)
            if match:
                accuracy = round(float(match.group(1)), 2)
                accuracy_array.append(accuracy)
    return accuracy_array


num_subjects = list(range(20, 150, 20))
accuracies1 = extract_acc('exp1_results/result-III-V3-V5.txt')
accuracies2 = extract_acc('exp1_results/result-III-V3.txt')
accuracies3 = extract_acc('exp1_results/result-V5.txt')
accuracies4 = extract_acc('exp1_results/result-V3.txt')
accuracies5 = extract_acc('exp1_results/result-III.txt')

plt.figure(figsize=(10, 8), dpi=100)
plt.plot(num_subjects, accuracies1, '#8C2155',marker='.', label='III+V3+V5')
plt.plot(num_subjects, accuracies2, '#F99083',marker='.', label='III+V3')
plt.plot(num_subjects, accuracies3, '#8AA29E',marker='.', label='V5')
plt.plot(num_subjects, accuracies4, '#98CE00',marker='.', label='V3')
plt.plot(num_subjects, accuracies5, '#FFC857',marker='.', label='III')


plt.xlabel('Number of subjects', fontsize=16)
plt.ylabel('Accuracy (%)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend()
plt.xlim((np.min(num_subjects),np.max(num_subjects)))
plt.grid(True, 'both')
plt.savefig('exp1_results/fig_exp1.png')
plt.show()
plt.close()