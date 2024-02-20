import matplotlib.pyplot as plt
import re

def extract_acc(filename):
    accuracy_array = []
    with open(filename, 'r') as file:
        for line in file:
            match = re.search(r'Accuracy: ([\d.]+)', line)
            if match:
                accuracy = round(float(match.group(1)), 2)
                accuracy_array.append(accuracy)
    return accuracy_array


patients = list(range(20, 150, 20))
accuracies1 = extract_acc('exp1_results/result-III-V3-V5.txt')
accuracies2 = extract_acc('exp1_results/result-III-V3.txt')
accuracies3 = extract_acc('exp1_results/result-V5.txt')
accuracies4 = extract_acc('exp1_results/result-V3.txt')
accuracies5 = extract_acc('exp1_results/result-III.txt')

plt.plot(patients, accuracies1, marker='.', label='III+V3+V5')
plt.plot(patients, accuracies2, marker='.', label='III+V3')
plt.plot(patients, accuracies3, marker='.', label='V5')
plt.plot(patients, accuracies4, marker='.', label='V3')
plt.plot(patients, accuracies5, marker='.', label='III')

plt.xlabel('Number of subjects')
plt.ylabel('Accuracy %')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('exp1_results/fig_exp1.png')