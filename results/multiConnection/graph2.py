import csv
import sys
import matplotlib.pyplot as plt


base = sys.argv[1]
#data = [[x+1] for x in range(30)]
algs = []
for end in ["Greedyf", "Centrality", "Degree", "Random"]:
    res = csv.reader(open("Gr" + end + base), delimiter="\t")
    col = [0]
    algs.append(col)
    for row in res:
        if (not(row[0][0].isdigit())):
            continue
        #data[int(row[0])-1].append(float(row[1]))
        col.append(float(row[1]))

plt.ylabel("active set size")
plt.xlabel("target set size")
ax = plt.subplot(1,1,1)
p1 = ax.plot(list(range(31)), algs[0],color='black', label="greedy")
p2 = ax.plot(list(range(31)), algs[1],color='black', ls='--', label="centrality")
p3 = ax.plot(list(range(31)), algs[2],color='black', ls='-.', label="degree")
p4 = ax.plot(list(range(31)), algs[3],color='black', ls=':', label="random")
handles,labels = ax.get_legend_handles_labels()
plt.legend(handles,labels,loc=4)

plt.show()
#plt.savefig(base+"plot")
