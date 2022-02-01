#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import sys

arr = list(map(int, open('layer_sizes', 'r').read().split()))
#plt.plot(arr)

#brr = [100 * 2**30 * 8 / 3] * 1024
#plt.plot(brr)

#x_100gb=400
#plt.fill_between(range(x_100gb, 1024), [0] * (1024 - x_100gb), arr[x_100gb:])
#plt.show()

fig = plt.figure(figsize=(16,9))
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax1.set_xlabel('Layer sum')
ax1.set_xlim(0, len(arr) * 2)
ax1.xaxis.set_major_locator(plt.MultipleLocator(100))
ax1.xaxis.set_minor_locator(plt.MultipleLocator(10))
ax1.set_ylabel('Layer size')
ax1.yaxis.label.set_color('blue')
ax1.plot([2*i for i in range(len(arr))], arr)

ax2 = ax1.twinx()  # :)
ax2.set_ylabel('Fraction of winnable positions')
ax2.yaxis.label.set_color('red')
ax2.set_ylim(0, 1)
ax2.tick_params(axis='y', colors='red')
fractions_stuff = open('count-ones-db.txt').read().split('\n')[:-1]
available_fraction_labels, available_fractions = [], []
for label_and_cnt in fractions_stuff:
  label, cnt = map(int, label_and_cnt.split())
  if arr[label // 2 - 1] == 0:
    continue
  available_fraction_labels.append(label)
  available_fractions.append(cnt / arr[label // 2 - 1])
  #available_fractions.append(.5)
for i in range(len(available_fraction_labels)):
  for j in range(i):
    if (available_fraction_labels[i] > available_fraction_labels[j]):
      available_fraction_labels[i], available_fraction_labels[j] = available_fraction_labels[j], available_fraction_labels[i]
      available_fractions[i], available_fractions[j] = available_fractions[j], available_fractions[i]
ax2.plot(available_fraction_labels, available_fractions, 'r')
plt.savefig('Figure_1.png')
plt.show()

plt.clf()
mem = 0
done = 0
percent = int(np.sum(arr) / 100)
p = 0
mem_need = []
for layer in arr[::-1]:
  #mem = max(mem, layer / 8 * 3 / 1e9)
  mem = layer / 8 * 3 / 1e9
  done += layer
  while done >= percent * p:
    mem_need.append(mem)
    p += 1

plt.plot(mem_need)
plt.title('Memory requirements')
plt.xlabel('Percentage of work done')
plt.ylabel('GB')
plt.savefig('Figure_2.png')
plt.show()
