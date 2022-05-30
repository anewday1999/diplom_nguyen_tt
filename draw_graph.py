from matplotlib import pyplot as plt

plt.plot(2,3, marker="o", color="red")
plt.plot(1,4, marker="o", color="red")
plt.plot(2,5, marker="o", color="red")
plt.plot(2,4.5, marker="o", color="red")
plt.plot(3,4.5, marker="o", color="blue")
plt.plot(4,3, marker="o", color="blue")
plt.plot(5,4, marker="o", color="blue")
plt.plot(4,5, marker="o", color="blue")
x1, y1 = [0, 5], [4.25, 4.25]
plt.plot(x1, y1, marker = 'o')
plt.show()