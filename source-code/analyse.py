import numpy as np
import matplotlib.pyplot as plt


def draw_accuracy(path_txt, epochs, save_path):
    datalist = np.loadtxt(path_txt)
    data = [x * 100 for x in datalist]
    print(data)
    y = data
    x = np.arange(0, epochs, 1)

    fig = plt.figure(figsize=[8, 6])
    sub = fig.add_subplot(111)

    sub.plot(x, y)
    sub.set_xlabel('Number of Epoches', fontsize=14)
    sub.set_ylabel('Accuracy', fontsize=14)
    x_scale = [0, epochs / 10, epochs / 5, 3 * epochs / 10, 2 * epochs / 5, epochs / 2,
               3 * epochs / 5, 7 * epochs / 10, 4 * epochs / 5, 9 * epochs / 10, epochs - 1]
    res = list(map(int, x_scale))
    sub.set_xticks(res)  # 主刻度
    sub.set_yticks([0, 20, 40, 60, 80, 100])  # 主刻度

    sub.grid(axis='both', which='both', linestyle='--')  # 网格
    sub.tick_params(axis='both', which='major', direction='out', length=5, width=2, grid_alpha=0.5)

    plt.show()
    res = save_path + "acc.jpg"
    fig.savefig(res, dpi=1000)  # dpi分辨率，默认100
    plt.close('all')


def test1():
    print('1')