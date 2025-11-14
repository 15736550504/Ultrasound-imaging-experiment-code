import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
import pandas as pd

ious = [0.1, 0.2, 0.3, 0.4, 0.5]
categories = ["EDH", "IPH", "IVH", "SAH", "SDH"]
Datasets = ["RSNA Test", "CQ500", "Private Dataset 1", "Private Dataset 2"]


def function1():
    df = pd.read_csv("diffence_iou.csv", encoding="GBK")
    print(df)
    plt.rc('font', family='Arial', size=15)
    fig = plt.figure(figsize=(15, 10), dpi=200)
    for category in categories:
        plt.subplot(2, 3, categories.index(category) + 1)
        for dataset in Datasets:
            data = df[df["Datasets"] == dataset]
            data2 = data[category].values.tolist()
            print(data2)
            plt.plot(ious, data2, linestyle='-', lw=2,
                     label=dataset, zorder=1,
                     alpha=1.0)

        plt.title('%s' % category)
        plt.xlabel('IoU-T')
        plt.ylabel('AP')
        axes = plt.gca()
        plt.legend(loc="best")
        plt.grid()
        axes.set_xlim([0.1, 0.5])
        axes.set_ylim([0.0, 1.05])
    plt.subplot(2, 3, 6)
    for dataset in Datasets[:1]:
        data = df[df["Datasets"] == dataset]
        data2 = data["Mean"].values.tolist()
        print(data2)
        plt.plot(ious, data2, linestyle='-', lw=2,
                 label=dataset, zorder=1,
                 alpha=1.0)

    # plt.title('%s' % "mAP of each dataset")
    plt.title('%s' % "")
    plt.xlabel('IoU-T')
    plt.ylabel('mAP')
    axes = plt.gca()
    plt.legend(loc="best")
    plt.grid()
    axes.set_xlim([0.1, 0.5])
    axes.set_ylim([0.65, 0.70])

    plt.savefig("diffence_iou.jpg")
    plt.show()


def figure1_1():
    classes = ["YOLOv8n-SGFB", "YOLOv8n"]
    # colors = ["#F2A461","#27736E", "#299D91", "#8BB17B", "#E8C56A"]
    # colors = ["#27736E", "#299D91", "#8BB17B", "#E8C56A", "#F2A461"]
    # colors = ["#27736E", "#299D91", "#8BB17B", "#E8C56A", "#F2A461"]
    # colors = ["#D27F5B", "#ECBC78", "#82BAB5", "#708EB6", "#92627B"]
    # colors = ["#92627B","#D27F5B", "#ECBC78", "#82BAB5", "#708EB6", ]
    colors = ["#92627B", "#D27F5B"]
    # colors = ["#ff0000", "#27736E", "#0000ff", "#EF801B","#000000",]
    # colors = ["#A05F9F","#F2A461","#27736E", "#299D91", "#8BB17B", ]
    # shapes = ["D", "o", "p", "v", "X"]
    shapes = ["D", "o", ]
    # df = pd.read_excel("map数据.xlsx", sheet_name="Sheet1")
    fig = plt.figure(figsize=(7, 6), dpi=250)
    plt.rc('font', family='Arial', size=15)
    i = 0

    data1 = [[0.951,
              0.947,
              0.931,
              0.873,
              0.812,
              0.672,
              0.35, ], [0.92,
                        0.919,
                        0.903,
                        0.84,
                        0.76,
                        0.591,
                        0.321, ]]
    i=0
    for cls in iter(classes):
        data = data1[i]

        ious = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        axes = plt.gca()
        plt.xticks(ious, ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'])
        plt.plot(ious, data, shapes[i], markersize="8", linestyle='-', lw=2, zorder=2,
                 alpha=1.0, color=colors[i], label=cls)
        # plt.grid()
        # for i, j in zip(ious, data):
        #     plt.text(i, j + 0.01, "%.3f" % j, ha='center', va='bottom')

        plt.xlabel('IoU-T', fontdict={"size": 20})
        plt.ylabel('AP', fontdict={"size": 20})
        plt.tick_params(labelsize=20)
        # axes.set_xlim([0.15, 0.65])
        # axes.set_ylim([0.74, 0.85])
        axes.set_xlim([0.0, 0.8])
        axes.set_ylim([0.0, 1])

        i += 1
        # i += 1
    plt.legend()
    # plt.show()
    plt.savefig("ious.pdf")


if __name__ == '__main__':
    figure1_1()
    # pass
