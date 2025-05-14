import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

from sklearn.metrics import confusion_matrix


def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=300):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(None)
    plt.xlabel("预测标签", fontsize=15)
    plt.ylabel("真实标签", fontsize=15)
    plt.yticks(range(label_name.__len__()), label_name, fontsize=15)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45, fontsize=15)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color, fontsize=15)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)


def draw_time_cost_bar_chart(fit_time_df, title="Time Cost of Each Tree", pdf_save_path=None, dpi=300):
    """
    绘制每棵树的训练时间折线图
    :param fit_time_df: 每棵树的训练时间数据框
    :param title: 图标题
    :param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    :param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    :return:
    """
    plt.figure(figsize=(15, 10))
    plt.plot(fit_time_df.iloc[:, 0], fit_time_df.iloc[:, 1], marker='o', linestyle='-', color='b',fontsize=20)
    plt.xlabel('Tree Number',fontsize=20)
    plt.ylabel('Time Cost (seconds)',fontsize=20)
    plt.title(title,fontsize=20)
    plt.xticks(rotation=45,fontsize=20)
    plt.tight_layout()

    if pdf_save_path is not None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
    # plt.show()  # 如需交互式显示可取消注释


def draw_compare_train_time_line_chart(huffman_time_df, onehot_time_df, title="不同编码下模型训练时间", pdf_save_path=None, dpi=300):
    """
    绘制huffman和onehot两种编码下的训练时间折线图
    :param huffman_time_df: huffman编码下的训练时间DataFrame，需包含'train_time'列
    :param onehot_time_df: onehot编码下的训练时间DataFrame，需包含'train_time'列
    :param title: 图标题
    :param pdf_save_path: 保存路径
    :param dpi: 分辨率
    :return:
    """
    plt.figure(figsize=(10, 6))
    plt.plot(huffman_time_df.index + 1, huffman_time_df['train_time'], marker='o', linestyle='-', color='b', label='哈夫曼编码')
    plt.plot(onehot_time_df.index + 1, onehot_time_df['train_time'], marker='s', linestyle='-', color='r', label='独热编码')
    plt.xlabel('训练轮数', fontsize=20)
    plt.ylabel('训练时间 (秒)', fontsize=20)
    plt.title(title, fontsize=20)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # 为右侧图例留出空间
    plt.ylim(bottom=0)  # y轴从0开始
    if pdf_save_path is not None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
    # plt.show()  # 如需交互式显示可取消注释
