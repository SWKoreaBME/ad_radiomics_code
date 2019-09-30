import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dummy_labelize_swk(data, n_classes):

    """
        Make labels into dummy form

        (example)

        input : [0, 1, 2, 0, 0]
        output : [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0]
        ]
    
    Returns:
        [array] -- [dummy for of labels]
    """
    
    label = np.zeros((len(data), n_classes), dtype=int)
    
    for k, i in zip(data, label):
        i[k] = 1
    
    return label

def boxplot():

    # all results box plot

    sns.set(style='whitegrid')

    plt.figure(figsize=(70,50))

    b = sns.boxplot(data=whole_rocs)
    b.set_xlabel("number of k",fontsize=40)
    b.set_ylabel("auc score",fontsize=40)
    b.tick_params(labelsize=30)

    return 