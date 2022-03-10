import random
from tkinter.tix import Tree
import numpy as np
import matplotlib.pyplot as plt

from get_data import get_data
from models import DecisionTree, node_score_error, node_score_entropy, node_score_gini


def loss_plot(ax, title, tree, pruned_tree, train_data, test_data):
    '''
        Example plotting code. This plots four curves: the training and testing
        average loss using tree and pruned tree.
        You do not need to change this code!
        Arguments:
            - ax: A matplotlib Axes instance.
            - title: A title for the graph (string)
            - tree: An unpruned DecisionTree instance
            - pruned_tree: A pruned DecisionTree instance
            - train_data: Training dataset returned from get_data
            - test_data: Test dataset returned from get_data
    '''
    fontsize=8
    ax.plot(tree.loss_plot_vec(train_data), label='train non-pruned')
    ax.plot(tree.loss_plot_vec(test_data), label='test non-pruned')
    ax.plot(pruned_tree.loss_plot_vec(train_data), label='train pruned')
    ax.plot(pruned_tree.loss_plot_vec(test_data), label='test pruned')


    ax.locator_params(nbins=3)
    ax.set_xlabel('number of nodes', fontsize=fontsize)
    ax.set_ylabel('loss', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    legend = ax.legend(loc='upper center', shadow=True, fontsize=fontsize-2)

def explore_dataset(filename, class_name):
    train_data, validation_data, test_data = get_data(filename, class_name)

    # TODO: Print 12 loss values associated with the dataset.
    # For each measure of gain (training error, entropy, gini):
    #      (a) Print average training loss (not-pruned)
    #      (b) Print average test loss (not-pruned)
    #      (c) Print average training loss (pruned)
    #      (d) Print average test loss (pruned)
    decision_tree_ent = DecisionTree(data=train_data, validation_data=None, gain_function=node_score_entropy)
    print(decision_tree_ent.loss(train_data)) # average training loss
    print(decision_tree_ent.loss(test_data))

    p_decision_tree_ent = DecisionTree(data=train_data, validation_data=validation_data, gain_function=node_score_entropy)
    print(p_decision_tree_ent.loss(train_data)) # average training loss
    print(p_decision_tree_ent.loss(test_data))

    decision_tree_train = DecisionTree(data=train_data, validation_data=None, gain_function=node_score_error)
    print(decision_tree_train.loss(train_data)) # average training loss
    print(decision_tree_train.loss(test_data))

    p_decision_tree_train = DecisionTree(data=train_data, validation_data=validation_data, gain_function=node_score_error)
    print(p_decision_tree_train.loss(train_data)) # average training loss
    print(p_decision_tree_train.loss(test_data))

    decision_tree_gini = DecisionTree(data=train_data, validation_data=None, gain_function=node_score_gini)
    print(decision_tree_gini.loss(train_data)) # average training loss
    print(decision_tree_gini.loss(test_data))

    p_decision_tree_gini = DecisionTree(data=train_data, validation_data=validation_data, gain_function=node_score_gini)
    print(p_decision_tree_gini.loss(train_data)) # average training loss
    print(p_decision_tree_gini.loss(test_data))


    # TODO: Feel free to print or plot anything you like here. Just comment
    # make sure to comment it out, or put it in a function that isn't called
    # by default when you hand in your code!
    figure,axes = plt.subplots(1,3)
    entropy_graph = loss_plot(axes[0],"entropy error", decision_tree_ent,  p_decision_tree_ent, train_data, test_data)
    train_graph = loss_plot(axes[1],"train gain", decision_tree_train,  p_decision_tree_train, train_data, test_data)
    gini_graph = loss_plot(axes[2],"gini gain", decision_tree_gini,  p_decision_tree_gini, train_data, test_data)
    plt.savefig("hw6_plots.png")
    plt.show()



def main():
    ########### PLEASE DO NOT CHANGE THESE LINES OF CODE! ###################
    random.seed(1)
    np.random.seed(1)
    #########################################################################

    explore_dataset('data/chess.csv', 'won')
    explore_dataset('data/spam.csv', '1')

main()
