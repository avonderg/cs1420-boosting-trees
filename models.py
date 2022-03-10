import numpy as np
import random
import copy
import math

def node_score_error(prob):
    '''
        TODO:
        Calculate the node score using the train error of the subdataset and return it.
        For a dataset with two classes, C(p) = min{p, 1-p}
    '''
    return np.minimum(prob, 1-prob)


def node_score_entropy(prob):
    '''
        TODO:
        Calculate the node score using the entropy of the subdataset and return it.
        For a dataset with 2 classes, C(p) = -p * log(p) - (1-p) * log(1-p)
        For the purposes of this calculation, assume 0*log0 = 0.
        HINT: remember to consider the range of values that p can take!

    '''
    if (prob > 0) and (prob < 1):
        left = (-prob)*np.log(prob)
        return left - ((1-prob)*np.log(1-prob))
    else:
        return 0


def node_score_gini(prob):
    '''
        TODO:
        Calculate the node score using the gini index of the subdataset and return it.
        For dataset with 2 classes, C(p) = 2 * p * (1-p)
    '''
    return (2*prob)*(1-prob)



class Node:
    '''
    Helper to construct the tree structure.
    '''
    def __init__(self, left=None, right=None, depth=0, index_split_on=0, isleaf=False, label=1):
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on
        self.isleaf = isleaf
        self.label = label
        self.info = {} # used for visualization


    def _set_info(self, gain, num_samples):
        '''
        Helper function to add to info attribute.
        You do not need to modify this. 
        '''

        self.info['gain'] = gain
        self.info['num_samples'] = num_samples


class DecisionTree:

    def __init__(self, data, validation_data=None, gain_function=node_score_entropy, max_depth=40):
        self.max_depth = max_depth
        self.root = Node()
        self.gain_function = gain_function

        indices = list(range(1, len(data[0])))

        self._split_recurs(self.root, data, indices)

        # Pruning
        if validation_data is not None:
            self._prune_recurs(self.root, validation_data)


    def predict(self, features):
        '''
        Helper function to predict the label given a row of features.
        You do not need to modify this.
        '''
        return self._predict_recurs(self.root, features)


    def accuracy(self, data):
        '''
        Helper function to calculate the accuracy on the given data.
        You do not need to modify this.
        '''
        return 1 - self.loss(data)


    def loss(self, data):
        '''
        Helper function to calculate the loss on the given data.
        You do not need to modify this.
        '''
        cnt = 0.0
        test_Y = [row[0] for row in data]
        for i in range(len(data)):
            prediction = self.predict(data[i])
            if (prediction != test_Y[i]):
                cnt += 1.0
        return cnt/len(data)


    def _predict_recurs(self, node, row):
        '''
        Helper function to predict the label given a row of features.
        Traverse the tree until leaves to get the label.
        You do not need to modify this.
        '''
        if node.isleaf or node.index_split_on == 0:
            return node.label
        split_index = node.index_split_on
        if not row[split_index]:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)


    def _prune_recurs(self, node, validation_data):
        '''
        TODO:
        Prune the tree bottom up recursively. Nothing needs to be returned.
        Do not prune if the node is a leaf.
        Do not prune if the node is non-leaf and has at least one non-leaf child.
        Prune if deleting the node could reduce loss on the validation data.
        NOTE:
        This might be slightly different from the pruning described in lecture.
        Here we won't consider pruning a node's parent if we don't prune the node 
        itself (i.e. we will only prune nodes that have two leaves as children.)
        HINT: Think about what variables need to be set when pruning a node!
        '''
        if (node.isleaf):
            pass
        else:
            if (node.left is not None): # if it is a node
                self._prune_recurs(node.left, validation_data) #check for nodes
            if (node.right is not None): # if it is a node
                self._prune_recurs(node.right, validation_data) # check for nodes
            if (node.left.isleaf and node.right.isleaf): #if both children leaves
                loss_before = self.loss(validation_data)
                left_node = node.left
                right_node = node.right
                node.left = None
                node.right = None
                node.isleaf = True

                #find label of leaf
                # nonzero_count = np.count_nonzero(validation_data[0])
                # zero_count = np.count_nonzero(validation_data[0] == 0)
                # if (nonzero_count > zero_count): #if it is true
                #     node.label = 1
                # else:
                #     node.label = 0

                loss_after = self.loss(validation_data)
                if (loss_before < loss_after): # if pruning did not minimize loss
                    node.left = left_node
                    node.right = right_node
                    node.isleaf = False
            #     loss_before = self.accuracy(validation_data)
            #     node.isleaf = True
            #     loss_after = self.accuracy(validation_data)
            #     if (loss_before <= loss_after): #pruning did not minimize loss
            #         node.left = None
            #         node.right = None
            #     else:
            #         node.isleaf = False
            # return



    def _is_terminal(self, node, data, indices):
        '''
        TODO:
        Helper function to determine whether the node should stop splitting.
        Stop the recursion if:
            1. The dataset is empty.
            2. There are no more indices to split on.
            3. All the instances in this dataset belong to the same class
            4. The depth of the node reaches the maximum depth.
        Return:
            - A boolean, True indicating the current node should be a leaf and 
              False if the node is not a leaf.
            - A label, indicating the label of the leaf (or the label the node would 
              be if we were to terminate at that node). If there is no data left, you
              can return either label at random.
        '''
        # nonzero_count = len(data[data[:,0] == 1])
        # zero_count = len(data[data[:,0] == 0])
        # if (nonzero_count > zero_count):
        #     val = 1
        # else:
        #     val = 0
        # # if statements
        # if (len(data)==0): #node is a leaf
        #     return True, random.randrange(0,2)
        # if (len(indices) == 0): #node is a leaf
        #     return True, val
        # if (len(data[data[:,0] == 0]) == len(data)) or (len(data[data[:,0] == 1]) == len(data)): #instances belong to same class
        #     zero_count = len(data[data[:,0] == 0])
        #     nonzero_count = len(data[data[:,0] == 1])
        #     val = 0
        #     if (zero_count == 0):
        #         val = 0
        #     else:
        #         val = 1
        #     return True, val
        # if (node.depth == self.max_depth):
        #     return True, val
        # else: #not a leaf
        #     return False, val

        if (len(data)==0): #node is a leaf
            return True, random.randrange(0,2)
        val = np.mean(data[:,0]) >= 0.5
        if (len(indices) == 0): #node is a leaf
            return True, val
        if (np.mean(data[:,0]) == 0 or np.mean(data[:,0]) == 1): #instances belong to same class
            return True, val
        if (node.depth == self.max_depth):
            return True, val
        else: #not a leaf
            return False, val


    def _split_recurs(self, node, data, indices):
        '''
        TODO:
        Recursively split the node based on the rows and indices given.
        Nothing needs to be returned.

        First use _is_terminal() to check if the node needs to be split.
        If so, select the column that has the maximum infomation gain to split on.
        Store the label predicted for this node, the split column, and use _set_info()
        to keep track of the gain and the number of datapoints at the split.
        Then, split the data based on its value in the selected column.
        The data should be recursively passed to the children.
        '''
        is_leaf, label = self._is_terminal(node, data, indices)
        node.label = label # set the label
        if (is_leaf): #split node or not
            node.isleaf = True
            return
        else:
            #calc gain for each index hypothetically split on, choose one maximizing gain
            gains = []
            for i in range(len(indices)):
                gains.append(self._calc_gain(data,i,self.gain_function))
            max_index = indices[np.argmax(gains)] #split on this index, remove from list
            indices.remove(max_index) #remove index we split on
            # not sure how to set this $
            node._set_info(np.max(gains), len(data))
            node.index_split_on = max_index
            # not sure how to set ^
            node.left =  Node(depth=node.depth + 1) #index of zero
            node.right = Node(depth=node.depth + 1)
            #index splitting on can take in either 0 or 1
            mask1 = data[:,max_index] == 0 #mask to use on original dataset
            left_data = data[mask1] #data for left child node
            mask2 = data[:,max_index] == 1 #mask to use on original dataset
            right_data = data[mask2] #data for right child node
            # remove index you alr split on from indices list, pass in list to recursive
            self._split_recurs(node.left,left_data,copy.copy(indices)) #shallow copy and deep copy?
            self._split_recurs(node.right,right_data,copy.copy(indices)) #shallow copy and deep copy?


    def _calc_gain(self, data, split_index, gain_function):
        '''
        TODO:
        Calculate the gain of the proposed splitting and return it.
        Gain = C(P[y=1]) - P[x_i=True] * C(P[y=1|x_i=True]) - P[x_i=False] * C(P[y=0|x_i=False])
        Here the C(p) is the gain_function. For example, if C(p) = min(p, 1-p), this would be
        considering training error gain. Other alternatives are entropy and gini functions.
        '''

        mask1 = data[:,split_index] == 0 #mask to use on original dataset
        left_data = data[mask1] #data for left child node
        mask2 = data[:,split_index] == 1 #mask to use on original dataset
        right_data = data[mask2] #data for right child node

        non_zero_count = np.mean(data[:,0]) if len(data) > 0 else 0.0
        p_split_left = np.mean(left_data[:,0]) if len(left_data) > 0 else 0.0
        p_split_right = np.mean(right_data[:,0]) if len(right_data) > 0 else 0.0
        
        gain_nonzero = gain_function(non_zero_count) 
        gain_left = gain_function(p_split_left) 
        gain_right = gain_function(p_split_right) 

        gain = gain_nonzero-(float(len(left_data))/len(data) * gain_left + float(len(right_data))/len(data) * gain_right)
        return gain   

    def print_tree(self):
        '''
        Helper function for tree_visualization.
        Only effective with very shallow trees.
        You do not need to modify this.
        '''
        print('---START PRINT TREE---')
        def print_subtree(node, indent=''):
            if node is None:
                return str("None")
            if node.isleaf:
                return str(node.label)
            else:
                decision = 'split attribute = {:d}; gain = {:f}; number of samples = {:d}'.format(node.index_split_on, node.info['gain'], node.info['num_samples'])
            left = indent + '0 -> '+ print_subtree(node.left, indent + '\t\t')
            right = indent + '1 -> '+ print_subtree(node.right, indent + '\t\t')
            return (decision + '\n' + left + '\n' + right)

        print(print_subtree(self.root))
        print('----END PRINT TREE---')


    def loss_plot_vec(self, data):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        self._loss_plot_recurs(self.root, data, 0)
        loss_vec = []
        q = [self.root]
        num_correct = 0
        while len(q) > 0:
            node = q.pop(0)
            num_correct = num_correct + node.info['curr_num_correct']
            loss_vec.append(num_correct)
            if node.left != None:
                q.append(node.left)
            if node.right != None:
                q.append(node.right)

        return 1 - np.array(loss_vec)/len(data)


    def _loss_plot_recurs(self, node, rows, prev_num_correct):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        labels = [row[0] for row in rows]
        curr_num_correct = labels.count(node.label) - prev_num_correct
        node.info['curr_num_correct'] = curr_num_correct

        if not node.isleaf:
            left_data, right_data = [], []
            left_num_correct, right_num_correct = 0, 0
            for row in rows:
                if not row[node.index_split_on]:
                    left_data.append(row)
                else:
                    right_data.append(row)

            left_labels = [row[0] for row in left_data]
            left_num_correct = left_labels.count(node.label)
            right_labels = [row[0] for row in right_data]
            right_num_correct = right_labels.count(node.label)

            if node.left != None:
                self._loss_plot_recurs(node.left, left_data, left_num_correct)
            if node.right != None:
                self._loss_plot_recurs(node.right, right_data, right_num_correct)
