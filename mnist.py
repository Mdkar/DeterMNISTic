# This code is adapted from http://www.clungu.com/Converting_a_DecisionTree_into_python_code/
# It outputs Rust code from a RandomForestClassifier model trained on the MNIST dataset
# this is actually how DeterMNISTic was created

import os
import struct
import numpy as np

from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.ensemble import RandomForestClassifier

PYTHON_INDENT_STEP = "  "

    
def stringify_list(_list):
    return f"[{', '.join(str(i) for i in _list)}]"

def stringify_list_java(_list):
    return f"{{{','.join(str(round(i, 3)) + 'f' for i in _list)}}}"

def probabilities(node_counts):
    """
    By default, the tree stores the number of datapoints from each class in a leaf node (as the node values)
    but we want to convert this into probabilities so the generated code acts like a propper model.

    We can use `softmax` of other squish-list-to-probabilities formulas (in this case `a / sum(A)`)
    """
    return node_counts / np.sum(node_counts)

def tree_to_code(tree, index):
    tree_ = tree.tree_
    print(f"def tree_model{index}(img):")

    def __recurse(node, depth):
        indent = PYTHON_INDENT_STEP * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # name = node_feature_name[node]
            name = f"img[{tree_.feature[node]}]"
            threshold = tree_.threshold[node]
            
            print(f"{indent}if ({name} <= {threshold}):")
            __recurse(tree_.children_left[node], depth + 1)

            print(f"{indent}else:")
            __recurse(tree_.children_right[node], depth + 1)
        else:
            print(f"{indent}return {stringify_list(probabilities(tree_.value[node][0]))}")

    __recurse(0, 1)
    print("")

def tree_to_java_code(tree, index):
    tree_ = tree.tree_
    print(f"public static float[] classifier{index}(int[] img) {{")

    def __recurse(node, depth):
        indent = PYTHON_INDENT_STEP * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # name = node_feature_name[node]
            name = f"img[{tree_.feature[node]}]"
            threshold = round(tree_.threshold[node], 3)
            
            print(f"if({name}<={threshold}f){{", end="")

            __recurse(tree_.children_left[node], depth + 1)

            print(f"}}else{{", end="")

            __recurse(tree_.children_right[node], depth + 1)
            print(f"}}", end="")
        else:
            print(f"return new float[] {stringify_list_java(probabilities(tree_.value[node][0]))};", end="")

    __recurse(0, 1)
    print("}")

def tree_to_rust_code(tree, index):
    tree_ = tree.tree_
    print(f"fn classifier{index}(img: &[u8; 784]) -> Vec<f32> {{")

    def __recurse(node, depth):
        indent = PYTHON_INDENT_STEP * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # name = node_feature_name[node]
            name = f"img[{tree_.feature[node]}]"
            threshold = tree_.threshold[node]
            
            print(f"{indent}if {name} <= {threshold} {{")
            __recurse(tree_.children_left[node], depth + 1)

            print(f"{indent}}} else {{")
            __recurse(tree_.children_right[node], depth + 1)
            print(f"{indent}}}")
        else:
            print(f"{indent}return vec!{stringify_list(probabilities(tree_.value[node][0]))};")

    __recurse(0, 1)
    print("}")

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
        
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)  

    return images, labels

X_train, y_train = load_mnist('EMNIST/raw', kind='emnist-letters-train')
print('// Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist('EMNIST/raw', kind='emnist-letters-test')
print('// Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

print("public class EmnistTree {\n\tpublic static int works() {\n\t\treturn 0;}")

# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
# print('Accuracy: %.2f' % dt.score(X_test, y_test))

# rf = RandomForestClassifier(n_estimators=10, n_jobs=-1, max_depth=10, min_samples_split=10)
rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10)
rf.fit(X_train, y_train)

print('// Accuracy: %.2f' % rf.score(X_test, y_test))

for i, tree in enumerate(rf.estimators_):
    tree_to_java_code(tree, i)

print("}")

