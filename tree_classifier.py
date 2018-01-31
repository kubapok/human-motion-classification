import csv
from sklearn import tree


class tree_classifier():

    classes_dict = {0: 'going_left',
                    1: 'going_right',
                    2: 'falling',
                    3: 'just_sitting',
                    4: 'just_standing',
                    5: 'just_lying',
                    6: 'sitting_down',
                    7: 'standing_up'}

    def train():
        going_left = tree_classifier.load_class('going_left')
        going_left_value = [0 for _ in range(len(going_left))]

        going_right = tree_classifier.load_class('going_right')
        going_right_value = [1 for _ in range(len(going_right))]

        falling = tree_classifier.load_class('falling')
        falling_value = [2 for _ in range(len(falling))]

        just_sitting = tree_classifier.load_class('just_sitting')
        just_sitting_value = [3 for _ in range(len(just_sitting))]

        just_standing = tree_classifier.load_class('just_standing')
        just_standing_value = [4 for _ in range(len(just_standing))]

        lying_down = tree_classifier.load_class('lying_down')
        lying_down_value = [5 for _ in range(len(lying_down))]

        sitting_down = tree_classifier.load_class('sitting_down')
        sitting_down_value = [6 for _ in range(len(sitting_down))]

        standing_up = tree_classifier.load_class('standing_up')
        standing_up_value = [7 for _ in range(len(standing_up))]

        X = going_left + going_right + falling + just_sitting + \
            just_standing + lying_down + sitting_down + standing_up
        Y = going_left_value + going_right_value + falling_value + just_sitting_value + \
            just_standing_value + lying_down_value + sitting_down_value + standing_up_value

        tree_classifier.clf = tree.DecisionTreeClassifier(max_depth = 10)
        tree_classifier.clf.fit(X, Y)

        return tree_classifier.clf.predict([[43.48047639929654, 4.3354936021207635, 3.59]])

    def predict(sample):
        return tree_classifier.clf.predict([sample])[0]

    def load_class(class_name):
        l2 = []
        with open(class_name + '.tsv', 'r') as tsv:
            for line in csv.reader(tsv, quotechar='\t'):
                l2.append(line[0].split())
        l = []
        for x in l2:
            l.append([float(r) for r in x])
        return l


print(tree_classifier.train())
