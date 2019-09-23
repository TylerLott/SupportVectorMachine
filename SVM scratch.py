import matplotlib.pyplot as plt
import numpy as np


#############################################
# support vector machine coded from scratch and extensively commented


class SupportVectorMachine:

    # have self and visualization which defines a plot and the axes for the plot
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.data = None
        self.min_feature_value = None
        self.max_feature_value = None
        self.w = None
        self.b = None

        # defines colors for the classes on the plot
        self.colors = {1: 'g', -1: 'r'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # this is the actual training for the svm
    # takes data as a parameter
    def fit(self, data):
        self.data = data

        # creates a dictionary for the options that are found later
        opt_dict = {}

        # creates a list of lists for the transforms to check all w possibilities
        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]

        # creates a list of all data
        all_data = []
        # for all yi, groups of (x,y), in the data we passed
        for yi in self.data:
            # for all features in the data at the list indices yi
            for feature_set in self.data[yi]:
                # adds the features from the feature set to a list of all data
                for feature in feature_set:
                    all_data.append(feature)

        # gives max and min values of the features in the set
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

        # makes the step values for optimization a function of the max value
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001,]

        # hard codes the multiple range for b
        b_range_multiple = 5

        # is the b multiple for the b steps
        b_multiple = 5

        # sets first optimum as arbitrary max value multiplied by 10
        latest_optimum = self.max_feature_value * 10

        # for loop to find the optimal w and b
        for step in step_sizes:

            # uses w of the latest optimum
            w = np.array([latest_optimum, latest_optimum])

            # not optimized until all steps in step_size have been gone through
            optimized = False

            # while not optimized it runs this optimization loop
            while not optimized:

                # gives 3 b values to optimize w for a range
                # between -1 * max feature * b range multiple
                # and max feature * b range multiple
                # with step b multiple * the step value
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple, step * b_multiple):
                    # cycles through all the transformations of w
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True

                        # weakest link in SVM fundamentally
                        # goes through all data and dots xi and w then add b
                        # if this is not greater than or equal to 1 it is not a found option
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:

                                    # if this isn't true for all xi in the set it breaks
                                    found_option = False
                                    break

                        # if the data is found to be greater than 1 the it adds it to the opt_dict with a class of the
                        # normal to the w_t value that it is testing with a list in that class of the w and b values
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                # takes the step for w by having w go to zero
                if w[0] < 0:
                    optimized = True
                    print('optimized a step')
                else:
                    w = w - step

            # once the w value has been optimized the opt_dict is sorted by the class which is the normal to the w_t
            # this sorts w_t from lowest value to the highest and put it into a list
            norms = sorted([n for n in opt_dict])

            # chooses the smallest normal for the w vector
            opt_choice = opt_dict[norms[0]]

            # sets w and b for the fitment to w_t and b for the lowest normalized w_t in the opt_dict
            self.w = opt_choice[0]
            self.b = opt_choice[1]

            # sets latest optimum to the last opt + the step *2 so that we can take smaller steps to get a
            # better optimum. then goes on to do the next step
            latest_optimum = opt_choice[0][0] + step * 2

    def predict(self, features):
        # sign of x dotted with w + b
        # classifies prediction as either + or -
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        # if the predicted point isn't exactly on the decision plane
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        # creates a scatter plot of the x and y of each data point as well as assigns a color fo rhte class
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # nested method that takes x, w, b, and v to creat a hyperplane
        def hyperplane(x, w, b, v):
            return (-w[0]*x-b+v) / w[1]

        # specifies data field to plot on
        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # plots the positive hyperplane on the defined plot field
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # plots the negative hyperplane on the defined plot field
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # plots the decision hyperplane on the defined plot field
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


# Small sample dataset to train on and make sure it's working
data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8],]),
             1: np.array([[5, 1],
                         [6, -1],
                         [7, 3],])
             }

# passes our data through the support vector machine object that we created
svm = SupportVectorMachine()
svm.fit(data=data_dict)


# uses our predict method to predict points in our support vector machine object
predict_us = [[1,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]

# visualizes our predictions
for p in predict_us:
    svm.predict(p)
svm.visualize()





