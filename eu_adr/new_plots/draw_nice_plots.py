import numpy as np
from matplotlib import pyplot as plt
import pickle


def draw_learning_comparison(splits, r_score, u_score, d_score, samples_per_split, scoring, orig_size, min, max):
    """
    Plot the different learning methods on same graph
    """
    # create ticks for x axis
    # add size of original data to get correct values on x
    # TODO do this properly so that the final point is all of the data
    ticks = np.linspace(orig_size + samples_per_split, orig_size + (splits*samples_per_split), splits)

    buffer_space = (max - min)/10.0
    # set up the figure
    plt.figure()
    plt.grid()
    plt.xlabel('Training Instances')
    plt.xlim([1200, 1500])
    plt.ylabel(scoring)
    # set the axis limits
    plt.ylim([min - buffer_space, max + buffer_space])
    plt.title('Cross validation %s comparison using %s batches' % (scoring, splits))

    plt.plot(ticks, r_score, label='Random Sampling')
    plt.plot(ticks, u_score, label='Uncertainty Sampling')
    plt.plot(ticks, d_score, label='Density Sampling')

    plt.legend(loc='best')

    f_name = '%s_%s_splits.eps' % (scoring, splits)
    plt.savefig(f_name, format='eps')
    plt.clf()


def do_dem():
    """
    Draw plots for the passed in pickles
    """
    # this needs to match number of records used
    num_records = 352
    orig_size = 1201

    scores_5 = pickle.load(open('splits5.p', 'rb'))
    scores_10 = pickle.load(open('splits10.p', 'rb'))
    scores_20 = pickle.load(open('splits20.p', 'rb'))
    scores_40 = pickle.load(open('splits40.p', 'rb'))

    # concatenate all scores to calculate axis limits via min, max
    a = np.concatenate((np.concatenate(scores_5[0][1:]), np.concatenate(scores_10[0][1:]),
                        np.concatenate(scores_20[0][1:]), np.concatenate(scores_40[0][1:])))

    p = np.concatenate((np.concatenate(scores_5[1][1:]), np.concatenate(scores_10[1][1:]),
                        np.concatenate(scores_20[1][1:]), np.concatenate(scores_40[1][1:])))

    r = np.concatenate((np.concatenate(scores_5[2][1:]), np.concatenate(scores_10[2][1:]),
                        np.concatenate(scores_20[2][1:]), np.concatenate(scores_40[2][1:])))

    f = np.concatenate((np.concatenate(scores_5[3][1:]), np.concatenate(scores_10[3][1:]),
                        np.concatenate(scores_20[3][1:]), np.concatenate(scores_40[3][1:])))

    mins = [a.min(), p.min(), r.min(), f.min()]
    maxs = [a.max(), p.max(), r.max(), f.max()]

    for i in xrange(4):
        samples_per_split = (4 * num_records)/(5 * 5)
        draw_learning_comparison(5, scores_5[i][1], scores_5[i][2], scores_5[i][3], samples_per_split,
                                 scores_5[i][0],  orig_size, mins[i], maxs[i])

        samples_per_split = (4 * num_records)/(5 * 10)
        draw_learning_comparison(10, scores_10[i][1], scores_10[i][2], scores_10[i][3], samples_per_split,
                                 scores_10[i][0], orig_size, mins[i], maxs[i])

        samples_per_split = (4 * num_records)/(5 * 20)
        draw_learning_comparison(20, scores_20[i][1], scores_20[i][2], scores_20[i][3], samples_per_split,
                                 scores_20[i][0], orig_size, mins[i], maxs[i])

        samples_per_split = (4 * num_records)/(5 * 40)
        draw_learning_comparison(40, scores_40[i][1], scores_40[i][2], scores_40[i][3], samples_per_split,
                                 scores_40[i][0], orig_size, mins[i], maxs[i])

if __name__ == '__main__':
    do_dem()
