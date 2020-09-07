import numpy as np
import os
import argparse
#from tensorflow.python.summary.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python.summary.summary_iterator import summary_iterator

import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tensorflow_log(path, epochs):

    valdir = os.path.join(path, 'validation')
    traindir = os.path.join(path, 'train')
    metricdir = os.path.join(path, 'metrics')

    val_events = os.path.join(valdir, os.listdir(valdir)[0])
    train_events = os.path.join(traindir, os.listdir(traindir)[0])
    metric_events = os.path.join(metricdir, os.listdir(metricdir)[0])

    #for summary in summary_iterator(metric_events):
    #    print(summary)

    val_acc = EventAccumulator(val_events)
    val_acc.Reload()
    train_acc = EventAccumulator(train_events)
    train_acc.Reload()
    metric_acc = EventAccumulator(metric_events)
    metric_acc.Reload()

    # Show all tags in the log file
    #print(val_acc.Tags())
    #print(train_acc.Tags())
    #print(metric_acc.Tags())

    val_epoch_loss = val_acc.Scalars('epoch_loss')
    val_epoch_accuracy = val_acc.Scalars('epoch_accuracy')
    #print(val_epoch_accuracy)

    train_epoch_loss = train_acc.Scalars('epoch_loss')
    train_epoch_accuracy = train_acc.Scalars('epoch_accuracy')

    #training_accuracies =   event_acc.Scalars('training-accuracy')
    #validation_accuracies = event_acc.Scalars('validation_accuracy')

    steps = len(train_epoch_accuracy)
    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in range(steps):
        y[i, 0] = train_epoch_accuracy[i][2] # value
        y[i, 1] = val_epoch_accuracy[i][2]
    
    print('Final train accuracy:', train_epoch_accuracy[-1])
    print('Final val accuracy:', val_epoch_accuracy[-1])

    plt.plot(x, y[:,0], label='training accuracy')
    plt.plot(x, y[:,1], label='validation accuracy')
    plt.xlim(0, epochs)
    plt.ylim(0, 1)

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Progress")
    plt.grid(True, axis='both')
    plt.legend(loc='lower right', frameon=True)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    #log_file = "./logs/events.out.tfevents.1456909092.DTA16004"
    plot_tensorflow_log(args.logdir, args.epochs)
