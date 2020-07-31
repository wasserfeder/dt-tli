lltinf - Signal temporal logic data classification
====================

usage: python main.py [-d D] [-n N] [-i I]
                             [{learn}] file

**Positional arguments:**

*{learn}*

Action to take: 

* 'learn': builds a classifier for the given training set. The resulting stl formula will be printed. 


*file*

.mat file containing the data

**Optional arguments:**

*  -d D, --depth D:
maximum depth of the shallow decision trees (default: 3)

*  -n N, --numtree N:
number of shallow decision trees for the boosted algorithm

*  -i I, --inc I:
Boolean variable for having incremental or non-incremental learning (0: non-inc, 1: inc)

Data set format
--------------------

The data set must be a .mat file with the following format three variables:

* data:
Matrix of real numbers that contains the signals.
Size: Nsignals x Nsamples.
Each row rapresents a signal, each column corresponds to a
sampling time.

* t:
Column vector of real numbers containing the sampling times.
Size: 1 x Nsamples.

* labels:
Column vector of real numbers that contains the labels for the
signals in data.
Size: 1 x Nsignals.
The label is +1 if the corresponging signal belongs to the
positive class C_p.
The label is -1 if the corresponging signal belongs to the
negative class C_N.
Can be omitted when used for classification.

Examples
--------------------
    $ python main.py -d 2 -n 2 -i 0 lean data/SimpleDS2/simpleDS2.mat


