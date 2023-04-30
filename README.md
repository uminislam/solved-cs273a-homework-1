Download Link: https://assignmentchef.com/product/solved-cs273a-homework-1
<br>
This homework (and many subsequent ones) will involve data analysis and reporting on methods and results using Python code. You have to submit <strong><em>a single PDF file </em></strong>that contains everything to Gradescope, and associated each page of the PDF to each problem. This includes any text you wish to include to describe your results, the complete code snippets of how you attempted each problem, any figures that were generated, and scans of any work on paper that you wish to include. It is important that you include enough detail that we know how you solved the problem, since otherwise we will be unable to grade it.

I recommend that you use Jupyter/iPython notebooks to write your report. It will help you not only ensure all of the code for the solutions is included, but also provide an easy way to export your results to a PDF file <a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>. I recommend liberal use of Markdown cells to create headers for each problem and sub-problem, explaining your implementation/answers, and including any mathematical equations. For parts of the homework you do on paper, scan it in such that it is legible (there are a number of free Android/iOS scanning apps, if you do not have access to a scanner), and include it as an image in the iPython notebook<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>. If you have any questions/concerns about using iPython, ask us on Campuswire. If you decide not to use iPython notebooks, but go with Microsoft Word or Latex to create your PDF file, you have to make sure all of the answers can be generated from the code snippets included in the document.

<strong>Summary so far:                   </strong>(1) submit a single, standalone PDF report, with all code; (2) I recommed Jupyter notebooks.

<strong>Points:          </strong>This homework adds up to a total of <strong>100 points</strong>, as follows:

Problem 0: Get Connected                              5 points

Problem 1: Python &amp; Data Exploration         20 points

Problem 2: kNN Predictions                            25 points

Problem 3: Naïve Bayes Classifiers                45 points

Statement of Collaboration                             5 points

<h2>Problem 0: Get Connected</h2>

Please visit our class forum on Campuswire: <a href="https://campuswire.com/p/GAF58E3D6">https://campuswire.com/p/GAF58E3D6</a><a href="https://campuswire.com/p/GAF58E3D6">.</a> Campuswire will be the place to post your questions and discussions, rather than by email to me or the TAs, since chances are that other students have the same or similar questions, and will be helped by seeing the discussion. Remember, your Campuswire participation will be taken into account for the participation grade as well. You do not need to mention anything regarding this in the report, we will be able to check whether you have visited Campuswire or not.

<h2>Problem 1: Python &amp; Data Exploration</h2>

In this problem, we will explore some basic statistics and visualizations of an example data set. First, download the zip file for Homework 1, which contains some course code (the mltools directory) and the “Fisher iris” data set, and load the latter into Python:

<table width="624">

 <tbody>

  <tr>

   <td width="624"><strong>import </strong>numpy as np<strong>import </strong>matplotlib.pyplot as pltiris = np.genfromtxt(“data/iris.txt”,delimiter=None) # load the text fileY = iris[:,-1]                                                          # target value is the last columnX = iris[:,0:-1]                                                     # features are the other columns</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

The Iris data consist of four real-valued features used to predict which of three types of iris flower was measured (a three-class classification problem).

<table width="50">

 <tbody>

  <tr>

   <td width="50">X.shape</td>

  </tr>

 </tbody>

</table>

<ol>

 <li>Useto get the number of features and the data points. Report both numbers, mentioning which number is which. <em>(5 points)</em></li>

</ol>

<table width="57">

 <tbody>

  <tr>

   <td width="57">plt.hist</td>

  </tr>

 </tbody>

</table>

<ol start="2">

 <li>For each feature, plot a histogram () of the data values <em>(5 points)</em></li>

</ol>

<table width="43">

 <tbody>

  <tr>

   <td width="43">np.std</td>

  </tr>

 </tbody>

</table>

<ol start="3">

 <li>Compute the mean &amp; standard deviation of the data points for each feature ( mean ,) <em>(5 points)</em></li>

</ol>

<table width="154">

 <tbody>

  <tr>

   <td width="57">plt.plot</td>

   <td width="21">or</td>

   <td width="77">plt.scatter</td>

  </tr>

 </tbody>

</table>

<ol start="4">

 <li>For each pair of features (1,2), (1,3), and (1,4), plot a scatterplot (see) of the feature values, colored according to their target value (class). (For example, plot all data points with <em>y </em>= 0 as blue, <em>y </em>= 1 as green, etc.) <em>(5 points)</em></li>

</ol>

<h2>Problem 2: kNN predictions</h2>

<table width="77">

 <tbody>

  <tr>

   <td width="77">knnClassify</td>

  </tr>

 </tbody>

</table>

In this problem, you will continue to use the Iris data and explore a KNN classifier using provided python class. While doing the problem, please explore the implementation to become familiar with how it works. First, we will shuffle and split the data into training and validation subsets:

<table width="624">

 <tbody>

  <tr>

   <td width="624">iris = np.genfromtxt(“data/iris.txt”,delimiter=None) # load the dataY = iris[:,-1]X = iris[:,0:-1]# Note: indexing with “:” indicates all values (in this case, all rows);# indexing with a value (“0”, “1”, “-1”, etc.) extracts only that value (here, columns); # indexing rows/columns with a range (“1:-1”) extracts any row/column in that range.<strong>import </strong>mltools as ml# We’ll use some data manipulation routines in the provided class code# Make sure the “mltools” directory is in a directory on your Python path, e.g.,# export PYTHONPATH=$$${PYTHONPATH}:/path/to/parent/dir # or add it to your path inside Python:# import sys# sys.path.append(‘/path/to/parent/dir/’);X,Y = ml.shuffleData(X,Y);                                                                 # shuffle data randomly# (This is a good idea in case your data are ordered in some pathological way, # as the Iris data are)Xtr,Xva,Ytr,Yva = ml.splitData(X,Y, 0.75); # split data into 75/25 train/validation</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

You may also find it useful to set the random number seed at the beginning (in general, for every assignment),

<table width="138">

 <tbody>

  <tr>

   <td width="138">numpy.random.seed(0)</td>

  </tr>

 </tbody>

</table>

e.g.,, to ensure consistent behavior each time.

<strong>Learner Objects           </strong>Our learners (the parameterized functions that do the prediction) will be defined as python objects, derived from either an abstract classifier or abstract regressor class. The abstract base classes have a few useful functions, such as computing error rates or other measures of quality. More importantly, the learners will all follow a generic behavioral pattern, allowing us to train the function on a data set (i.e., set the parameters of the model to perform well on those data), and make predictions on a data set.

<table width="50">

 <tbody>

  <tr>

   <td width="50">Xtr,Ytr</td>

  </tr>

 </tbody>

</table>

You can build now and <em>train </em>a kNN classifier onand make predictions on some data Xva with it:

<table width="624">

 <tbody>

  <tr>

   <td width="624">knn = ml.knn.knnClassify()                                          # create the object and train itknn.train(Xtr, Ytr, K) # where K is an integer, e.g. 1 for nearest neighbor prediction YvaHat = knn.predict(Xva) # get estimates of y for each data point in Xva# Alternatively, the constructor provides a shortcut to “train”: knn = ml.knn.knnClassify( Xtr, Ytr, K );YvaHat = predict( knn, Xva );</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

If your data are 2D, you can visualize a data set and a classifier’s decision regions using e.g.,

<table width="624">

 <tbody>

  <tr>

   <td width="262">ml.plotClassify2D( knn, Xtr, Ytr );</td>

   <td width="362"># make 2D classification plot with data (Xtr,Ytr)</td>

  </tr>

 </tbody>

</table>

1

<table width="50">

 <tbody>

  <tr>

   <td width="50">predict</td>

  </tr>

 </tbody>

</table>

This function plots the training data and colored points as per their labels, then calls knn ’sfunction on a densely spaced grid of points in the 2D space, and uses this to produce the background color. Calling the function

with knn=None will plot only the data.

<ol>

 <li>Modify the code listed above to use only the first two features of <em>X </em>(e.g., let <em>X </em>be only the first two columns of iris , instead of the first four), and visualize (plot) the classification boundary for varying values of</li>

</ol>

<table width="97">

 <tbody>

  <tr>

   <td width="97">plotClassify2D</td>

  </tr>

 </tbody>

</table>

<em>K </em>=[1, 5, 10, 50] using. <em>(10 points)</em>

<ol start="2">

 <li>Again using only the first two features, compute the error rate (number of misclassifications) on both the training and validation data as a function of <em>K </em>=[1, 2, 5, 10, 50, 100, 200]. You can do this most easily with a for-loop:</li>

</ol>

<table width="591">

 <tbody>

  <tr>

   <td width="591">K=[1,2,5,10,50,100,200]; <strong>for </strong>i,k <strong>in enumerate</strong>(K):learner = ml.knn.knnClassify(… # TODO: complete code to train model Yhat = learner.predict(…         # TODO: predict results on training data errTrain[i] = … # TODO: count what fraction of predictions are wrong#TODO: repeat prediction / error evaluation for validation data plt.semilogx(…   #TODO: average and plot results on semi-log scale</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

8

<table width="57">

 <tbody>

  <tr>

   <td width="57">semilogx</td>

  </tr>

 </tbody>

</table>

Plot the resulting error rate functions using a semi-log plot (), with training error in red and validation error in green. Based on these plots, what value of K would you recommend? <em>(10 points)</em>

<ol start="3">

 <li>Provide the same plots as the previous, but with all the features in the dataset. Are the plots very different? Is your recommendation different? <em>(5 points)</em></li>

</ol>

<h2>Problem 3: Naïve Bayes Classifiers</h2>

In order to reduce my email load, I decide to implement a machine learning algorithm to decide whether or not I should read an email, or simply file it away instead. To train my model, I obtain the following data set of binary-valued features about each email, including whether I know the author or not, whether the email is long or short, and whether it has any of several key words, along with my final decision about whether to read it (<em>y </em>=+1

<table width="472">

 <tbody>

  <tr>

   <td colspan="3" width="270">for “read”, <em>y </em>= <em>−</em>1 for “discard”).</td>

   <td rowspan="2" width="75"><em>x</em><sub>4</sub></td>

   <td rowspan="2" width="80"><em>x</em><sub>5</sub></td>

   <td rowspan="2" width="47"><em>y</em></td>

  </tr>

  <tr>

   <td width="121"><em>x</em><sub>1</sub></td>

   <td width="59"><em>x</em><sub>2</sub></td>

   <td width="90"><em>x</em><sub>3</sub></td>

  </tr>

  <tr>

   <td width="121">know author? 0</td>

   <td width="59">is long? 0</td>

   <td width="90">has ‘research’ 1</td>

   <td width="75">has ‘grade’ 1</td>

   <td width="80">has ‘lottery’0</td>

   <td width="47">read?-1</td>

  </tr>

  <tr>

   <td width="121">1</td>

   <td width="59">1</td>

   <td width="90">0</td>

   <td width="75">1</td>

   <td width="80">0</td>

   <td width="47">-1</td>

  </tr>

  <tr>

   <td width="121">0</td>

   <td width="59">1</td>

   <td width="90">1</td>

   <td width="75">1</td>

   <td width="80">1</td>

   <td width="47">-1</td>

  </tr>

  <tr>

   <td width="121">1</td>

   <td width="59">1</td>

   <td width="90">1</td>

   <td width="75">1</td>

   <td width="80">0</td>

   <td width="47">-1</td>

  </tr>

  <tr>

   <td width="121">0</td>

   <td width="59">1</td>

   <td width="90">0</td>

   <td width="75">0</td>

   <td width="80">0</td>

   <td width="47">-1</td>

  </tr>

  <tr>

   <td width="121">1</td>

   <td width="59">0</td>

   <td width="90">1</td>

   <td width="75">1</td>

   <td width="80">1</td>

   <td width="47">1</td>

  </tr>

  <tr>

   <td width="121">0</td>

   <td width="59">0</td>

   <td width="90">1</td>

   <td width="75">0</td>

   <td width="80">0</td>

   <td width="47">1</td>

  </tr>

  <tr>

   <td width="121">1</td>

   <td width="59">0</td>

   <td width="90">0</td>

   <td width="75">0</td>

   <td width="80">0</td>

   <td width="47">1</td>

  </tr>

  <tr>

   <td width="121">1</td>

   <td width="59">0</td>

   <td width="90">1</td>

   <td width="75">1</td>

   <td width="80">0</td>

   <td width="47">1</td>

  </tr>

  <tr>

   <td width="121">1</td>

   <td width="59">1</td>

   <td width="90">1</td>

   <td width="75">1</td>

   <td width="80">1</td>

   <td width="47">-1</td>

  </tr>

 </tbody>

</table>

In the case of any ties, we will prefer to predict class +1.

I decide to try a naïve Bayes classifier to make my decisions and compute my uncertainty.

<ol>

 <li>Compute all the probabilities necessary for a naïve Bayes classifier, i.e., the class probability <em>p</em>(<em>y</em>) and all the individual feature probabilities <em>p</em>(<em>x<sub>i</sub>|y</em>), for each class <em>y </em>and feature <em>x<sub>i </sub></em><em>(10 points)</em></li>

 <li>Which class would be predicted for <u>x </u>=(0 0 0 0 0)? What about for <u>x </u>=(1 1 0 1 0)? <em>(10 points)</em></li>

 <li>Compute the posterior probability that <em>y </em>=+1 given the observation <u>x </u>=(1 1 0 1 0). <em>(5 points)</em></li>

 <li>Why should we probably not use a “joint” Bayes classifier (using the joint probability of the features <em>x</em>, as opposed to a naïve Bayes classifier) for these data? <em>(10 points)</em></li>

 <li>Suppose that, before we make our predictions, we lose access to my address book, so that we cannot tell whether the email author is known. Should we re-train the model, and if so, how? (e.g.: how does the model, and its parameters, change in this new situation?) Hint: what will the naïve Bayes model over only features <em>x</em><sub>2 </sub>. . . <em>x</em><sub>5 </sub>look like, and what will its parameters be? <em>(10 points)</em></li>

</ol>

<h2>Statement of Collaboration (5 points)</h2>

It is <strong>mandatory </strong>to include a <em>Statement of Collaboration </em>in each submission, with respect to the guidelines below. Include the names of everyone involved in the discussions (especially in-person ones), and what was discussed.

All students are required to follow the academic honesty guidelines posted on the course website. For programming assignments, in particular, I encourage the students to organize (perhaps using Campuswire) to discuss the task descriptions, requirements, bugs in my code, and the relevant technical content <em>before </em>they start working on it. However, you should not discuss the specific solutions, and, as a guiding principle, you are not allowed to take anything written or drawn away from these discussions (i.e. no photographs of the blackboard,

written notes, referring to Campuswire, etc.). Especially <em>after </em>you have started working on the assignment, try to restrict the discussion to Campuswire as much as possible, so that there is no doubt as to the extent of your collaboration.


