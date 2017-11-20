# Relationships Builder

For those couples who call it quits after a year, the main reason would be among the differences in their opinion, life styles and maybe financial stress. Therefore, ultimately even if the problem is being looked at from different scenarios and angles the main reason for any couple to break up would be because they are experiencing money troubles. After looking at the Mint transaction data, it can be inferred that the people who have broken up would be able to have a more successful current or future relationship based on the purchasing power of any individual. This is a machine learning project which pulls data from the mint financial transactions and then prepares the data as required. This resultant data is then fed to a neural network algorithm which performs an unsupervised learning over it to classify people based on the purchasing power of the individuals in the Big-Data set.

<strong>Scenario A:</strong>
Say, if a person A is interested in visiting restaurants quite often but if person B is not quite keen on visiting restaurants, then there is a higher likelihood that the relationship will not sustain for long. But then again if we try to understand the root cause in this situation the frequency of a person visiting a restaurant will be directly proportional to the money that the person invests for it which holds a dependency on the individual’s purchasing power. 

<strong>Scenario B:</strong>
Say, if a person A is interested in travelling and travels using public transport or taxi quite often but if person B is not quite keen on travelling, then there is a higher likelihood that the relationship will not sustain for long. This may be as a result of one not being available for the other. But then again if we try to understand the root cause in this situation the frequency of a person travelling professionally or for leisure will be directly proportional to the money that the person invests for it which holds a dependency on the individual’s purchasing power. 

<ul><strong>Anomalies / Outliers:</strong>
  <li>There can be a few people for whom the logic might not hold true and that would be clearly marked as false positives for the confusion matrix generated after using the neural networks machine learning algorithm for supervised learning for building relationships. </li>
  <li>I tried extracting features using the K Nearest Neighbor search from the scikit learn libarary in python but the results obtained where not as distinguishable and accurate as the neural network algorithm.</li>
</ul>

<a href="https://goo.gl/sR51o9">Check out the documentation for the solution and much more!</a>

<strong>Tech Used:</strong> Python (Neural Networks & other machine learning tools)



