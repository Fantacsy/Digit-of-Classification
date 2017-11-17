In this project, our goal is to classify images of digits using k-nearest neighbor classification and perceptron
algorithm. Through the result, we can see when k = 3, 3-NN method reach the minimum test error. 

Secondly, I compute the confusion matrix of 3-NN method. (The confusion matrix is a 10×10 matrix, where each row is labelled
0,...,9 and each column is labelled 0,...,9. The entry of the matrix at row i and column j is Cij/Nj where Cij is the
number of test examples that have label j but are classified as label i by the classifier, and Nj is the number of test
examples that have label j.) And found that it has high probability to make a mistake on P(output = 3, real = 5) = 0.077
and P(output = 0, real = 6) = 0.11.

Thirdly, I applied perceptron algorithm to classify 0 and 6, and logistic regression to classify 3 and 5. For perceptron algorithm,
the test error decrease to P(output = 3, real = 5) = 0.008. For logistic regression model, the test error decrease to
P(output = 0, real = 6) = 0.014

