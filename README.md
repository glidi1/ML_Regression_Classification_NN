# Machine Learning Lab Exercises (PyTorch)

Three lab notebooks from my MSc at Queen Mary University of London. Each one builds a standard model from scratch in PyTorch, without using the built-in model classes, so the maths is visible in the code.

The notebooks started from lab templates supplied by the module. The exercises ask for the core functions to be filled in (marked by the question numbers in each notebook), and the surrounding text and plotting code come from the template.

| Notebook | Topic | Data |
|---|---|---|
| `wk3_Regression.ipynb` | Linear regression with gradient descent, learning rate experiments, L2 regularisation | Diabetes progression data (442 patients, 10 features) |
| `wk4_Classification_I.ipynb` | Logistic regression, one-vs-rest for three classes, why XOR fails | Iris |
| `wk6_NN.ipynb` | Feed-forward network with hand-written backpropagation, then a PyTorch MLP | XOR, Iris |

## What is implemented

- **Regression:** feature standardisation, a linear regression layer, mean squared error with an L2 penalty, and a manual gradient descent step.
- **Classification:** sigmoid, a logistic regression layer, binary cross-entropy, a training loop, and three one-vs-rest classifiers combined with an argmax.
- **Neural networks:** a two-input, two-hidden-neuron, one-output network with forward pass and backward pass written by hand, then a one-hidden-layer MLP on Iris tried at several widths.

## Findings

- **Regression:** after standardising, BMI has the largest weight. A learning rate of 1 diverges to NaN, 0.1 gave the lowest error of the four rates tried, and 0.01 and 0.001 were still far from converged after the same number of steps.
- **Classification:** setosa is separated from the other two classes by a straight line. With three one-vs-rest classifiers, test accuracy was between 86% and 93% across runs, and every error was between versicolour and virginica. Logistic regression cannot fit XOR because the classes are not linearly separable.
- **Neural networks:** a hidden layer solves XOR. On Iris, test accuracy by hidden layer width:

| Hidden neurons | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Test accuracy | 0.63 | 0.93 | 0.83 | 0.93 | 0.97 | 0.97 |

The Iris test set is only 30 samples, so one sample is worth about 3 points and small differences between runs are noise.

## Running them

The regression notebook downloads its data from a public URL, so it needs an internet connection. The other two load Iris from scikit-learn.

```bash
pip install -r requirements.txt
jupyter notebook
```

## Tech

Python, PyTorch, scikit-learn, pandas, Matplotlib, Seaborn

## Author

Elidian Terziu
[LinkedIn](https://linkedin.com/in/elidian-terziu)
