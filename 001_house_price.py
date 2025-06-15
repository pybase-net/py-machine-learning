import numpy as np
import matplotlib.pyplot as plt


def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


def main():
    # Load the dataset
    data = np.loadtxt('./001_house_price/dataset.csv',
                      delimiter=',', skiprows=1)
    # Split the dataset into features and target variable
    x_train = data[:, :-1]  # Features (all columns except the last)
    y_train = data[:, -1]   # Target variable (last column)
    # Print the shapes of the features and target variable
    # shape: dimensions of the arrays
    print(f"Features shape: {x_train.shape}")
    print(f"Target shape: {y_train.shape}")

    # Plot the first feature against the target variable
    plt.style.use('./deeplearning.mplstyle')
    plt.scatter(x_train, y_train, marker='o', c='#1f77b4', edgecolor='k', s=50)
    plt.title("Housing Prices")
    plt.ylabel('Price (in billions of VNDs)')
    plt.xlabel('Area (in m2)')
    plt.show()

    # f(x) = w * x + b
    # where w is the weight and b is the bias
    # and plots the first feature against the target variable.
    w = 100
    b = 100

    tmp_f_wb = compute_model_output(x_train, w, b,)

    # Plot our model prediction
    plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')

    # Plot the data points
    plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

    # Set the title
    plt.title("Housing Prices")
    plt.ylabel('Price (in billions of VNDs)')
    plt.xlabel('Area (in m2)')
    plt.legend()
    plt.show()
    print('Pause')


if __name__ == "__main__":
    main()
# This code loads a dataset from a CSV file, splits it into features and target variable,
