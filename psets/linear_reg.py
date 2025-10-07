
def mult_linear_regression(X, y, learning_rate=0.01, iterations=1000):
    n, k = X.shape

    # adding in intercept col
    X_b = np.hstack((np.ones((n, 1)), X))  # shape (n, k+1)

    # initialize the betas randomly
    beta = np.random.randn(k + 1)

    # Track loss over iterations
    loss_hist = []

    for i in range(iterations):
        # finding y predictions
        y_pred = np.dot(X_b, beta)

        # updating loss MSE function
        loss = (1 / (2 * n)) * np.sum((y_pred - y) ** 2)
        loss_hist.append(loss)

        # calc gradients for each
        gradients = (1 / n) * np.dot(X_b.T, (y_pred - y))

        # updating betas based on slides from class
        beta = beta - learning_rate * gradients

    return beta, loss_hist
