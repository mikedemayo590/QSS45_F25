def mult_logistic_regression(X, y):
    n, k = X.shape

    # adding in intercept col
    X_b = np.hstack((np.ones((n, 1)), X))  # shape (n, k+1)

    # initialize the betas randomly then build from loss function
    beta = np.random.randn(k + 1)
    loss_hist = []

    for i in range(iterations):
        # finding predicted probabilities using sigmoid
        z = np.sum(X_b * beta, axis=1)
        y_pred = 1 / (1 + np.exp(-z))

        # updating loss using cross-entropy function
        eps = 1e-8  # prevents log(0)
        loss = -(1 / n) * np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
        loss_hist.append(loss)

        # calc gradients for each based on logistic gradient descent
        gradients = (1 / n) * np.sum((y_pred - y).reshape(-1, 1) * X_b, axis=0)

        # updating betas based on slides from class
        beta = beta - learning_rate * gradients

    return beta, loss_hist