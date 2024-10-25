import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import csv

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


class ConcretePredictionsMLP(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(256)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(1)(x)
        return x


class ConcretePredictionsMLP_KL(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(256)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(2)(x)  # Output mean and log_variance
        return x


def read_data() -> pd.DataFrame:
    raw_data = pd.read_excel("data/concrete_dataset.xls")
    raw_data.columns = [col.split("(")[0].strip() for col in raw_data.columns]
    return raw_data


def normalize_data(data: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def split_data_kfold(data: np.ndarray, target: np.ndarray, k: int = 10) -> list:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    return list(kf.split(data))


def pca_transform(data: np.ndarray, n_components: int = 2) -> np.ndarray:
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)


def baseline_model(y_train: np.ndarray) -> float:
    return np.mean(y_train)


def ridge_regression(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, lambda_: float = 0.1
) -> np.ndarray:
    ridge = Ridge(alpha=lambda_)
    ridge.fit(X_train, y_train)
    return ridge.predict(X_test)


def mlp_model(
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_test: jnp.ndarray,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    weight_decay: float = 0.01,
) -> jnp.ndarray:
    key = jax.random.PRNGKey(0)
    mlp = ConcretePredictionsMLP()
    params = mlp.init(key, X_train)

    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, X, y):
        preds = mlp.apply(params, X)
        mse_loss = jnp.mean((preds - y) ** 2)
        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params))
        return mse_loss + weight_decay * l2_loss

    @jax.jit
    def train_step(params, opt_state, X, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def predict(params, X):
        return mlp.apply(params, X)

    for _ in range(num_epochs):
        params, opt_state, _ = train_step(params, opt_state, X_train, y_train)

    return predict(params, X_test)


def mlp_kl_model(
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_test: jnp.ndarray,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    weight_decay: float = 0.01,
    kl_weight: float = 0.1,
) -> jnp.ndarray:
    key = jax.random.PRNGKey(0)
    mlp = ConcretePredictionsMLP_KL()
    params = mlp.init(key, X_train)

    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, X, y):
        outputs = mlp.apply(params, X)
        mean, log_var = outputs[:, 0], outputs[:, 1]
        var = jnp.exp(log_var)

        mse_loss = jnp.mean((mean - y) ** 2)
        kl_loss = -0.5 * jnp.mean(1 + log_var - mean**2 - var)
        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params))

        return mse_loss + kl_weight * kl_loss + weight_decay * l2_loss

    @jax.jit
    def train_step(params, opt_state, X, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def predict(params, X):
        outputs = mlp.apply(params, X)
        return outputs[:, 0]  # Return only the mean predictions

    for _ in range(num_epochs):
        params, opt_state, _ = train_step(params, opt_state, X_train, y_train)

    return predict(params, X_test)


def ridge_regression_cv(X, y, lambda_values, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = []

    for lambda_ in tqdm(lambda_values, desc="Lambda values"):
        fold_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            ridge = Ridge(alpha=lambda_)
            ridge.fit(X_train, y_train)
            y_pred = ridge.predict(X_test)
            mse = np.mean((y_test - y_pred) ** 2)
            fold_scores.append(mse)

        mse_scores.append(np.mean(fold_scores))

    return mse_scores


def random_forest_model(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf.predict(X_test)


def main():
    data = read_data()
    X = data.drop(columns=["Concrete compressive strength"])
    y = data["Concrete compressive strength"].values

    X_normalized = normalize_data(X)

    results = []

    # Create a single figure for all lambda vs MSE plots
    plt.figure(figsize=(12, 8))

    for n_components in range(1, 9):  # Loop from 1 to 8 components
        print(f"\nAnalyzing with {n_components} PCA components:")
        X_pca = pca_transform(X_normalized, n_components=n_components)

        # Define a range of lambda values
        lambda_values = np.logspace(-3, 3, 100)

        # Perform ridge regression with cross-validation
        mse_scores = ridge_regression_cv(X_pca, y, lambda_values)

        # Find the best lambda
        best_lambda_index = np.argmin(mse_scores)
        best_lambda = lambda_values[best_lambda_index]
        best_mse = mse_scores[best_lambda_index]

        print(f"Best lambda: {best_lambda:.4f}")
        print(f"Best MSE: {best_mse:.4f}")

        # Plot the results on the same figure
        plt.semilogx(lambda_values, mse_scores, label=f"{n_components} components")

        kf_splits = split_data_kfold(X_pca, y)

        baseline_errors = []
        ridge_errors = []
        mlp_errors = []
        mlp_kl_errors = []
        rf_errors = []

        for train_index, test_index in tqdm(kf_splits, desc="KFold splits"):
            X_train, X_test = X_pca[train_index], X_pca[test_index]
            y_train, y_test = y[train_index], y[test_index]

            baseline_pred = baseline_model(y_train)
            ridge_pred = ridge_regression(X_train, y_train, X_test, lambda_=best_lambda)
            mlp_pred = mlp_model(
                jnp.array(X_train), jnp.array(y_train), jnp.array(X_test)
            )
            mlp_kl_pred = mlp_kl_model(
                jnp.array(X_train), jnp.array(y_train), jnp.array(X_test)
            )
            rf_pred = random_forest_model(X_train, y_train, X_test)

            baseline_errors.append(np.mean((y_test - baseline_pred) ** 2))
            ridge_errors.append(np.mean((y_test - ridge_pred) ** 2))
            mlp_errors.append(np.mean((y_test - mlp_pred) ** 2))
            mlp_kl_errors.append(np.mean((y_test - mlp_kl_pred) ** 2))
            rf_errors.append(np.mean((y_test - rf_pred) ** 2))

        results.append(
            {
                "n_components": n_components,
                "best_lambda": best_lambda,
                "baseline_mse": np.mean(baseline_errors),
                "ridge_mse": np.mean(ridge_errors),
                "mlp_mse": np.mean(mlp_errors),
                "mlp_kl_mse": np.mean(mlp_kl_errors),
                "rf_mse": np.mean(rf_errors),
            }
        )

        print(f"Baseline Model Mean MSE: {np.mean(baseline_errors):.4f}")
        print(f"Ridge Regression Mean MSE: {np.mean(ridge_errors):.4f}")
        print(f"MLP Model Mean MSE: {np.mean(mlp_errors):.4f}")
        print(f"MLP KL Model Mean MSE: {np.mean(mlp_kl_errors):.4f}")
        print(f"Random Forest Model Mean MSE: {np.mean(rf_errors):.4f}")

    # Finalize and save the combined lambda vs MSE plot
    plt.xlabel("Lambda")
    plt.ylabel("Mean Squared Error")
    plt.title("Ridge Regression: Lambda vs MSE for Different PCA Components")
    plt.legend(title="PCA Components", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/combined_ridge_regression_lambda_vs_mse.png")
    plt.close()

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/pca_components_analysis.csv", index=False)
    print("\nResults have been saved to 'results/pca_components_analysis.csv'")

    # Plot MSE vs number of components for each model
    plt.figure(figsize=(12, 8))
    plt.plot(results_df["n_components"], results_df["baseline_mse"], label="Baseline")
    plt.plot(results_df["n_components"], results_df["ridge_mse"], label="Ridge")
    plt.plot(results_df["n_components"], results_df["mlp_mse"], label="MLP")
    plt.plot(results_df["n_components"], results_df["mlp_kl_mse"], label="MLP KL")
    plt.plot(results_df["n_components"], results_df["rf_mse"], label="Random Forest")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Mean Squared Error")
    plt.title("Model Performance vs Number of PCA Components")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/model_performance_vs_pca_components.png")
    plt.close()


if __name__ == "__main__":
    main()
