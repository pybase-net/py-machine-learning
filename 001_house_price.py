import numpy as np
import matplotlib.pyplot as plt

# === Replace this with the actual function ===


def compute_cost(x, y, w, b):
    m = x.shape[0]
    f_wb = w * x + b
    cost = (1 / (2 * m)) * np.sum((f_wb - y) ** 2)
    return cost


# === Load dataset ===
data = np.loadtxt('./001_house_price/dataset.csv', delimiter=',', skiprows=1)
x_train = data[:, :-1].flatten()  # Ensure 1D
y_train = data[:, -1]

# 🔍 Narrow and high-precision search range
w_values = np.arange(0, 2, 0.001)
b_values = np.arange(0, 2, 0.001)

min_cost = float('inf')
best_w, best_b = 0, 0

for w in w_values:
    for b in b_values:
        cost = compute_cost(x_train, y_train, w, b)
        if cost < min_cost:
            min_cost = cost
            best_w, best_b = w, b
# Print best parameters
print(f"✅ Best w: {best_w:.2f}, Best b: {best_b:.2f}, Cost: {min_cost:.2f}")

b = best_b
w = best_w

# === Compute model predictions ===
f_wb = w * x_train + b

# === Compute cost array for w-range ===
w_values = np.arange(0, 405, 5)
cost_values = np.array([compute_cost(x_train, y_train, tmp_w, b)
                       for tmp_w in w_values])
highlight_cost = compute_cost(x_train, y_train, w, b)

# === Start plotting ===
fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# === LEFT: Prediction vs Actual ===
ax[0].scatter(x_train, y_train, marker='x', color='red', label='Actual Value')
ax[0].plot(x_train, f_wb, color='deepskyblue',
           linewidth=3, label='Our Prediction')

# Cost lines for each point
for xi, yi in zip(x_train, y_train):
    prediction = w * xi + b
    cost_i = ((prediction - yi) ** 2) / 2
    ax[0].vlines(xi, yi, prediction, linestyle='dotted', color='purple',
                 lw=2, label='cost for point' if xi == x_train[0] else "")
    ax[0].annotate(f"{cost_i:.0f}", xy=(xi, yi + (prediction - yi) / 2),
                   color='purple', textcoords='offset points', xytext=(5, 0))

ax[0].set_title("Housing Prices")
ax[0].set_xlabel("Size (m2)")
ax[0].set_ylabel("Price (in billions VND)")
ax[0].legend()

# Cost formula text
ctot = compute_cost(x_train, y_train, w, b)
ax[0].text(
    0.15, 0.02, f"cost = (1/m)*(... ) = {ctot:.0f}", transform=ax[0].transAxes, color='purple')

# === RIGHT: Cost vs w ===
ax[1].plot(w_values, cost_values, c='deepskyblue', linewidth=4)
ax[1].scatter(w, highlight_cost, color='red', s=100, label=f'cost at w={w}')
ax[1].hlines(highlight_cost, 0, w, color='purple',
             linestyle='dotted', linewidth=3)
ax[1].vlines(w, 0, highlight_cost, color='purple',
             linestyle='dotted', linewidth=3)
ax[1].set_title(f"Cost vs. w, (b fixed at {b})")
ax[1].set_xlabel("w")
ax[1].set_ylabel("Cost")
ax[1].legend()

fig.suptitle(
    f"Minimize Cost: Current Cost = {highlight_cost:.0f}", fontsize=14)


plt.show()

#  Best w: 0.28, Best b: 0.24, Cost: 0.91

# -----------------------------
# Step 5: Use model to predict
# -----------------------------


def predict(x, w, b):
    """Make predictions."""
    return w * x + b


def predict_house_price(area_m2):
    return predict(area_m2, best_w, best_b)


def predict_house_square(area_billion_vnd):
    """Predict the area in m² given the price in billions VND."""
    return (area_billion_vnd - best_b) / best_w


area = 100  # e.g., 100 m²
predicted_price = predict_house_price(area)
print(
    f"Predicted price for {area} m²: {predicted_price:.2f} (in billions VND)")

budget = 5  # e.g., 5 billion VND
predicted_area = predict_house_square(budget)
print(
    f"Predicted area for {budget} billion VND: {predicted_area:.2f} m²")
