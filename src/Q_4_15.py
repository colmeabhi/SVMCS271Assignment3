import numpy as np
import matplotlib.pyplot as plt

def f_score(X, z, lam, b, x):
    """Compute scoring function f(x)"""
    return sum(lam[i] * z[i] * np.dot(X[i], x) for i in range(len(X))) + b

def update_pair(X, z, lam, b, i, j, C, eps):
    """Update lambda_i, lambda_j pair"""
    d = 2 * np.dot(X[i], X[j]) - np.dot(X[i], X[i]) - np.dot(X[j], X[j])
    
    if abs(d) <= eps:
        return lam, b, False
    
    # Compute errors
    E_i = f_score(X, z, lam, b, X[i]) - z[i]
    E_j = f_score(X, z, lam, b, X[j]) - z[j]
    
    # Save old values
    lam_i_old = lam[i]
    lam_j_old = lam[j]
    
    # Update lambda_j
    lam[j] = lam_j_old - (z[j] * (E_i - E_j)) / d
    
    # Compute bounds
    if z[i] == z[j]:
        L = max(0, lam[i] + lam[j] - C)
        H = min(C, lam[i] + lam[j])
    else:
        L = max(0, lam[j] - lam[i])
        H = min(C, C + lam[j] - lam[i])
    
    # Clip lambda_j
    if lam[j] > H:
        lam[j] = H
    elif lam[j] < L:
        lam[j] = L
    
    # Update lambda_i
    lam[i] = lam_i_old + z[i] * z[j] * (lam_j_old - lam[j])
    
    # Update b
    b_i = b - E_i - z[i] * (lam[i] - lam_i_old) * np.dot(X[i], X[i]) - z[j] * (lam[j] - lam_j_old) * np.dot(X[i], X[j])
    b_j = b - E_j - z[i] * (lam[i] - lam_i_old) * np.dot(X[i], X[j]) - z[j] * (lam[j] - lam_j_old) * np.dot(X[j], X[j])
    
    if 0 < lam[i] < C:
        b = b_i
    elif 0 < lam[j] < C:
        b = b_j
    else:
        b = (b_i + b_j) / 2
    
    changed = abs(lam[i] - lam_i_old) > eps or abs(lam[j] - lam_j_old) > eps
    return lam, b, changed

def generate_pairs(n):
    """Generate pairs as specified in problem"""
    pairs = []
    # Forward pairs
    for r in range(1, n):
        for i in range(n - r):
            pairs.append((i, i + r))
    # Backward pairs
    for r in range(1, n):
        for i in range(r, n):
            pairs.append((i, i - r))
    return pairs

def ssmo_systematic(X, z, C=2.5, eps=1e-5):
    """SSMO with systematic pair selection"""
    n = len(X)
    lam = np.zeros(n)
    b = 0.0
    
    pairs = generate_pairs(n)
    
    for iteration in range(10):  # Max 10 iterations as specified
        changed = False
        for i, j in pairs:
            lam, b, pair_changed = update_pair(X, z, lam, b, i, j, C, eps)
            if pair_changed:
                changed = True
        if not changed:
            break
    
    return lam, b

def ssmo_random(X, z, C=2.5, eps=1e-5):
    """SSMO with random pair selection"""
    n = len(X)
    lam = np.zeros(n)
    b = 0.0
    
    np.random.seed(42)
    for _ in range(1000):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i != j:
            lam, b, _ = update_pair(X, z, lam, b, i, j, C, eps)
    
    return lam, b

def get_weights(X, z, lam):
    """Get linear weights w1, w2"""
    w = np.zeros(2)
    for i in range(len(X)):
        w += lam[i] * z[i] * X[i]
    return w

def plot_result(X, z, lam, b, title):
    """Plot data points and separating hyperplane"""
    plt.figure(figsize=(8, 6))
    
    # Plot points
    for i in range(len(X)):
        if z[i] == 1:
            plt.scatter(X[i][0], X[i][1], marker='s', s=100, facecolors='none', 
                       edgecolors='blue', label='+1' if i == 0 else "")
        else:
            plt.scatter(X[i][0], X[i][1], marker='o', s=100, color='red', 
                       label='-1' if i == 3 else "")
    
    # Plot hyperplane
    w = get_weights(X, z, lam)
    if abs(w[1]) > 1e-6:
        x_line = np.linspace(0, 4, 100)
        y_line = -(w[0] * x_line + b) / w[1]
        plt.plot(x_line, y_line, 'g-', label='Separating hyperplane')
    
    plt.xlim(0, 4)
    plt.ylim(0, 5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def solve_problem_4_15():
    # Training data from problem
    X = np.array([[3, 3], [3, 4], [2, 3], [1, 1], [1, 3], [2, 2]])
    z = np.array([1, 1, 1, -1, -1, -1])
    
    print("Problem 4.15: SSMO Algorithm")
    print("=" * 40)
    
    # Part (a): Systematic pair selection
    lam_a, b_a = ssmo_systematic(X, z, C=2.5, eps=1e-5)
    w_a = get_weights(X, z, lam_a)
    
    print("Part (a) - Systematic selection:")
    print(f"Lambda values: {lam_a}")
    print(f"b: {b_a:.6f}")
    print(f"Weights: w1={w_a[0]:.6f}, w2={w_a[1]:.6f}")
    print()
    
    # Part (b): Random pair selection
    lam_b, b_b = ssmo_random(X, z, C=2.5, eps=1e-5)
    w_b = get_weights(X, z, lam_b)
    
    print("Part (b) - Random selection:")
    print(f"Lambda values: {lam_b}")
    print(f"b: {b_b:.6f}")
    print(f"Weights: w1={w_b[0]:.6f}, w2={w_b[1]:.6f}")
    print()
    
    # Part (c): Plot results
    plot_result(X, z, lam_a, b_a, "Part (a): Systematic Pair Selection")
    plot_result(X, z, lam_b, b_b, "Part (b): Random Pair Selection")

if __name__ == "__main__":
    solve_problem_4_15()