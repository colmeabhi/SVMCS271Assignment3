import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_data():
    # Malware samples (first 20 for training, next 20 for testing)
    malware_data = np.array([
        [-2.5502, 0.458, 0.3097], [-2.4916, 0.398, 0.8671], [-2.4591, 0.381, 0.2878], [-2.3937, 0.387, 0.3369],
        [-2.5805, 0.412, 0.3344], [-2.4426, 0.944, 0.2908], [-2.5148, 0.989, 0.2814], [-2.4417, 0.402, 0.3266],
        [-2.4508, 0.490, 0.3223], [-12.4561, 0.479, 0.7914], [-2.5332, 0.491, 0.4302], [-2.4849, 0.488, 0.3293],
        [-2.6171, 0.455, 0.8409], [-2.5150, 0.432, 0.3612], [-14.4404, 0.408, 0.2755], [-2.5892, 0.425, 0.3998],
        [-2.4532, 0.421, 0.3486], [-2.4831, 1.397, 0.3550], [-2.5505, 0.368, 0.3432], [-2.4479, 0.426, 0.3346],
        [-2.4743, 0.383, 0.3506], [-2.5167, 0.466, 0.3861], [-2.5318, 0.458, 0.3588], [-2.3913, 0.437, 0.3407],
        [-2.6346, 0.436, 0.4723], [-2.5553, 0.460, 0.3527], [-2.5426, 0.477, 0.3269], [-2.3792, 0.431, 0.3136],
        [-13.5807, 1.408, 0.3260], [-2.5571, 0.429, 0.3729], [-2.5179, 0.355, 0.3303], [-2.5161, 0.398, 0.3033],
        [-2.6699, 0.478, 0.3950], [-2.4019, 0.397, 0.3360], [-2.4906, 0.447, 0.3694], [-2.5358, 0.929, 0.9425],
        [-12.5585, 0.419, 0.3946], [-2.3902, 0.478, 0.3299], [-2.5675, 0.416, 0.3566], [-2.4462, 0.400, 0.3332]
    ])
    
    # Benign samples (first 20 for training, next 20 for testing)
    benign_data = np.array([
        [-20.1718, 0.930, 0.6909], [-13.8231, 0.854, 0.7998], [-12.2302, 0.928, 0.7324], [-23.7316, 0.924, 0.7543],
        [-9.4449, 0.801, 0.6843], [-33.5896, 0.917, 0.7021], [-148.4577, 0.908, 0.8879], [-11.9680, 0.916, 0.7166],
        [-8.0129, 0.930, 0.6830], [-14.7196, 0.979, 0.7142], [-12.9691, 0.927, 0.6771], [-35.6650, 0.882, 0.6901],
        [-14.8911, 0.972, 0.8415], [-33.0356, 0.865, 0.7811], [-14.0974, 0.827, 0.6921], [-12.8733, 0.953, 0.7454],
        [-16.8113, 0.870, 0.6873], [-30.8435, 0.915, 0.8512], [-9.0773, 0.938, 0.7999], [-22.3555, 0.848, 0.7783],
        [-21.6937, 0.858, 0.7068], [-14.2945, 0.906, 0.6834], [-21.9569, 0.999, 0.7130], [-27.6297, 0.863, 0.7892],
        [-19.0987, 0.927, 0.7036], [-22.5225, 0.940, 0.7543], [-31.2162, 0.908, 0.7014], [-148.9674, 0.993, 0.8182],
        [-42.8055, 1.036, 0.9166], [-51.2141, 0.949, 0.7801], [-21.3982, 1.002, 0.8206], [-17.8242, 0.992, 0.8215],
        [-169.1587, 1.018, 0.9209], [-45.4216, 1.059, 0.8323], [-22.7345, 0.858, 0.7568], [-15.0389, 0.857, 0.6616],
        [-13.6486, 0.851, 0.6433], [-14.1127, 0.848, 0.6434], [-15.7107, 0.875, 0.6644], [-33.7041, 1.037, 0.7858]
    ])
    
    # Split into training and testing
    X_train = np.vstack([malware_data[:20], benign_data[:20]])
    y_train = np.hstack([np.ones(20), -np.ones(20)])
    
    X_test = np.vstack([malware_data[20:], benign_data[20:]])
    y_test = np.hstack([np.ones(20), -np.ones(20)])
    
    return X_train, y_train, X_test, y_test

def rfe_svm():
    X_train, y_train, X_test, y_test = load_data()
    feature_names = ['HMM', 'SSD', 'OGS']
    
    print("Problem 4.12")
    print("=" * 50)
    
    # Part (a): Train with all features
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    weights = svm.coef_[0]
    
    print(f"All features - Accuracy: {accuracy:.3f}")
    print(f"Weights: {dict(zip(feature_names, weights))}")
    print()
    
    # Part (b): RFE process
    current_features = list(range(3))
    current_names = feature_names.copy()
    
    for step in range(2):  # Remove 2 features (3->2->1)
        # Find feature with smallest absolute weight
        abs_weights = np.abs(weights)
        min_idx = np.argmin(abs_weights)
        removed_feature = current_names[min_idx]
        
        # Remove feature
        current_features.pop(min_idx)
        current_names.pop(min_idx)
        
        # Retrain with remaining features
        X_train_reduced = X_train[:, current_features]
        X_test_reduced = X_test[:, current_features]
        
        svm = SVC(kernel='linear', C=1.0)
        svm.fit(X_train_reduced, y_train)
        y_pred = svm.predict(X_test_reduced)
        accuracy = accuracy_score(y_test, y_pred)
        weights = svm.coef_[0]
        
        print(f"Removed {removed_feature} - Accuracy: {accuracy:.3f}")
        print(f"Remaining weights: {dict(zip(current_names, weights))}")
        print()

if __name__ == "__main__":
    rfe_svm()