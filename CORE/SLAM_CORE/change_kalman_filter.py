import numpy as np

# Define the measurement model for line feature
def line_feature_measurement(x, y, a, b, c):
    return np.abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

# Initialize state, covariance, process noise, and measurement noise
n_states = 6  # Example state size, adjust according to your problem
n_measurements = 2  # Example measurement size, adjust according to your problem

x = np.zeros((n_states, 1))  # State vector
P = np.eye(n_states)  # Covariance matrix
Q = np.eye(n_states) * 0.01  # Process noise covariance
R = np.eye(n_measurements) * 0.1  # Measurement noise covariance

# Measurement matrix (H) for point measurements
H_point = np.array([[1, 0, 0, 0, 0, 0],  # Measure x
                    [0, 1, 0, 0, 0, 0]])  # Measure y

# Measurement matrix (H) for line measurements (extend as needed)
H_line = np.array([[0, 0, 1, 1, 1, 0]])  # Example, adjust according to your state

# Kalman filter update step incorporating both point and line features
def kalman_filter_update(x, P, z_point, z_line, H_point, H_line, R):
    # Predict step
    x_pred = x  # Assuming no state transition for simplicity
    P_pred = P + Q
    
    # Point measurement update
    y_point = z_point - H_point @ x_pred
    S_point = H_point @ P_pred @ H_point.T + R[:2, :2]
    K_point = P_pred @ H_point.T @ np.linalg.inv(S_point)
    x = x_pred + K_point @ y_point
    P = (np.eye(n_states) - K_point @ H_point) @ P_pred
    
    print(x[:,0].shape)
    # Line measurement update
    y_line = z_line - line_feature_measurement(x[0,0], x[1,0], x[2,0], x[3,0], x[4,0])
    H_line_extended = np.array([[x[2,0] / np.sqrt(x[2,0]**2 + x[3,0]**2),  # Partial derivative w.r.t x
                                 x[3,0] / np.sqrt(x[2,0]**2 + x[3,0]**2),  # Partial derivative w.r.t y
                                 x[0,0] / np.sqrt(x[2,0]**2 + x[3,0]**2),  # Partial derivative w.r.t a
                                 x[1,0] / np.sqrt(x[2,0]**2 + x[3,0]**2),  # Partial derivative w.r.t b
                                 1 / np.sqrt(x[2,0]**2 + x[3,0]**2),     # Partial derivative w.r.t c
                                 0]])  # Adjust if you have more states

    S_line = H_line_extended @ P @ H_line_extended.T + R[2:, 2:]
    K_line = P @ H_line_extended.T @ np.linalg.inv(S_line)
    x = x + K_line @ y_line
    P = (np.eye(n_states) - K_line @ H_line_extended) @ P
    
    return x, P

# Example measurements
z_point = np.array([[2.0], [3.0]])  # Point measurement
z_line = np.array([[1.0]])  # Line measurement

# Run Kalman filter update
x, P = kalman_filter_update(x, P, z_point, z_line, H_point, H_line, R)

print("Updated state:", x)
print("Updated covariance:", P)

