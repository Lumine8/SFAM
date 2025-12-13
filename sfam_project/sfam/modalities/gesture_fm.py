import torch
import numpy as np

class GestureFM:
    """
    Gesture Feature Manager:
    Transforms raw coordinate sequences (x, y, timestamp) into a 
    behavioral feature vector using Differential Physics.
    
    Extracts:
    1. Velocity (Speed + Direction)
    2. Acceleration (Force/Control)
    3. Jerk (Smoothness/Tremor)
    """
    def __init__(self, output_dim=64, epsilon=1e-6):
        self.output_dim = output_dim
        self.epsilon = epsilon  # Prevents division by zero

    def process(self, raw_points: list):
        """
        Input: List of dicts [{'x': 10, 'y': 20, 't': 0.0}, ...]
        Output: Tensor of shape [1, output_dim]
        """
        if len(raw_points) < 4:
            # Need at least 4 points to calculate Jerk
            return torch.zeros(1, self.output_dim)

        # 1. Convert to Numpy [N, 3] -> (x, y, t)
        data = np.array([[p['x'], p['y'], p['t']] for p in raw_points], dtype=np.float32)

        # 2. Extract Components
        pos = data[:, :2]  # x, y
        time = data[:, 2]  # t

        # 3. Calculate Differential Physics
        # delta_t between steps
        dt = np.diff(time)
        dt = np.maximum(dt, self.epsilon)  # Clamp min time to avoid Inf

        # Velocity: v = dx / dt
        delta_pos = np.diff(pos, axis=0)
        velocity = delta_pos / dt[:, None]

        # Acceleration: a = dv / dt
        # Note: We lose 1 point of length with each diff, so we slice dt
        delta_vel = np.diff(velocity, axis=0)
        acceleration = delta_vel / dt[1:, None]

        # Jerk: j = da / dt (Measures smoothness/shake)
        delta_acc = np.diff(acceleration, axis=0)
        jerk = delta_acc / dt[2:, None]

        # 4. Feature Fusion
        # We calculate magnitude for each to get a single 1D scalar profile per metric
        # v_mag = sqrt(vx^2 + vy^2)
        v_mag = np.linalg.norm(velocity, axis=1)
        a_mag = np.linalg.norm(acceleration, axis=1)
        j_mag = np.linalg.norm(jerk, axis=1)

        # Concatenate all features into one long sequence
        combined_features = np.concatenate([v_mag, a_mag, j_mag])

        # 5. Fixed-Size Projection (Padding or Truncating)
        processed_vector = self._resize_sequence(combined_features, self.output_dim)

        # Return as Batch Tensor
        return torch.tensor(processed_vector, dtype=torch.float32).unsqueeze(0)

    def _resize_sequence(self, sequence, target_len):
        """
        Ensures the output vector is exactly 'target_len' size.
        """
        current_len = len(sequence)
        
        if current_len == target_len:
            return sequence
        
        if current_len > target_len:
            # Truncate: Take the middle section (usually most representative)
            start = (current_len - target_len) // 2
            return sequence[start : start + target_len]
        
        if current_len < target_len:
            # Pad: Zero padding at the end
            return np.pad(sequence, (0, target_len - current_len), mode='constant')

# Expose a default instance
processor = GestureFM()