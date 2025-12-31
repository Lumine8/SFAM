import torch
import numpy as np

class GestureFM:
    """
    Gesture Feature Manager (Sequence-Ready):
    Transforms raw coordinate sequences into a time-series tensor
    compatible with LSTM/TemporalEncoder.
    
    Output Shape: (1, max_seq_len, 6) 
    Features per step: [vx, vy, ax, ay, jx, jy]
    """
    def __init__(self, max_seq_len=64, epsilon=1e-6):
        self.max_seq_len = max_seq_len
        self.epsilon = epsilon
        self.input_dim = 6  # (vx, vy, ax, ay, jx, jy)

    def process(self, raw_points: list):
        """
        Input: List of dicts [{'x': 10, 'y': 20, 't': 0.0}, ...]
        Output: Tensor of shape [1, max_seq_len, 6]
        """
        # 1. Validation
        if len(raw_points) < 4:
            # Return zero sequence if too short
            return torch.zeros(1, self.max_seq_len, self.input_dim)

        # 2. Convert to Numpy
        data = np.array([[p['x'], p['y'], p['t']] for p in raw_points], dtype=np.float32)
        pos = data[:, :2]  # (N, 2)
        time = data[:, 2]  # (N,)

        # 3. Differential Physics (Vectorized)
        # dt: (N-1,)
        dt = np.diff(time)
        dt = np.maximum(dt, self.epsilon)[:, None] # Reshape to (N-1, 1) for broadcasting

        # Velocity: (N-1, 2)
        delta_pos = np.diff(pos, axis=0)
        velocity = delta_pos / dt

        # Acceleration: (N-2, 2)
        # We must re-slice dt to match length of diff(velocity)
        delta_vel = np.diff(velocity, axis=0)
        acceleration = delta_vel / dt[1:] 

        # Jerk: (N-3, 2)
        delta_acc = np.diff(acceleration, axis=0)
        jerk = delta_acc / dt[2:]

        # 4. Alignment
        # Derivatives reduce length. We truncate everything to the shortest length (Jerk's length).
        # To align time, we take the *last* available metrics for V and A to match J's timeline.
        min_len = len(jerk)
        
        # Slice from the end to align timestamps
        v_aligned = velocity[-min_len:]      # (N-3, 2)
        a_aligned = acceleration[-min_len:]  # (N-3, 2)
        j_aligned = jerk                     # (N-3, 2)

        # 5. Stack Features: (Time, 6)
        # [vx, vy, ax, ay, jx, jy] at each timestep
        sequence = np.hstack([v_aligned, a_aligned, j_aligned])

        # 6. Normalize (Crucial for LSTMs)
        # Mouse coordinates can be 0-1920, velocities can be 5000+. 
        # Tanh prefers -1 to 1. Simple robust scaling:
        sequence = np.tanh(sequence * 0.01) 

        # 7. Pad/Truncate to Fixed Sequence Length
        processed_seq = self._resize_sequence(sequence)

        # Return Batch: (1, Seq_Len, 6)
        return torch.tensor(processed_seq, dtype=torch.float32).unsqueeze(0)

    def _resize_sequence(self, sequence):
        """
        Pads or truncates the sequence (Time dimension) to self.max_seq_len.
        """
        curr_len = sequence.shape[0]
        feat_dim = sequence.shape[1]
        
        if curr_len == self.max_seq_len:
            return sequence
            
        if curr_len > self.max_seq_len:
            # Truncate: Keep the middle (most active part of gesture)
            start = (curr_len - self.max_seq_len) // 2
            return sequence[start : start + self.max_seq_len, :]
            
        if curr_len < self.max_seq_len:
            # Pad: Zero padding at the END
            pad_len = self.max_seq_len - curr_len
            # Pad format for np.pad: ((top, bottom), (left, right))
            return np.pad(sequence, ((0, pad_len), (0, 0)), mode='constant')

# Expose instance
processor = GestureFM()