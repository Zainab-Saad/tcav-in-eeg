import numpy as np
from scipy.interpolate import CubicSpline


def generate_random_curve(signal_len, sigma=0.2, knot=4, random_state=0):
    # 0.0 endpoint and len(signal_len)-1 endpoint
    # And knot points in between
    x = np.arange(0, signal_len, (signal_len - 1) / (knot + 1))
    random_generator = np.random.default_rng(random_state)
    y = random_generator.normal(loc=1.0, scale=sigma, size=(knot + 2))
    cs = CubicSpline(x, y)
    rc = cs(np.arange(signal_len))
    return rc


def mag_warp(signal_data, sigma=0.2, knot=4, random_state=0, randomize_per_channel=False):
    new_signal_data = []
    for idx, channel_data in enumerate(signal_data):
        if randomize_per_channel:
            random_state += 1
        new_channel_data = channel_data * generate_random_curve(len(channel_data), sigma=sigma, knot=knot,
                                                                random_state=random_state)
        new_signal_data.append(new_channel_data)
    return np.array(new_signal_data, dtype="float32")


def distort_timeseries(signal_len, sigma=0.2, knot=4, random_state=0):
    t = generate_random_curve(signal_len, sigma=sigma, knot=knot, random_state=random_state)
    t_cum = np.cumsum(t)
    # Make the last value to have signal_len
    t_scale = (signal_len - 1) / t_cum[-1]
    t_cum = t_cum * t_scale
    return t_cum


def time_warp(signal_data, sigma=0.2, knot=4, random_state=0, randomize_per_channel=False):
    new_signal_data = []
    for channel_data in signal_data:
        t_new = distort_timeseries(len(channel_data), sigma=sigma, knot=knot, random_state=random_state)
        new_channel_data = np.interp(np.arange(len(channel_data)), t_new, channel_data)
        new_signal_data.append(new_channel_data)
        if randomize_per_channel:
            random_state += 1
    return np.array(new_signal_data, dtype="float32")


def add_noise(signal_data, sigma=0.02, random_state=0, randomize_per_channel=False):
    if randomize_per_channel:
        for idx, _ in enumerate(signal_data):
            random_generator = np.random.default_rng(random_state)
            noise = random_generator.normal(loc=1, scale=sigma, size=signal_data.shape[1])
            signal_data[idx] *= noise
            random_state += 1
        return signal_data
    random_generator = np.random.default_rng(random_state)
    noise = random_generator.normal(loc=0, scale=sigma, size=signal_data.shape)
    return signal_data * noise
