"""Generate and save the contours with a moving peak."""
import numpy as np
import pickle


def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y) -> np.ndarray:
    """Function that makes a 2d gaussian and returns the rewardcontour as a 2d numpy  array"""
    return np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))


def generate_contours():
    """Generate the contours for the experiment."""
    # Define the grid dimensions
    grid_size = 1000
    x = np.linspace(0, 999, grid_size)
    y = np.linspace(0, 999, grid_size)
    x, y = np.meshgrid(x, y)

    # Parametrize the circular motion of the gaussian peak
    t_period = 100                                          # period/number of timesteps of the rotaton
    angle = np.linspace(0, 2*np.pi, num=t_period)           # angular location for each peak
    r = 250                                                 # radius from the center of the grid to the peak
    xs, ys = r*np.cos(angle) + 500, r*np.sin(angle) + 500   # get the xs and ys of the peak
    sigma_x, sigma_y = 100, 100                             # standard deviations

    # Create the contours and store them in a list
    contours = []
    for x_peak, y_peak in zip(xs, ys):
        z = gaussian_2d(x, y, x_peak, y_peak, sigma_x, sigma_y)
        z = (z * 500) / np.max(z)  # scale so that the peak is 500
        contours.append(z)

    with open(f'/n/ramanathan_lab/aboesky/reward_contours/skinnycontours_{t_period}.pkl', 'wb') as f:
        pickle.dump(contours, f)


if __name__ == '__main__':
    generate_contours()
