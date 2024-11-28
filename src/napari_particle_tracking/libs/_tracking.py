import numpy as np
import pandas as pd
import trackpy as tp


def track(
    points: pd.DataFrame,
    search_range: float = 2,
    memory: int = 0,
    adaptive_stop: float = 0.95,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Track the points.

    Parameters
    ----------
    points : pd.DataFrame
        The points to be tracked.
    search_range : float
        The search range for the tracking.
    memory : int
        The memory for the tracking.
    show_progress : bool
        Whether to show the progress of the tracking.

    Returns
    -------
    pd.DataFrame
        The tracked points.
    """
    tp.quiet(not show_progress)
    result = tp.link_df(
        points,
        search_range=search_range,
        memory=memory,
        adaptive_stop=adaptive_stop,
    )
    result.rename(columns={"particle": "track_id"}, inplace=True)
    # make sure track_id starts from 1
    result["track_id"] = result["track_id"] - result["track_id"].min() + 1
    return result


def histogram(data, binsize=5):
    try:
        data = np.array(data).ravel()
        if len(data) > 1:
            data = data[~np.isnan(data)]
        vmin = np.min(data)
        vmax = np.max(data)
        # if abs(vmax - vmin) <= binsize:
        #     binsize = 1 if np.std(data) == 0 else np.std(data)
        if vmin == vmax:
            vmax = vmin + 1
        bins = list(np.arange(start=vmin, stop=vmax, step=binsize))
        bins.append(bins[-1] + binsize)
    except Exception as err:
        # print(f"vmin {vmin}, vmax {vmax}, binsize = {binsize}")
        print(f"{err=}, {type(err)=}")
        raise

    hist, edges = np.histogram(data, bins=bins)
    return hist, edges, binsize


from scipy.optimize import curve_fit

# add straight line fit and r2 calculation
# https://github.com/elilouise/Scientific-Computational-Records/blob/main/Fitting%20data%20to%20polynomials.ipynb

def msd(pos, limit=25):
    pos_columns = ["x", "y"]
    if pos.shape[1] != 2:
        raise ValueError("pos should have 2 columns")
    if pos.shape[0] < 2:
        raise ValueError("pos should have at least 2 rows")

    if pos.ndim > 2:
        pos_columns = ["x", "y", "z"]

    result_columns = [f"<{p}>" for p in pos_columns] + [
        f"<{p}^2>" for p in pos_columns
    ]
    limit = min(limit, len(pos[:, 0]) - 1)
    lagtimes = np.arange(1, limit + 1)
    msd_list = []
    for lt in lagtimes:
        diff = pos[lt:] - pos[:-lt]
        msd_list.append(
            np.concatenate(
                (np.nanmean(diff, axis=0), np.nanmean(diff**2, axis=0))
            )
        )
    result = pd.DataFrame(msd_list, columns=result_columns, index=lagtimes)
    result["msd"] = result[result_columns[-len(pos_columns) :]].sum(1)
    return result["msd"]


def msd_fit_function(delta, d, alfa):
    return (4 * d) * np.power(delta, alfa)

def line(x, m, c):
    return m * x + c

# def _get_goodness_of_fit(msd, delay_time=50.0):
#     # goodness of fit https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/regression-and-correlation/coefficient-of-determination-r-squared.html
#     # straight line fit
#     _msd = msd
#     _time_delay = np.arange(0, len(_msd)) *delay_time
#     pp, residuals, _, _, _ = np.polyfit(_time_delay, _msd, 1, full=True)
#     slop, intercept = pp
#     _straing_line_fit = slop * _time_delay + intercept
    
#     y_dash = np.average(_msd)
#     total_some_of_square = np.sum((_msd - y_dash) ** 2)

#     r_2 = 1 - (residuals[0]/total_some_of_square)

#     adjusted_r_2 = 1 - (1 - r_2) * (len(_msd) - 1) / (len(_msd) - len(pp) - 1)

#     return r_2,  adjusted_r_2, _straing_line_fit, _time_delay

def msd_goodness_of_fit(actual_y, predicted_y, fiting_paramets =2):
    residual = actual_y - predicted_y
    residual_some_of_square = np.sum(residual ** 2)
    
    y_dash = np.average(actual_y)
    total_some_of_square = np.sum((actual_y - y_dash) ** 2)

    r_2 = 1 - (residual_some_of_square/total_some_of_square)
    adjusted_r_2 = 1 - (1 - r_2) * (len(actual_y) - 1) / (len(actual_y) - fiting_paramets)

    return r_2, adjusted_r_2

def basic_msd_fit(
    msd_y, delta=3.8, fit_function=msd_fit_function, maxfev=1000000
):
    y = np.array(msd_y)
    x = np.array(list(range(1, len(y) + 1))) * delta

    init = np.array([0.001, 0.01])
    best_value, _ = curve_fit(fit_function, x, y, p0=init, maxfev=maxfev)
    _y = msd_fit_function(x, best_value[0], best_value[1])

    # r2 : goodness of fit for the msd and _y 
    r_2, adjusted_r_2 = msd_goodness_of_fit(y, _y)
    

    return pd.DataFrame({"alpha": [best_value[1]]* len(y), 
                         "fit":_y, 
                         "diffusion_coefficient":best_value[0], 
                         "r_2": r_2, 
                         "adjusted_r_2":adjusted_r_2}, index=x)
