import pandas as pd
import datatable as dt
from datatable import sum, f, by, join, ifelse, log, update
from datatable.math import isna
import time
import datetime
import calendar


def load_cdnow():
    data = pd.read_csv('utils/data/CDNOW_master.txt', delim_whitespace=True, header=None)
    data.columns = ['cid', 'date', 'count', 'sales']
    return data


def caltocbs(elog, days_in_period=1, calibration_end="", observation_end=""):
    """
    Convert Event Log to customer-level summary statistic

    DataTable implementation for the conversion of an event log into a
    customer-by-sufficient-statistic (CBS) dataframe, with a row for each
    customer

    Holdout period is defined as the period between calibration_end and
    observation_end

    Parameters
    ----------
    elog: :obj: DataFrame
        a Pandas DataFrame that contains the customer_id col and the datetime col.
        cid: string
            the column in transactions DataFrame that denotes the customer_id
        date:  string
            the column in transactions that denotes the datetime the purchase was made.
        sales: string
            the columns in the transactions that denotes the monetary value of the transaction.
    calibration_end: string
         a string or datetime to denote the final date of the calibration period in the format "%Y-%m-%d".
         If not given, defaults to the max 'datetime_col'.
    observation_end: string
        a string or datetime to denote the final date of the observation period in the format "%Y-%m-%d".
        Events after this date are truncated. If not given, defaults to the max 'datetime_col'.
    days_in_period: int, optional
        Time unit of aggregation. Events that occur within the same time period are aggregated and their sales summed
        Default: '1' for daily aggregation.
    Returns
    -------
    :obj: DataFrame:
        cid: string
            column containing customer id
        first: datetime
            column containing date of customer's first event
        t_x: int
            column containing number periods between a customer's first and last event in the calibration period
        sales: double
            column containing total customer sales in the calibration period
        litt: double
            column containing sum of the logarithm of a customer's inter-event timings within the calibration period
        x: int
            column containing number periods in which a customer had a event in the calibration period
        x_star: int
            column containing number periods in which a customer had a event in the holdout period
        sales_star: double
            column containing total customer sales in the holdout period
        t_star: double
            number of periods in the holdout period
    """

    calibration_end_ts = calendar.timegm(time.strptime(calibration_end, "%Y-%m-%d"))
    observation_end_ts = calendar.timegm(time.strptime(observation_end, "%Y-%m-%d"))

    elog['date'] = pd.to_datetime(elog['date'], format='%Y%m%d').astype(int) / 10 ** 9
    mult = 86400/days_in_period
    elog_dt = dt.Frame(elog)

    elog_dt = elog_dt[f.date <= observation_end_ts, :]
    elog_dt_train = elog_dt[f.date > calibration_end_ts, :]
    elog_dt = elog_dt[f.date <= calibration_end_ts, :]

    elog_dt = elog_dt[:, sum(f.sales), by("cid", "date")]
    elog_dt[:, update(first=dt.min(f.date)), by(f.cid)]
    elog_dt = elog_dt[:, {'t': f.date - f.first}, by("cid", "date", "sales", "first")]
    elog_dt = elog_dt[:, f[:].extend({"prev_value": dt.shift(f.t)}), by("cid")]
    elog_dt['prev_value'] = ifelse(isna(f.prev_value), 0, f.prev_value)
    elog_dt = elog_dt[:, {'itt': f.t - f.prev_value}, by("cid", "date", "sales", "first", "t")]
    elog_dt['litt'] = ifelse(f.itt == 0, 0, log(f.itt / mult))
    elog_dt[:, update(t_x=dt.max(f.t)/mult), by(f.cid)]
    elog_dt[:, update(sales=dt.sum(f.sales)), by(f.cid)]
    elog_dt[:, update(litt=sum(f.litt)), by(f.cid)]

    elog_dt_train[:, update(x_star=dt.count()), by(f.cid)]
    elog_dt_train = elog_dt_train[:, sum(f.sales), by(f.cid, f.x_star)]
    elog_dt_train['sales_star'] = elog_dt_train['sales']
    elog_dt_train = elog_dt_train[:, ["cid", "x_star", "sales_star"]]
    elog_dt_train.key = "cid"
    cbs = elog_dt[:, {"x": dt.count() - 1}, by("cid", "first", "t_x", "sales", "litt")]
    cbs = cbs[:, :, join(elog_dt_train)]
    cbs['x_star'] = ifelse(isna(f.x_star), 0, f.x_star)
    cbs['sales_star'] = ifelse(isna(f.sales_star), 0, f.sales_star)
    cbs['t_star'] = (observation_end_ts - calibration_end_ts) / mult
    cbs = cbs.to_pandas()
    cbs['first'] = cbs['first'].map(lambda x: datetime.datetime.fromtimestamp(x))
    return cbs
