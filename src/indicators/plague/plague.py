import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import List, Union

import pandas as pd


def max_week(year):
    """
    Check the max week in a year to prevent
    date errors in fromisocalendar() when a
    week 53 doesn't exist.

    Based off this post:
    https://stackoverflow.com/questions/60945041/setting-more-than-52-weeks-in-python-from-a-date
    """
    has_week_53 = date.fromisocalendar(year, 52, 1) + timedelta(
        days=7
    ) != date.fromisocalendar(year + 1, 1, 1)
    if has_week_53:
        return 53
    else:
        return 52


plague_raw_dir = os.path.join(
    os.getenv("AA_DATA_DIR"), "private", "raw", "mdg", "institut_pasteur"
)


def latest_plague_path(dir: Union[Path, str] = plague_raw_dir):
    """Find latest plague data


    Parameters
    ----------
    dir : Union[Path, str], optional
        Data directory with raw plague data.

    Returns
    -------
    Path, str
    """
    files = os.listdir(dir)
    date_max = date.fromisoformat("1900-01-01")
    path_max = None
    str_max = None
    for str in files:
        date_str = re.search(r"\d{4}-\d{2}-\d{2}", str)
        if date_str is not None:
            dt = date.fromisoformat(date_str.group())
            if dt > date_max:
                date_max = dt
                path_max = str
                str_max = date_str
    return path_max, str_max


def load_plague_data(
    path: Union[Path, str],
    keep_cases: Union[List[str], None] = ["PROB", "CONF"],
    delimiter: str = ";",
):
    """Load plague data

    Loads plague data and does some minor processing
    and corrections. Can filter the data to only
    cases of interest. Defaults to only keeping
    probable and confirmed.

    Parameters
    ----------
    path : Union[Path, str]
        Path to the data file.
    keep_cases : Union[List[str], None], optional
        List of cases to keep. Valid cases are
        'SUSP', 'PROB', and 'CONF'. If ``None``,
        keeps all.
    delimiter : str, optional
        Passed to  ``pandas.read_csv()``, defaults
        to ';'.

    Returns
    -------
    ``pandas.DataFrame``

    """
    # print("heh")
    if path.suffix == ".csv":
        df = pd.read_csv(path, delimiter=delimiter)
    elif path.suffix == ".xls":
        df = pd.read_excel(path)
    else:
        raise ValueError(
            "Path doesn't have a valid extension. "
            "Should either be .csv or .xls"
        )
    # adjust column names
    df.columns = df.columns.str.lower()
    df.rename(columns={"mdg_com_code": "ADM3_PCODE"}, inplace=True)
    # to make pcodes correspond with shp file
    df.ADM3_PCODE = df.ADM3_PCODE.str.replace("MDG", "MG")

    # create a datetime from the year and week as this is easier with plotting
    # first, make sure no invalid iso weeks (sometimes had week as 53 when max
    # iso weeks in a year were 52)
    df["max_week"] = [max_week(x) for x in df.year]
    df["week"] = df[["week", "max_week"]].min(axis=1)
    df["date"] = df.apply(
        lambda x: date.fromisocalendar(x.year, x.week, 1), axis=1
    )
    df["date"] = pd.to_datetime(df["date"])

    # simplify type names if long so all datasets match
    df.cases_class.replace(
        to_replace=["CONFIRME", "SUSPECTE", "PROBABLE"],
        value=["CONF", "SUSP", "PROB"],
        inplace=True,
    )

    if keep_cases is not None:
        df = df[df.cases_class.isin(keep_cases)]

    return df


def aggregate_to_date(
    df: pd.DataFrame,
    cases_cols=["cases_number"],
    start_date: Union[str, None] = None,
    end_date: Union[str, None] = None,
):
    """Aggregate plague data to date

    Parameters
    ----------
    df : pd.DataFrame
        Plague data
    start_date : Union[str, None], optional
        Start date, if provided in ISO8601 string
        format, designates the earliest date for data
        for infilling with ``0`` if not present in
        ``df``.
    end_date : Union[str, None]
        End date, if provided in ISO8601 string
        format, designates the latest date for data
        for infilling with ``0`` if not present in
        ``df``.
    Returns
    -------
    pd.DataFrame
    """
    # group by date and sum cases
    df_date = df.groupby(["date", "year", "week"], as_index=False).sum()
    df_date.set_index("date", inplace=True)

    # extend data frame to start and end date if relevant
    if start_date is not None:
        df_date = df_date.append(
            pd.DataFrame(
                [[0]],
                columns=cases_cols,
                index=[pd.to_datetime(start_date, format="%Y-%m-%d")],
            )
        )
        df_date = df_date.sort_index()
    if end_date is not None:
        df_date = df_date.append(
            pd.DataFrame(
                [[0]],
                columns=[cases_cols],
                index=[pd.to_datetime(end_date, format="%Y-%m-%d")],
            )
        )

    # rename index to date for later
    df_date.index.names = ["date"]

    # fill the weeks that are not included with 0
    # else they will be ignored when computing the historical average
    df_date = df_date.asfreq("W-Mon").fillna(0)

    # compute the year and week numbers from the dates
    df_date[["year", "week"]] = df_date.index.isocalendar()[["year", "week"]]
    df_date.reset_index(inplace=True)
    df_date.drop("max_week", axis=1, inplace=True)
    df_date["date"] = pd.to_datetime(df_date["date"])

    return df_date
