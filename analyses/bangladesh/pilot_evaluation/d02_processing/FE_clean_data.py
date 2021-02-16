import pandas as pd


def clean_df(df, group):
    """

    Gets rid of redundant columns, converts to date time format
    df: dataframe output from FE_flood_extent.py
    group: ADM4, ADM3, ADM2, ADM1

    """
    name = group + '_EN'
    pcode = group + '_PCODE'
    copy = df
    copy['date'] = pd.to_datetime(copy['date'], format="%Y-%m-%d").dt.strftime('%Y-%m-%d')
    # copy = copy.groupby(['date', group]).mean().reset_index()
    output = copy[[name, pcode, 'flood_fraction', 'date']]
    return output


def select_df(df, select):
    """

    Select single admin area from a grouped df
    df: dataframe output from group_df()
    group: name of the admin area to select

    """
    copy = df
    adm_col = copy.columns[1]  # Match on pcode column
    return df.loc[df[adm_col] == select].reset_index()
