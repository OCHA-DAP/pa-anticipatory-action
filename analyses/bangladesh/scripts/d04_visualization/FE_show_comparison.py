import matplotlib.pyplot as plt
import seaborn as sns

def compare_estimates(data, sel_union):
    """
    Return a graph for the selected union, comparing Sentinel-1 estimates with interview estimates.
    data = dataframe with the flooding estimates over time for BGD admin-4 regions
    sel_union = name of ADM4 region
    """

    # Get only the data for the selected union
    df_selected = data.loc[data['ADM4_EN'] == sel_union]
    # Select the relevant flooding estimate data
    # Get the various estimates of flood extent by date (in %)
    flood_extent = df_selected[['flood_fraction', 'Interview_1', 'Interview_2', 'Interview_3', 'date']]
    # Melt data to long format for visualization
    flood_extent_long = flood_extent.melt(id_vars=['date'])
    # Colours for the line graph
    col_mapping = {
        'Interview_1': '#520057',
        'Interview_2': '#db00e8',
        'Interview_3': '#d096d4',
        'flood_fraction': '#ff9626'
    }
    # Get the start, end, and peak times
    start_dates = get_dates(data, sel_union, 'start')
    peak_dates = get_dates(data, sel_union, 'pick')
    end_dates = get_dates(data, sel_union, 'end')
    # Create simple line plot to compare the satellite estimates and interviewed estimates
    fig = plt.figure()
    ax = plt.axes()
    sns.lineplot(x='date', y='value', hue='variable', data=flood_extent_long, palette=col_mapping)
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Percent flooding estimate')
    plt.ylim([0, 110])
    plt.title('Estimates of flooding in {}, Bangladesh'.format(sel_union))

    for date in start_dates:
        plt.axvline(date, ls='--', color='#0fbd09', label='Flood Start', lw=0.75)
    for date in peak_dates:
        plt.axvline(date, ls='--', color='#f20a30', label='Flood Peak', lw=0.75)
    for date in end_dates:
        plt.axvline(date, ls='--', color='#032cfc', label='Flood End', lw=0.75)

    plt.legend(loc='lower right', bbox_to_anchor=(1.05, 1))
    leg = plt.legend()
    leg.get_texts()[1].set_text('Sentinel-1')
    leg.get_texts()[0].set_text('Legend')

    plt.tight_layout()
    plt.savefig("Results_images/{}.png".format(sel_union), bbox_inches='tight', pad_inches=0.2)