import cdstoolbox as ct

# Define 'retrieve' parameters
LEADTIME = 2
VALIDITY_MONTHS = 3
LEADTIME_MONTHS = [str(i) for i in range(LEADTIME, LEADTIME + VALIDITY_MONTHS)]
# hcyears = reference years to use to determine terciles
HCYEARS = (1993, 2016)
FCST_START_YEAR = "2021"
FCST_START_MONTH = "01"
# origin = organization producing model
ORIGIN = "ecmwf"
# system is the version of the model
SYSTEM = "5"
VARIABLE = "total_precipitation"


def cdmcoords_to_leadtime(ds, leadtime_index):
    dscoords = ct.cdm.get_coordinates(ds)
    start_dates = dscoords["forecast_reference_time"]["data"]

    dsout_list = []
    for ileadt in leadtime_index:
        this_leadt_list = []
        if (
            "forecast_reference_time"
            in dscoords["forecast_reference_time"]["dims"]
        ):
            for this_start in start_dates:
                this_start_ds = ct.cube.select(
                    ds, forecast_reference_time=this_start
                )
                this_leadt_list.append(
                    ct.cube.index_select(this_start_ds, time=ileadt)
                )
            this_leadt_ds = ct.cube.concat(
                this_leadt_list, dim={"forecast_reference_time": start_dates}
            )
        else:
            this_leadt_ds = ct.cube.index_select(ds, time=ileadt)

        dsout_list.append(this_leadt_ds)

    dsout = ct.cube.concat(dsout_list, dim={"leadtime_index": leadtime_index})

    return dsout


@ct.application(title="Tercile probabilities")
@ct.output.download()
@ct.output.download()
@ct.output.download()
@ct.output.download()
@ct.output.download()
@ct.output.download()
def application():
    hcst_list = []
    start_date_list = []
    # retrieve historical forecasts
    # till 2016 the forecasts are hindcasts, after they are forecasts
    for yy in range(HCYEARS[0], HCYEARS[1] + 1):
        hcst_yy = ct.catalogue.retrieve(
            "seasonal-monthly-single-levels",
            {
                "originating_centre": ORIGIN,
                "system": SYSTEM,
                "variable": VARIABLE,
                "product_type": "monthly_mean",
                "year": ["{}".format(yy)],
                "month": FCST_START_MONTH,
                "leadtime_month": LEADTIME_MONTHS,
            },
        )

        print(" ####### HCST_yy {} ####### ".format(yy))
        # structure historical data of yy to have entry per leadtime
        hcst_yy_ok = cdmcoords_to_leadtime(
            hcst_yy, list(range(len(LEADTIME_MONTHS)))
        )
        hcst_yy_ok_mean = ct.cube.average(hcst_yy_ok, dim="leadtime_index")

        hcst_list.append(hcst_yy_ok_mean)

        # get_coordinates gets a dictionary of all coordinate values
        # (seems to include parameters)
        hcst_yy_coords = ct.cdm.get_coordinates(hcst_yy_ok_mean)
        start_date_list.append(
            hcst_yy_coords["forecast_reference_time"]["data"]
        )

    # concat all historical forecasts of all years
    hcst = ct.cube.concat(
        hcst_list, dim={"forecast_reference_time": start_date_list}
    )

    print(" ####### HCST ####### ")
    print(hcst)

    quantiles = [0.33, 0.66]
    qbnds_list = []

    for qq in quantiles:
        # ct.cube.quantile returns the boundary quantile
        # value for quantile qq and data hcst
        qbnds_list.append(
            ct.cube.quantile(
                hcst, qq, dim=["forecast_reference_time", "realization"]
            )
        )

    hcst_qbnds = ct.cube.concat(qbnds_list, dim={"quantiles": quantiles})

    print(" ####### hcst_qbnds ####### ")
    print(hcst_qbnds)

    # retrieve the forecast data
    fcst_cdm = ct.catalogue.retrieve(
        "seasonal-monthly-single-levels",
        {
            "originating_centre": ORIGIN,
            "system": SYSTEM,
            "variable": VARIABLE,
            "product_type": "monthly_mean",
            "year": FCST_START_YEAR,
            "month": FCST_START_MONTH,
            "leadtime_month": LEADTIME_MONTHS,
        },
    )

    print(" ####### FCST_CDM ####### ")
    print(fcst_cdm)

    fcst = cdmcoords_to_leadtime(fcst_cdm, list(range(len(LEADTIME_MONTHS))))
    print(" ####### FCST ####### ")
    print(fcst)

    fcst_mean = ct.cube.average(fcst, dim="leadtime_index")
    fcst_coords = ct.cdm.get_coordinates(fcst_mean)

    # number of models in the ensemble
    ens_size = len(fcst_coords["realization"]["data"])

    # returns True if greater than and else False
    above = ct.operator.gt(
        fcst_mean, ct.cube.select(hcst_qbnds, quantiles=0.66)
    )
    below = ct.operator.lt(
        fcst_mean, ct.cube.select(hcst_qbnds, quantiles=0.33)
    )

    print(" ####### ABOVE ####### ")
    print(above)

    print(" ####### BELOW ####### ")
    print(below)

    # P_below is defined as the numbers of models that predict
    # rainfall in the historical bottom tercile/total number of models *100
    P_above = 100.0 * (ct.cube.sum(above, dim="realization") / float(ens_size))
    P_below = 100.0 * (ct.cube.sum(below, dim="realization") / float(ens_size))
    P_normal = 100.0 - (P_above + P_below)

    return P_below, P_above, P_normal, hcst, fcst_mean, hcst_qbnds
