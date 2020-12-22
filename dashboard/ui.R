# This is the front-end of the dashboard

fluidPage(
    
    # App title ----
    titlePanel("Anticipatory Action Pilots"),
    
    # Sidebar layout with input and output definitions ----
    sidebarLayout(
        
        # Sidebar panel for inputs ----
        sidebarPanel(
            
            # Input: Select the random distribution type ----
            radioButtons("country", "Select a country:",
                         c("Ethiopia" = "eth",
                           "Bangladesh" = "bgd",
                           "Somalia" = "som",
                           "Malawi" = "mwi")),
            #             selected = character(0)), no default
            
            radioButtons("source", "Select a source:",
                         c("FewsNet" = "fn",
                           "Global IPC" = "gbl")),
            
       uiOutput('projectionPeriods')
       
        ),
        
        # Main panel for displaying outputs
        mainPanel(
            
            # Output: Tabset w/ plot, summary, and table ----
            tabsetPanel(type = "tabs",
                        tabPanel("Triggered Regions", textOutput("triggered_regions_list")),
                        tabPanel("Trigger Map", plotOutput("trigger_map")),
                        tabPanel("Projections", DT::dataTableOutput("projections_table")),
                        tabPanel("Rainfall Forecasts", 
                                 tags$br(), 
                                 htmlOutput("iri"),
                                 #htmlOutput("icpac"),
                                 img(src="icpac_rainfall_20201222_JFM2021.png", align = "center", width=600, height=600),
                                 htmlOutput("chc"),
                                 htmlOutput("nmme"),
                                 textOutput("eth_nma")),
                        tabPanel("Reports", 
                              #   uiOutput("reports"),
                              #   tags$h3("Full Reports"),
                                 tags$br(),
                                 "Click on the links below to access the forecasts published by agencies.",
                                 tags$br(),
                                 tags$hr(),
                                 tags$a(href="https://fews.net/east-africa/ethiopia/food-security-outlook/october-2020", 
                                        "Food Insecurity: FewsNet"),
                                 tags$br(),
                                 tags$a(href="http://www.ipcinfo.org/ipc-country-analysis/details-map/en/c/1152818/?iso3=ETH", 
                                        "Food Insecurity: Global IPC"),
                                 tags$br(),
                                 tags$a(href="https://iri.columbia.edu/our-expertise/climate/forecasts/seasonal-climate-forecasts/", 
                                     "Rainfall: IRI"),
                                 tags$br(),
                                 tags$a(href="https://www.icpac.net/seasonal-forecast/", 
                                     "Rainfall: ICPAC"),
                                 tags$br(),
                                 tags$a(href="https://blog.chc.ucsb.edu/?p=898", 
                                     "Rainfall: CHC"),
                                 tags$br(),
                                 tags$a(href="https://www.cpc.ncep.noaa.gov/products/NMME/prob/PROBprate.S.html", 
                                     "Rainfall: NMME"),
                                 tags$br(),
                                 tags$a(href="http://www.ethiomet.gov.et/other_forecasts/seasonal_forecast", 
                                     "Rainfall: Ethiopian NMA"),
                              
                              
                              
                              )
                        )
            )
       #,
        #    br(),
        #    span(strong("Regions that meet the trigger for:"), style = "color:navy"),
        #    br(),
            
        #    textOutput("period"),
        #    br(),
        #    textOutput("triggered_regions_list")
        )
)
