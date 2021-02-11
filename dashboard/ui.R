# This is the front-end of the dashboard

fluidPage(
    
    # styling
  #  theme = "bootstrap.css",
    
    # App title ----
    titlePanel(title="Anticipatory Action Pilots"), 
    
    # logo (must be in a subfolder named www that is sister to ui.r and not be listed with its path)
    img(src="double_logo_header.jpg", align = "center", width=1275, height=100),
        
    br(),
    span(strong("Please note that this dashboard is under development and intended for forecast exploration only. It should not be used as the main source of information for decision-making. Feedback and suggestions for functionalities can be directed to Josée Poirier (josee.poirier@un.org)."), style = "color:red"),
    br(),
    br(),
    
    # Sidebar layout with input and output definitions
    sidebarLayout(
        
        # Sidebar panel for inputs
        sidebarPanel(
            
            # Input: select country
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
            
            # Tabset w/ plot, summary, and table
            tabsetPanel(type = "tabs",
                        tabPanel("Triggered Regions", textOutput("triggered_regions_list")),
                        tabPanel("Trigger Map", plotOutput("trigger_map")),
                        tabPanel("Projections", DT::dataTableOutput("projections_table")),
                        tabPanel("Rainfall Forecasts", 
                                 tags$br(), 
                                 htmlOutput("iri"),
                                 htmlOutput("icpac"),
                                 #img(src="icpac_rainfall_20201222_JFM2021.png", align = "center", width=600, height=600),                                 
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
)
)