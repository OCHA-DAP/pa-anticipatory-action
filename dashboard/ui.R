# This is the front-end of the dashboard

fluidPage(
    
    # styling
  #  theme = "bootstrap.css",
    
   # logo (must be in a subfolder named www that is sister to ui.r and not be listed with its path)
    img(src="double_logo_header.jpg", align = "center", width=1275, height=100),
  
    # App title ----
    titlePanel(title="Anticipatory Action in Ethiopia"), 
    
    br(),
    span(strong("Please note that this dashboard is under development and intended for forecast exploration only. Feedback and suggestions can be directed to Josée Poirier (josee.poirier@un.org)."), style = "color:red"),
    br(),
    br(),
    
    # define layout
    sidebarLayout(
      
       # position sidebar to the right of main panel
        position = "right",
        
        # Sidebar panel for definition of trigger
        sidebarPanel(
          h4(strong(div("Trigger Definition", style = "color:#007CE0")), align = "center"),
          h5(strong(div("Food Insecurity", style = "color:#1EBFB3"))),
          h6(p("At least 20% population of one or more ADMIN1 regions be projected at IPC4+"),
             p(strong("OR")),
             p("At least 30% of ADMIN1 population be projected at IPC3+ AND with an increase by 5 percentage points compared to current state")),
          tags$hr(),
          h5(strong("Drought", style = "color:#1EBFB3")),
          h6(p("At least 50% probability of below average rainfall from at least two seasonal rainfall forecasts"),
             p(strong("OR")),
             p("Drought named as a driver of the deterioration of the situation in food security by FewsNet or Global IPC")
          )),
        
        # Main panel for displaying outputs
        mainPanel(
          
          
          tabsetPanel(
  
            # Tabs definitions
            tabsetPanel(type = "tabs",
                        tabPanel("Trigger Status", 
                                br(),            
                                h4(strong("Key Messages")),
                                h5(strong(div(tags$li("The trigger thresholds have been met, allowing for the disbursement of the second tranche of projects under the activation of the Ethiopia AA framework."), style = "color:#007CE0")),
                                    tags$li("Food security has deteriorated since October 2020 and is projected to further worsen in the coming months."),
                                    tags$li("The following regions meet the food insecurity trigger over the next 3-4 months: Afar, Oromia, Somali, SNNP, Tigray."),
                                    tags$li("Below average rainfall is projected in parts of the country including sections of Afar, Oromia, Somali, SNNP, Tigray although with a probability less than 50%."),
                                    tags$li("Below average rains is mentioned as a driver of projected food insecurity.")),
                                 br(),
                                 uiOutput("summary_link"),
                                 br(),
                                 em(textOutput("last_update"))
                                 ),
                        
                        tabPanel("Food Insecurity",
                                 fluidRow(plotOutput('trigger_map')),
                                 fluidRow(DT::dataTableOutput("projections_table"))
                                ),
                        tabPanel("Rainfall Forecasts", 
                                 tags$br(), 
                                 htmlOutput("iri_text"),
                                 htmlOutput("iri"),
                                 tags$hr(),
                                 tags$br(),
                                 htmlOutput("icpac_text"),
                                 htmlOutput("icpac"),
                                 tags$hr(),
                                 tags$br(),
                                 htmlOutput("nma_text"),
                                 img(src="eth_nma_precipitation_202101_Belg2021.png", align = "center", width=600, height=400),
                                 tags$hr(),
                                 tags$br(),
                                 htmlOutput("nmme_text"),
                                 htmlOutput("nmme"),
                                 tags$hr(),
                                 tags$br(),
                                 htmlOutput("chc_text"),
                                 htmlOutput("chc"),
                                 tags$hr(),
                                 tags$br(),
                                 htmlOutput("copernicus_text"),
                                 htmlOutput("copernicus")
                                 ),
                                 #textOutput("eth_nma")),
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
                                     "Rainfall: Ethiopian NMA")
                              
                              
                              
                              )
                        )
            )
)
)
)