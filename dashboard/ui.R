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
                        tabPanel("Reports", uiOutput("reports"))
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
)
