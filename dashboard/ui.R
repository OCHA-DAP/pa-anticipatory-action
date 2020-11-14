# This is the front-end of the dashboard

#
# plotOutput("plot_map")
#

fluidPage(
    
    # App title ----
    titlePanel("Anticipatory Action Pilots"),
    
    # Sidebar layout with input and output definitions ----
    sidebarLayout(
        
        # Sidebar panel for inputs ----
        sidebarPanel(
            
            # Input: Select the random distribution type ----
            radioButtons("dist", "Select a country:",
                         c("Ethiopia" = "norm",
                           "Bangladesh" = "unif",
                           "Somalia" = "lnorm",
                           "Malawi" = "exp")),
            
            # br() element to introduce extra vertical spacing ----
            br()
        ),
        
        # Main panel for displaying outputs ----
        mainPanel(
            
            # Output: Tabset w/ plot, summary, and table ----
            tabsetPanel(type = "tabs",
                        tabPanel("Trigger", htmlOutput("text_eth")),
                        tabPanel("Projections", verbatimTextOutput("summary")),
                        tabPanel("Reports", tableOutput("table"))
            ),
            
                p("p creates a paragraph of text."),
                p("A new p() command starts a new paragraph. Supply a style attribute to change the format of the entire paragraph.", style = "font-family: 'times'; font-si16pt"),
                strong("strong() makes bold text."),
                em("em() creates italicized (i.e, emphasized) text."),
                br(),
                code("code displays your text similar to computer code"),
                div("div creates segments of text with a similar style. This division of text is all blue because I passed the argument 'style = color:blue' to div", style = "color:blue"),
                br(),
                p("span does the same thing as div, but it works with",
                  span("groups of words", style = "color:blue"),
                  "that appear inside a paragraph."),
            
            textOutput("conditional_text"),
            
        )
    )
)