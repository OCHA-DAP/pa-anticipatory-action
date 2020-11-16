#
# This is the server logic of the dashboard.


#output$plot_map <- renderPlot({
    
#    put the function which makes your map here *
#})



# Define server logic for random distribution app ----
server <- function(input, output) {
    
    # Reactive expression to generate the requested distribution ----
    # This is called whenever the inputs change. The output functions
    # defined below then use the value computed from this expression
    d <- reactive({
        dist <- switch(input$dist,
                       norm = rnorm,
                       unif = runif,
                       lnorm = rlnorm,
                       exp = rexp,
                       rnorm)
        
        dist(500)
    })
    
    # Generate a plot of the data ----
    # Also uses the inputs to build the plot label. Note that the
    # dependencies on the inputs and the data reactive expression are
    # both tracked, and all expressions are called in the sequence
    # implied by the dependency graph.
    output$plot <- renderPlot({
        dist <- input$dist
        n <- input$n
        
        hist(d(),
             main = paste("r", dist, "(", n, ")", sep = ""),
             col = "#75AADB", border = "white")
    })
    
    # Generate a summary of the data ----
    output$summary <- renderPrint({
        summary(d())
    })
    
    # Generate an HTML table view of the data ----
    output$table <- renderTable({
        d()
    })
    
    # select text to display
    output$text_eth <- renderUI({
                 str1 <- paste('Food insecurity criterion for',input$dist, '30% projected in IPC3+ and 5% increase')
                 str2 <- paste('Drought criterion:', 'Named as driver for food insecurity by FewsNet or Global IPC')
                HTML(paste(str1, str2, sep = '<br/>'))
    })
    
    test <- reactive({     
        my_number <- as.numeric(length(input$dist))
        ifelse(my_number <= 250,1,0)
    }) 
    
    output$conditional_text <- renderText({
        if(test() == 1){
            paste("Trigger met for these regions:",input$dist)
        }})
}