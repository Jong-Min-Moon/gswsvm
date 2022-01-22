library(plotly) #plotly 패키지 로드
p <- plot_ly(iris,
             x = Sepal.Length, y = Sepal.Width, z = Petal.Length,
             color = iris$Species, colors = c('#BF382A', '#0C4B8E')
             ) %>%
    add_markers() %>%
  + layout(
      scene = list(
        xaxis = list(title = 'Sepal.Length'), #x축 제목설정
        yaxis = list(title = 'Sepal.Width'),  #y축 제목설정
        zaxis = list(title = 'Petal.Length') #z축 제목설정
                   ) #list
      ) #layout
p