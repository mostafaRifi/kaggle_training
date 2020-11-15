library(reticulate)
library(tidyverse)


conda_list()[[1]][3] %>% 
  use_condaenv(required = TRUE)