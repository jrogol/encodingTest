# based on the blog post: https://blogs.rstudio.com/ai/posts/2018-11-26-embeddings-fun-and-profit/
library(dplyr)
library(keras)

# Filter for complete cases.
sales <- DMwR2::sales %>% 
  filter(!is.na(Quant),!is.na(Val))

# Eliminate the unknowns, refactor Insp.
sales <- sales %>% 
  filter(Insp != "unkn") %>% 
  droplevels()

# Of the four columns, ID is the salesperson ID
# Prod is the product Id
# Quant is the quantity of the item sold
# Val is the value of the transaction.
# Insp iindicates if the transaction was inspected and determined to be fraud or not.

sales_embed <- sales

top_n <- 500

# Identify the top 500 products.
top_prods <- sales_embed %>% 
  group_by(Prod) %>% 
  summarise(cnt = n()) %>% 
  arrange(desc(cnt)) %>% 
  head(top_n) %>% 
  select(Prod) %>% 
  pull()

sales_embed <- sales_embed %>% 
  filter(Prod %in% top_prods) %>% 
  droplevels()

sales_embed <-
  sales_embed %>% 
  select(-ID) %>% 
  mutate(across(Quant:Val, \(x) scale(x)),
    # Turn factors into Integers
    across(where(is.factor), \(x) as.integer(x) - 1))

set.seed(1754)
train_indices <- sample(1:nrow(sales_embed), 0.7 * nrow(sales_embed))

# Train/Test Splits -------------------------------------------------------
# These must be matrices.
X_train <- sales_embed[train_indices, 1:3] %>% as.matrix()
y_train <-  sales_embed[train_indices, 4] %>% as.matrix()

X_valid <- sales_embed[-train_indices, 1:3] %>% as.matrix()
y_valid <-  sales_embed[-train_indices, 4] %>% as.matrix()




# Input Layers ------------------------------------------------------------
# Gah. Have to install tensorflow with Python, which isn't configured....

# Name these for ease of use.
prod_input <- layer_input(shape = 1,name = "product")
cont_input <- layer_input(shape = 2, name = "QuantVal")

# -------------------------------------------------------------------------
prod_embed <- prod_input %>% 
  layer_embedding(input_dim = sales_embed$Prod %>% max() + 1,
                  output_dim = 256
  ) %>%
  layer_flatten()
cont_dense <- cont_input %>% layer_dense(units = 256, activation = "selu")

output <- layer_concatenate(
  list(prod_embed, cont_dense)) %>%
  layer_dropout(dropout_rate) %>% 
  layer_dense(units = 256, activation = "selu") %>%
  layer_dropout(dropout_rate) %>% 
  layer_dense(units = 256, activation = "selu") %>%
  layer_dropout(dropout_rate) %>% 
  layer_dense(units = 256, activation = "selu") %>%
  layer_dropout(dropout_rate) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(inputs = list(prod_input, cont_input), outputs = output)

model %>% compile(loss = "binary_crossentropy", optimizer = "adam", metrics = "accuracy")

model %>% fit(
  list(X_train[ , 1], X_train[ , 2:3]),
  y_train,
  validation_data = list(list(X_valid[ , 1], X_valid[ , 2:3]), y_valid),
  class_weights = list("0" = 0.1, "1" = 0.9),
  batch_size = 128,
  epochs = 200
)

model %>% evaluate(list(X_train[ , 1], X_train[ , 2:3]), y_train) 
model %>% evaluate(list(X_valid[ , 1], X_valid[ , 2:3]), y_valid)




