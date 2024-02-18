

# Pacotes -----------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(skimr)

# Dados -------------------------------------------------------------------

data("diamonds")

# Análise -----------------------------------------------------------------

glimpse(diamonds)
skim(diamonds)

# A função skim() permite olhar rapidamente para algumas medidas descritivas
# da base. Vamos usar o X para fazer um modelo pq é uma variável com uma 
# variação razoavel

qplot(x, price, data = diamonds)
qplot(x, log(price), data = diamonds)
qplot(price, data = diamonds, geom = "histogram")

# Podemos ver que x e price apresentam uma relação linear, o que permite 
# modelarmos essas variaveis com uma regressao linear

# Passo a passo -----------------------------------------------------------

# Precisamos passar para o R duas coisas:
# 1. A função f que queremos usar
# 2. Ajustar essa função f para um conjunto de dados

# Passo 1: Especificar a função -------------------------------------------

# Devemos especificar
# a) A função (a hipótese) com seus respectivos hiperparâmetros
# b) O pacote 'motor' (engine)
# c) A tarefa/modo ("regression" ou "classification")

especificacao_modelo <- parsnip::linear_reg() |> # passo a)
  parsnip::set_engine("lm") |>                   # passo b)
  parsnip::set_mode("regression")                # passo c)

# Outros exemplos

especificacao_modelo2 <- decision_tree() |> 
  set_engine("rpart") |> 
  set_mode("regression")

especificacao_modelo3 <- rand_forest() |> 
  set_engine("ranger") |> 
  set_mode("regression")

# Passo 2: Ajustar o modelo -----------------------------------------------

diamonds <- diamonds |> 
  mutate(
    log_price = log(price),
    flag_maior_que_8 = ifelse(x > 8, "sim", "nao")
  )

modelo <- especificacao_modelo |> 
  fit(log_price ~ x, data = diamonds)

# A função fit() encontra o melhor parametro (ou seja, aquele que minimiza o
# erro)

# Passo 3: Analisar as previsões ------------------------------------------

somente_x <- diamonds |> 
  select(x, flag_maior_que_8)

pred <- predict(modelo, new_data = somente_x)

# A função predict() retorna ?

# Adicionando a coluna pred na base original ------------------------------

diamonds_com_previsao <- diamonds |> 
  add_column(pred) |> 
  mutate(.pred_price = exp(.pred))

# Entendendo o erro -------------------------------------------------------

### Curva da função -------------------------------------------------------

diamonds_com_previsao |> 
  filter(x > 0) |> 
  sample_n(1000) |> 
  ggplot() +
  geom_point(aes(x, price), alpha = .3) +
  geom_point(aes(x, .pred_price), color = "red") +
  theme_bw()

### Observado vs Esperado -------------------------------------------------

diamonds_com_previsao |> 
  filter(x > 0, .pred_price < 2000) |> 
  sample_n(1000) |> 
  ggplot() +
  geom_point(aes(.pred_price, price)) +
  geom_abline(slope = 1, intercept = 0, color = "purple", size = 1) +
  theme_bw()

### Resíduos contra o preditor ---------------------------------------------

diamonds_com_previsao |> 
  filter(x > 0) |> 
  sample_n(1000) |> 
  ggplot() +
  geom_point(aes(price, price - .pred_price)) +
  theme_bw()

# Como quantificar a qualidade de um modelo? ------------------------------

library(yardstick)

metrics <- metric_set(rmse, mae, rsq, mape)

# Métricas de erro# Métricas de erromape()
diamonds_com_previsao |> 
  metrics(truth = price, estimate = .pred_price)

diamonds_com_previsao |> 
  rmse(truth = price, estimate = .pred_price)
diamonds_com_previsao |> 
  mae(truth = price, estimate = .pred_price)
diamonds_com_previsao |> 
  rsq(truth = price, estimate = .pred_price)
diamonds_com_previsao |> 
  mape(truth = price, estimate = .pred_price)

# Importante ter em mente que os valores do estimate estão em $ e a correta 
# interpretação depende da metrica. No caso do RMSE, temos que o modelo "erra"
# cerca de $2622 para mais ou para menos. No fim, o que queremos é que o valor
# seja baixo.

# Exemplo de programa "final" ---------------------------------------------

modelo

# Salvamos o modelo em um arquivo .rds que pode ser aplicado em varios outros
# lugares
saveRDS(modelo, "modelo.rds")

# Lendo o modelo final
modelo_final <- readRDS("modelo.rds")

# Dados novos (só exemplo)
dado_novo <- tibble(x = 10)

# Aplicando o modelo nos dados novos
predict(modelo_final, dado_novo)









