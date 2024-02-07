
# Pacotes -----------------------------------------------------------------

library(tidyverse)
library(skimr)
library(tidymodels)
library(parsnip)

# Dados -------------------------------------------------------------------

diamonds

# Vamos pensar na seguinte situação: numa joalheira, chegou um novo diamante e 
# precisamos desenvolver um modelo de ML para definir o preço dele. Se não temos
# o preço, mas outras caracterísitcas, como cor, corte, peso, largura etc., 
# podemos definir um preço com ML. Esse preço é um "chute", porque quando o 
# diamante for comercializado, as pessoas podem não comprá-lo, ou podem pedir
# desconto etc.

### Olhando as variáveis ----
glimpse(diamonds)

### Olhando para o comportamento das variáveis ----
skim(diamonds)

# Gráfico da relação entre x e price ----

qplot(x, log(price), data = diamonds)
qplot(price, data = diamonds, geom = "histogram")

# Precisamos passar para o R ----
# 1. A f que queremos usar
# 2. Ajusta a f para um conjunto de dados

# Passo 1: Especificações -------------------------------------------------

# a) A f (a hipótese) coms seus respectivo hiperparâmetros
# b) O pacote "motor" (engine)
# c) A tarefa/modo ("regression" ou "classification")

# Funções são do parsnip

especificacao_modelo <- linear_reg() |> # Tipo de modelo que queremos
  set_engine("lm") |>    # Motor que queremos (modelo)
  set_mode("regression") # Regressão que queremos
  
# Outros exemplos...

# especificacao_modelo <- decision_tree() |> # Tipo de modelo que queremos
#   set_engine("rpart") |>                   # Motor que queremos
#   set_mode("regression")                   # Regressão que queremos

# Passo 2: Ajuste do modelo -----------------------------------------------

diamantes <- diamonds  |>
  mutate(
    log_price = log(price),
    flag_maior_que_8 = ifelse(x > 8, "sim", "nao")
)

modelo <- especificacao_modelo |>
  fit(log_price ~ x + flag_maior_que_8, data = diamantes)
parsnip::
print(modelo)

# Esses dois são iguais
reg1 <- extract_fit_engine(modelo)
reg2 <- lm(price ~ x, data = diamantes)

# Passo 3: Analisar as previsões ------------------------------------------

so_o_x <- diamantes |>
  select(x, flag_maior_que_8)

pred <- predict(modelo, new_data =  so_o_x)

# Aqui só vamos usar tidyverse p/ baixo

diamonds_com_previsao <- diamantes |>
  add_column(pred) |>
  mutate(.pred_price = exp(.pred))

# Pontos observados + curva da f
diamonds_com_previsao |>
  filter(x > 0) |>
  sample_n(1000) |>
  ggplot() +
  geom_point(aes(x, price), alpha = 0.3) +
  geom_point(aes(x, .pred_price), color = "red") +
  theme_bw()

# Observado vs Esperado
diamonds_com_previsao |>
  filter(x > 0) |>
  ggplot() +
  geom_point(aes(.pred_price, price)) +
  geom_abline(slope = 1, intercept = 0, colour = "purple", size = 1) +
  theme_bw()

diamonds_com_previsao |>
  filter(x > 0) |>
  sample_n(1000) |>
  ggplot() +
  geom_point(aes(price, price - .pred_price))

# Como quantificar a qualidade de um modelo?

library(yardstick)

metrics <- metric_set(rmse, mae, rsq)

# Métricas de erro
diamonds_com_previsao |>
  metrics(truth = price, estimate = .pred_price)

diamonds_com_previsao |>
  mae(truth = price, estimate = .pred_price)
diamonds_com_previsao |>
  rsq(truth = price, estimate = .pred_price)
diamonds_com_previsao |>
  rmse(truth = price, estimate = .pred_price)




