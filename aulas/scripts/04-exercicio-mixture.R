
# Pacotes ------------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(ISLR2)

# Dados -------------------------------------------------------------------

data("Hitters")

# Hitters <- na.omit(Hitters)

?Hitters

glimpse(Hitters)
skimr::skim(Hitters)

# Ideia do modelo ---------------------------------------------------------

# Construir um modelo que nos ajude a definir um pagamento para um jogador
# Ou seja, queremos prever o salario

# Base treino e teste -----------------------------------------------------

set.seed(123)

# 25% de teste e 75% de treino
hitters_initial_split <- Hitters |> initial_split(3 / 4)
hitters_initial_split

# Define treino
hitters_train <- training(hitters_initial_split)
skimr::skim(hitters_train)

# Define teste
hitters_test <- testing(hitters_initial_split)
skimr::skim(hitters_test) 

# Dataprep ----------------------------------------------------------------

hitters_recipe <- recipe(Salary ~ ., data = hitters_train) |>
  step_naomit(everything(), skip = TRUE) |>
  step_rm(all_nominal()) |>
  step_normalize(all_numeric_predictors())


rec_prep <- hitters_recipe |>
  prep()

rec_prep |>
  bake(new_data = NULL)

# juice faz a mesma coisa tudo de uma vez
juice(rec_prep) |> glimpse()

# Definicao do modelo -----------------------------------------------------

hitters_model <- linear_reg(
  penalty = tune(), 
  mixture = tune() ## Tunando mixture ----
) |>
  set_engine("glmnet") |> 
  set_mode("regression")

# Criando o workflow ------------------------------------------------------

hitters_wflow <- workflow() |>
  add_recipe(hitters_recipe) |>
  add_model(hitters_model)

# Tunagem de hiperparametros ----------------------------------------------

## Reamostragem com cross-validation --------------------------------------

hitters_resamples <- vfold_cv(hitters_train, v = 5)

hitters_grid <- grid_regular(
  penalty(c(-1, 2)),
  mixture(c(0, 1)), ### Definindo grid do mixture ----
  levels = 10 
)

# hitters_grid <- tibble(penalty = seq(0.1, 2, length.out = 20))

hitters_tune_grid <- tune_grid(
  hitters_wflow,
  resamples = hitters_resamples,
  grid = hitters_grid,
  metrics = metric_set(rmse, rsq, mape, mae),
  control = control_grid(verbose = TRUE, allow_par = FALSE)
)

collect_metrics(hitters_tune_grid)

# Inspecao da tunagem -----------------------------------------------------

# hitters_tune_grid$.metrics[[3]]
autoplot(hitters_tune_grid)

collect_metrics(hitters_tune_grid) |>
  ggplot(aes(x = penalty, y = mean)) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin = mean - 1.9 * std_err, ymax = mean + 1.9 * std_err)) +
  facet_wrap(~.metric, scales = "free_y") +
  scale_x_log10()


show_best(hitters_tune_grid, n = 1, metric = "rmse")
show_best(hitters_tune_grid, n = 1, metric = "rsq")

# Seleciona o melhor conjunto de hiperparametros
hitters_best_hiperparams <- select_best(hitters_tune_grid, "rmse")

# Adiciona mais um "passo" no workflow
hitters_wflow <- hitters_wflow |>
  # Finalize é como se o "formulario" tivesse sido preenchido (ver linha 114)
  finalize_workflow(hitters_best_hiperparams)

# Desempenho do modelo final ----------------------------------------------

# Adiciona mais um passo
hitters_model_train <- hitters_wflow |>
  # fit ajusta a receita
  fit(data = hitters_train)

# Erro de TESTE (RMSE)
pred <- predict(hitters_model_train, hitters_test)
bind_cols(pred, hitters_test) |> rmse(truth = Salary, estimate = .pred)

# Erro de TREINO (RMSE)
pred <- predict(hitters_model_train, hitters_train)
bind_cols(pred, hitters_train) |> rmse(truth = Salary, estimate = .pred)

# Erro de TESTE (RSQ)
pred <- predict(hitters_model_train, hitters_test)
bind_cols(pred, hitters_test) |> rsq(truth = Salary, estimate = .pred)

# Erro de TREINO (RSQ)
pred <- predict(hitters_model_train, hitters_train)
bind_cols(pred, hitters_train) |> rsq(truth = Salary, estimate = .pred)


# Construindo o modelo final ----------------------------------------------

# Constroi o ultimo modelo (sem fazer o procedimento manual acima)
hitters_last_fit <- hitters_wflow |>
  last_fit(split = hitters_initial_split)

collect_metrics(hitters_last_fit)
collect_predictions(hitters_last_fit) |>
  ggplot(aes(.pred, Salary)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1)

# Modelo final (na base final) --------------------------------------------

# Ajustando na base completa
hitters_final_model <- hitters_wflow |> fit(data = Hitters)

# Predicoes ---------------------------------------------------------------

hitters_com_previsao <- Hitters |>
  mutate(
    salary_pred = predict(hitters_final_model, new_data = .)$.pred
  )

# Chega um atleta novo
atleta_novo <- tibble(
  AtBat = 293L,
  Hits = 100L,
  HmRun = 1L,
  Runs = 30L,
  RBI = 29L,
  Walks = 14L,
  Years = 1L,
  CAtBat = 293L,
  CHits = 66L,
  CHmRun = 1L,
  CRuns = 30L,
  CRBI = 29L,
  CWalks = 14L,
  PutOuts = 446L,
  Assists = 33L,
  Errors = 50L,
  League = NA,
  Division = NA,
  NewLeague = NA
)

predict(hitters_final_model, new_data = atleta_novo)

# Olhando os beta
hitters_final_model |>
  extract_fit_engine() |>
  coef(s = 21.5443469003188) # lambda: para ver esse valor é só rodar hitters_wflow

# Guardar o modelo para usar depois ---------------------------------------

saveRDS(hitters_final_model, file = "hitters_final_model.rds")

modelo <- readRDS("aulas/scripts/hitters_final_model.rds")

predict(modelo, Hitters)
