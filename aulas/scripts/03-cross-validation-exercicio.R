# Dados -------------------------------------------------------------------

library(basesCursoR)

df <- basesCursoR::pegar_base("ssp")

glimpse(df)

dados <- df |>
  filter(ano != 2020) |>
  filter(municipio_nome == "São Paulo")
# group_by(ano) |>
# summarise(across(
#   .cols = estupro:vit_latrocinio,
#   .fns = mean
# ))

# Ideia do Modelo ---------------------------------------------------------

# Queremos um modelo que preveja a média de crimes em 2020 para cada tipo de
# crime (furto, roubo etc.)

# Base de treino e de teste -----------------------------------------------

dados_initial_split <- dados |> initial_split(prop = 3 / 4)
dados_initial_split

# Definindo base de treino
dados_train <- training(dados_initial_split)

# Definindo base de teste
dados_test <- testing(dados_initial_split)

# Dataprep ----------------------------------------------------------------

# Precisamos padronizar os dados para fazer com que todos os betas fiquem na
# mesma escala

dados_recipe <- recipe(furto_outros ~ ., data = dados) |>
  step_naomit(everything(), skip = TRUE) |>
  step_rm(all_nominal()) |>
  step_normalize(all_numeric_predictors())

rec_prep <- dados_recipe %>%
  prep()

rec_prep %>%
  bake(new_data = NULL)

juice(rec_prep) %>% glimpse()

# Definicao do modelo -----------------------------------------------------

dados_model <- linear_reg(
  penalty = tune(),
  mixture = 1
) |>
  set_engine("glmnet") |>
  set_mode("regression")

# Criando o workflow ------------------------------------------------------

# O modelo é só a f, mas o programa que construimos para ajustar a f deve ter
# um controle de erro muito bom

# Um workflow é um conjunto de passos

dados_wflow <- workflow() |> 
  add_recipe(dados_recipe) |> 
  add_model(dados_model)

# Tunagem de hiperparametros ----------------------------------------------

# Reamostragem com cross-validation ---------------------------------------
dados_resamples <- vfold_cv(dados_train, v = 5)

# Onde queremos que o lambda seja procurado?
dados_grid <- grid_regular(
  penalty(c(-1, 2)), # entre exp(-1) e exp(2)
  levels = 10 # tente 10 lambdas
)

# dados_grid <- tibble(penalty = seq(0.1, 2, length.out = 20))

dados_tune_grid <- tune_grid(
  dados_wflow,
  resamples = dados_resamples,
  grid = dados_grid,
  metrics = metric_set(rmse, rsq),
  control = control_grid(verbose = TRUE, allow_par = FALSE)
)

collect_metrics(dados_tune_grid) |> view()

# Inspecao da tunagem -----------------------------------------------------

# dados_tune_grid$.metrics[[3]]
autoplot(dados_tune_grid)

collect_metrics(dados_tune_grid) %>%
  ggplot(aes(x = penalty, y = mean)) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin = mean - 1.9 * std_err, ymax = mean + 1.9 * std_err)) +
  facet_wrap(~.metric, scales = "free_y") +
  scale_x_log10()


show_best(dados_tune_grid, n = 1, metric = "rmse")
show_best(dados_tune_grid, n = 1, metric = "rsq")

# Seleciona o melhor conjunto de hiperparametros
dados_best_hiperparams_rsq <- select_best(dados_tune_grid, "rsq")
dados_best_hiperparams <- select_best(dados_tune_grid, "rmse")

dados_wflow <- dados_wflow %>%
  finalize_workflow(dados_best_hiperparams)

# Desempenho do modelo final ----------------------------------------------

dados_model_train <- dados_wflow %>%
  fit(data = dados_train)

# Erro de TESTE (RMSE)
pred <- predict(dados_model_train, dados_test)
bind_cols(pred, dados_test) |> rmse(truth = furto_outros, estimate = .pred)

# Erro de TREINO (RMSE)
pred <- predict(dados_model_train, dados_train)
bind_cols(pred, dados_train) %>% rmse(truth = furto_outros, estimate = .pred)

# Erro de TESTE (RSQ)
pred <- predict(dados_model_train, dados_test)
bind_cols(pred, dados_test) %>% rsq(truth = furto_outros, estimate = .pred)

# Erro de TREINO (RSQ)
pred <- predict(dados_model_train, dados_train)
bind_cols(pred, dados_train) %>% rsq(truth = furto_outros, estimate = .pred)

# Construindo o modelo final ----------------------------------------------

# Constroi o ultimo modelo (sem fazer o procedimento manual acima)
dados_last_fit <- dados_wflow %>%
  last_fit(split = dados_initial_split)

collect_metrics(dados_last_fit)
collect_predictions(dados_last_fit) %>%
  ggplot(aes(.pred, furto_outros)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1)

# Modelo final (na base final) --------------------------------------------

# Ajustando na base completa
dados_final_model <- dados_wflow %>% fit(data = dados)

# Predicoes ---------------------------------------------------------------

dados_com_previsao <- dados %>%
  mutate(
    furto_outros_pred = predict(dados_final_model, new_data = .)$.pred
  )

glimpse(dados_com_previsao)

# Olhando o RMSE
dados_com_previsao |> 
  select(mes, ano, furto_outros, furto_outros_pred) |> 
  mutate(rmse = sqrt(mean((furto_outros - furto_outros_pred)^2)))

# Chegam dados sobre 2020
ano_2020 <- df |> filter(ano == 2020) |> 
  mutate(furto_outros = NA)

predict(dados_final_model, new_data = ano_2020)

x_2020 <- df |> filter(ano == 2020) |> 
  mutate(
    furto_outros_pred = predict(dados_final_model, new_data = ano_2020)$.pred
    )

glimpse(x_2020)

# Olhando o RMSE
x_2020 |> 
  select(mes, ano, furto_outros, furto_outros_pred) |> 
  mutate(diff = furto_outros - furto_outros_pred,
         rmse = sqrt(mean((furto_outros - furto_outros_pred)^2))) |> 
  filter(diff > -.5 & diff < .5)


# Olhando os beta
dados_final_model %>%
  extract_fit_engine() %>%
  coef(s = 0.215443469003188) # lambda: para ver esse valor é só rodar dados_wflow






