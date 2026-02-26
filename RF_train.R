# ==================== 参数设置区域 ====================
# 随机种子
SEED <- 123
SEED_LEARNING_CURVE <- 234

# 交叉验证折数
CV_FOLDS <- 5

# 工作路径
WORK_DIR <- ""
OUTPUT_DIR <- ""

# 创建输出目录
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
}

# 模型选择的评价指标 (可选: "Accuracy", "AUC", "F1", "Precision", "Recall")
SELECTION_METRIC <- "AUC"

# 随机森林参数网格
RF_GRID <- expand.grid(
  num.trees = c(1000),
  mtry = c(3),
  min.node.size = c(32),
  max.depth = c(3),
  sample.fraction = c(0.7)
)

# 学习曲线采样比例（用于过拟合诊断）
LEARNING_CURVE_SAMPLES <- c(0.2, 0.4, 0.6, 0.8, 1.0)

# ==================== 加载库 ====================
library(readxl)
library(ranger)
library(writexl)
library(caret)
library(pROC)
library(ggplot2)
library(reshape2)

# ==================== 设置工作环境 ====================
setwd(WORK_DIR)
set.seed(SEED)

# 验证评价指标设置
valid_metrics <- c("Accuracy", "AUC", "F1", "Precision", "Recall")
if (!(SELECTION_METRIC %in% valid_metrics)) {
  stop(paste0("错误: SELECTION_METRIC 必须是以下之一: ", 
              paste(valid_metrics, collapse = ", ")))
}

cat("==================== 模型配置 ====================\n")
cat("选择指标:", SELECTION_METRIC, "\n")
cat("交叉验证折数:", CV_FOLDS, "\n")
cat("随机种子:", SEED, "\n")
cat("==============================================\n\n")

# ==================== 读取数据 ====================
data <- read_excel("")
feature_names <- names(data)

# 提取特征和目标变量
y <- data[[1]]
X <- as.data.frame(data[, -1])

# 转换目标变量
y_factor <- as.factor(y)
y_numeric <- as.integer(y_factor) - 1
num_class <- length(levels(y_factor))

cat("数据加载完成\n")
cat("样本数:", nrow(X), "\n")
cat("特征数:", ncol(X), "\n")
cat("类别数:", num_class, "\n\n")

# ==================== 计算ROC-AUC的函数 ====================
calculate_auc <- function(actual, prob_matrix) {
  if (ncol(prob_matrix) == 2) {
    roc_obj <- roc(actual, prob_matrix[, 2], quiet = TRUE)
    return(as.numeric(auc(roc_obj)))
  } else {
    auc_values <- numeric(ncol(prob_matrix))
    for (i in 1:ncol(prob_matrix)) {
      binary_actual <- ifelse(actual == (i-1), 1, 0)
      roc_obj <- roc(binary_actual, prob_matrix[, i], quiet = TRUE)
      auc_values[i] <- as.numeric(auc(roc_obj))
    }
    return(mean(auc_values))
  }
}

# ==================== 计算综合评价指标 ====================
calculate_metrics <- function(actual, predicted, prob_matrix) {
  actual <- factor(actual, levels = levels(predicted))
  cm <- confusionMatrix(predicted, actual)
  
  auc_score <- calculate_auc(as.numeric(actual) - 1, prob_matrix)
  
  if (nlevels(actual) > 2) {
    metrics <- list(
      Accuracy = cm$overall["Accuracy"],
      Precision = mean(cm$byClass[,"Pos Pred Value"], na.rm = TRUE),
      Recall = mean(cm$byClass[,"Sensitivity"], na.rm = TRUE),
      F1 = mean(cm$byClass[,"F1"], na.rm = TRUE),
      Kappa = cm$overall["Kappa"],
      Mean_Sensitivity = mean(cm$byClass[,"Sensitivity"], na.rm = TRUE),
      Mean_Specificity = mean(cm$byClass[,"Specificity"], na.rm = TRUE),
      Mean_Pos_Pred_Value = mean(cm$byClass[,"Pos Pred Value"], na.rm = TRUE),
      Mean_Neg_Pred_Value = mean(cm$byClass[,"Neg Pred Value"], na.rm = TRUE),
      Mean_Detection_Rate = mean(cm$byClass[,"Detection Rate"], na.rm = TRUE),
      Mean_Balanced_Accuracy = mean(cm$byClass[,"Balanced Accuracy"], na.rm = TRUE),
      AUC = auc_score
    )
  } else {
    metrics <- list(
      Accuracy = cm$overall["Accuracy"],
      Precision = cm$byClass["Pos Pred Value"],
      Recall = cm$byClass["Sensitivity"],
      F1 = cm$byClass["F1"],
      Kappa = cm$overall["Kappa"],
      Mean_Sensitivity = cm$byClass["Sensitivity"],
      Mean_Specificity = cm$byClass["Specificity"],
      Mean_Pos_Pred_Value = cm$byClass["Pos Pred Value"],
      Mean_Neg_Pred_Value = cm$byClass["Neg Pred Value"],
      Mean_Detection_Rate = cm$byClass["Detection Rate"],
      Mean_Balanced_Accuracy = cm$byClass["Balanced Accuracy"],
      AUC = auc_score
    )
  }
  
  return(metrics)
}

# ==================== 学习曲线分析（过拟合诊断）====================
analyze_learning_curve <- function(X, y, best_params) {
  learning_results <- list()
  
  # 固定验证集
  set.seed(SEED_LEARNING_CURVE)
  n_total <- nrow(X)
  val_size <- floor(n_total * 0.2)
  val_idx <- sample(1:n_total, val_size)
  train_pool_idx <- setdiff(1:n_total, val_idx)
  
  X_val <- X[val_idx, ]
  y_val <- y_factor[val_idx]
  
  for (sample_ratio in LEARNING_CURVE_SAMPLES) {
    n_train <- floor(length(train_pool_idx) * sample_ratio)
    train_idx <- sample(train_pool_idx, n_train)
    
    X_train <- X[train_idx, ]
    y_train <- y_factor[train_idx]
    
    model <- ranger(
      y = y_train,
      x = X_train,
      num.trees = as.integer(best_params$num.trees),
      mtry = as.integer(best_params$mtry),
      min.node.size = as.integer(best_params$min.node.size),
      max.depth = ifelse(best_params$max.depth == 0, NULL, as.integer(best_params$max.depth)),
      probability = TRUE,
      seed = SEED
    )
    
    train_prob <- predict(model, X_train)$predictions
    val_prob <- predict(model, X_val)$predictions
    
    train_auc <- calculate_auc(as.numeric(y_train) - 1, train_prob)
    val_auc <- calculate_auc(as.numeric(y_val) - 1, val_prob)
    
    learning_results[[as.character(sample_ratio)]] <- list(
      sample_size = n_train,
      train_auc = train_auc,
      val_auc = val_auc,
      gap = train_auc - val_auc
    )
  }
  
  return(learning_results)
}

# ==================== 绘制学习曲线 ====================
plot_learning_curve <- function(learning_results, output_dir) {
  df <- do.call(rbind, lapply(names(learning_results), function(ratio) {
    data.frame(
      sample_size = learning_results[[ratio]]$sample_size,
      train_auc = learning_results[[ratio]]$train_auc,
      val_auc = learning_results[[ratio]]$val_auc
    )
  }))
  
  plot_title <- gsub("_", " ", basename(output_dir))
  
  p <- ggplot(df, aes(x = sample_size)) +
    geom_line(aes(y = train_auc, linetype = "Training"), size = 1, color = "black") +
    geom_point(aes(y = train_auc, shape = "Training"), size = 3, color = "black") +
    geom_line(aes(y = val_auc, linetype = "Validation"), size = 1, color = "black") +
    geom_point(aes(y = val_auc, shape = "Validation"), size = 3, color = "black") +
    scale_linetype_manual(values = c("Training" = "solid", "Validation" = "dashed")) +
    scale_shape_manual(values = c("Training" = 16, "Validation" = 17)) +
    labs(
      title = plot_title,
      x = "Training Set Size",
      y = "AUC Score",
      linetype = "Dataset",
      shape = "Dataset"
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      legend.position = "bottom",
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 9),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text = element_text(color = "black"),
      axis.title = element_text(color = "black")
    )
  
  ggsave(file.path(output_dir, "learning_curve.tiff"), p, 
         width = 8, height = 6, dpi = 600, compression = "lzw")
  
  write_xlsx(list(LearningCurve = df), 
             path = file.path(output_dir, "learning_curve.xlsx"))
  
  return(p)
}

# ==================== 绘制交叉验证性能箱线图 ====================
plot_cv_performance <- function(all_metrics, output_dir) {
  metrics_df <- do.call(rbind, lapply(1:length(all_metrics), function(i) {
    data.frame(
      Fold = paste0("Fold", i),
      AUC = all_metrics[[i]]$AUC,
      Accuracy = all_metrics[[i]]$Accuracy,
      Precision = all_metrics[[i]]$Precision,
      Recall = all_metrics[[i]]$Recall,
      F1 = all_metrics[[i]]$F1
    )
  }))
  
  metrics_long <- reshape2::melt(metrics_df, id.vars = "Fold", 
                                 variable.name = "Metric", 
                                 value.name = "Score")
  
  plot_title <- gsub("_", " ", basename(output_dir))
  
  p <- ggplot(metrics_long, aes(x = Metric, y = Score)) +
    geom_boxplot(alpha = 0.7, fill = "grey70", color = "black") +
    geom_jitter(width = 0.2, alpha = 0.5, size = 2, color = "black") +
    stat_summary(fun = mean, geom = "point", shape = 23, size = 3, 
                 fill = "black", color = "black") +
    labs(
      title = plot_title,
      subtitle = "Black diamonds indicate mean values",
      x = "Evaluation Metric",
      y = "Score"
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 10),
      panel.grid.major = element_line(color = "grey80"),
      panel.grid.minor = element_blank(),
      axis.text = element_text(color = "black"),
      axis.title = element_text(color = "black")
    ) +
    ylim(0, 1)
  
  ggsave(file.path(output_dir, "cv_performance_boxplot.tiff"), p, 
         width = 10, height = 6, dpi = 600, compression = "lzw")
  
  write_xlsx(list(CVPerformance = metrics_df), 
             path = file.path(output_dir, "cv_performance_boxplot.xlsx"))
  
  return(p)
}

# ==================== 计算DCA曲线数据 ====================
calculate_dca <- function(actual, prob_positive, thresholds = seq(0, 1, by = 0.01)) {
  dca_data <- data.frame(
    threshold = thresholds,
    net_benefit_model = NA,
    net_benefit_all = NA,
    net_benefit_none = 0
  )
  
  n <- length(actual)
  prevalence <- mean(actual)
  
  for (i in 1:length(thresholds)) {
    threshold <- thresholds[i]
    
    if (threshold == 0) {
      dca_data$net_benefit_model[i] <- prevalence
      dca_data$net_benefit_all[i] <- prevalence
    } else if (threshold == 1) {
      dca_data$net_benefit_model[i] <- 0
      dca_data$net_benefit_all[i] <- 0
    } else {
      predicted_positive <- prob_positive >= threshold
      tp <- sum(predicted_positive & actual == 1)
      fp <- sum(predicted_positive & actual == 0)
      
      net_benefit_model <- (tp / n) - (fp / n) * (threshold / (1 - threshold))
      dca_data$net_benefit_model[i] <- net_benefit_model
      
      net_benefit_all <- prevalence - (1 - prevalence) * (threshold / (1 - threshold))
      dca_data$net_benefit_all[i] <- net_benefit_all
    }
  }
  
  return(dca_data)
}

# ==================== 绘制DCA曲线 ====================
plot_dca <- function(all_predictions, y_factor, output_dir) {
  all_actual <- c()
  all_prob <- c()
  
  for (fold_name in names(all_predictions)) {
    fold_data <- all_predictions[[fold_name]]
    all_actual <- c(all_actual, fold_data$Actual)
    
    prob_cols <- grep("^Prob_", names(fold_data), value = TRUE)
    all_prob <- c(all_prob, fold_data[[prob_cols[length(prob_cols)]]])
  }
  
  actual_binary <- ifelse(all_actual == levels(y_factor)[length(levels(y_factor))], 1, 0)
  
  dca_data <- calculate_dca(actual_binary, all_prob)
  
  dca_long <- data.frame(
    threshold = rep(dca_data$threshold, 3),
    net_benefit = c(dca_data$net_benefit_model, 
                   dca_data$net_benefit_all,
                   dca_data$net_benefit_none),
    strategy = rep(c("Model", "Treat All", "Treat None"), 
                  each = nrow(dca_data))
  )
  
  plot_title <- gsub("_", " ", basename(output_dir))
  
  p <- ggplot(dca_long, aes(x = threshold, y = net_benefit, 
                            color = strategy, linetype = strategy)) +
    geom_line(size = 1) +
    scale_color_manual(values = c("Model" = "black", 
                                  "Treat All" = "darkgray", 
                                  "Treat None" = "lightgray")) +
    scale_linetype_manual(values = c("Model" = "solid", 
                                     "Treat All" = "dashed", 
                                     "Treat None" = "dotted")) +
    labs(
      title = plot_title,
      x = "Threshold Probability",
      y = "Net Benefit",
      color = "Strategy",
      linetype = "Strategy"
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      legend.position = "bottom",
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 9),
      panel.grid.major = element_line(color = "grey80"),
      panel.grid.minor = element_blank(),
      axis.text = element_text(color = "black"),
      axis.title = element_text(color = "black")
    ) +
    coord_cartesian(ylim = c(min(dca_long$net_benefit[is.finite(dca_long$net_benefit)], na.rm = TRUE) - 0.05, 
                             max(dca_long$net_benefit[is.finite(dca_long$net_benefit)], na.rm = TRUE) + 0.05))
  
  ggsave(file.path(output_dir, "dca_curve.tiff"), p, 
         width = 8, height = 6, dpi = 600, compression = "lzw")
  
  write_xlsx(list(DCA = dca_data), 
             path = file.path(output_dir, "dca_curve.xlsx"))
  
  return(p)
}

# ==================== 主训练流程 ====================
cat("开始模型训练...\n\n")

folds <- sample(rep(1:CV_FOLDS, length.out = nrow(X)))

all_predictions <- list()
all_metrics <- list()
all_models <- list()
all_best_params <- list()

for (fold in 1:CV_FOLDS) {
  cat(sprintf("========== Fold %d/%d ==========\n", fold, CV_FOLDS))
  
  train_idx <- which(folds != fold)
  test_idx <- which(folds == fold)
  
  X_train <- X[train_idx, ]
  y_train <- y_factor[train_idx]
  X_test <- X[test_idx, ]
  y_test <- y_factor[test_idx]
  
  best_model <- NULL
  best_score <- 0
  best_params <- NULL
  best_metrics <- NULL
  
  pb <- txtProgressBar(min = 0, max = nrow(RF_GRID), style = 3)
  
  for (i in 1:nrow(RF_GRID)) {
    model <- ranger(
      y = y_train,
      x = X_train,
      num.trees = as.integer(RF_GRID$num.trees[i]),
      mtry = as.integer(RF_GRID$mtry[i]),
      min.node.size = as.integer(RF_GRID$min.node.size[i]),
      max.depth = ifelse(RF_GRID$max.depth[i] == 0, NULL, as.integer(RF_GRID$max.depth[i])),
      probability = TRUE,
      seed = SEED
    )
    
    pred_prob <- predict(model, X_test)$predictions
    pred_class <- apply(pred_prob, 1, which.max)
    pred_factor <- factor(pred_class, levels = 1:num_class, labels = levels(y_factor))
    
    metrics <- calculate_metrics(y_test, pred_factor, pred_prob)
    
    current_score <- metrics[[SELECTION_METRIC]]
    
    if (current_score > best_score) {
      best_score <- current_score
      best_model <- model
      best_params <- RF_GRID[i, ]
      best_metrics <- metrics
    }
    
    setTxtProgressBar(pb, i)
  }
  close(pb)
  
  cat(sprintf("\n最佳%s: %.4f\n", SELECTION_METRIC, best_score))
  
  all_models[[fold]] <- best_model
  all_best_params[[fold]] <- best_params
  saveRDS(best_model, file = file.path(OUTPUT_DIR, paste0("Fold", fold, "_best_model.rds")))
  
  final_prob_matrix <- predict(best_model, X_test)$predictions
  final_pred_class <- apply(final_prob_matrix, 1, which.max)
  final_pred_factor <- factor(final_pred_class, levels = 1:num_class, labels = levels(y_factor))
  
  all_metrics[[fold]] <- best_metrics
  
  result_df <- data.frame(
    RowIndex = test_idx,
    Actual = y[test_idx],
    Predicted = as.character(final_pred_factor)
  )
  
  for (class in 1:ncol(final_prob_matrix)) {
    result_df[[paste0("Prob_", levels(y_factor)[class])]] <- final_prob_matrix[, class]
  }
  
  all_predictions[[paste0("Fold", fold)]] <- result_df
}

# ==================== 保存每折预测结果 ====================
write_xlsx(all_predictions, path = file.path(OUTPUT_DIR, "每折预测结果.xlsx"))

# ==================== 选择最佳折叠 ====================
fold_scores <- sapply(all_metrics, function(x) x[[SELECTION_METRIC]])
best_fold_idx <- which.max(fold_scores)

cat(sprintf("\n找到最佳模型：Fold%d (%s: %.4f)\n", 
            best_fold_idx, SELECTION_METRIC, fold_scores[best_fold_idx]))

best_model_final <- all_models[[best_fold_idx]]
saveRDS(best_model_final, file = file.path(OUTPUT_DIR, "best_model.rds"))

# ==================== 计算平均指标 ====================
avg_metrics <- list()
metric_names <- names(all_metrics[[1]])
for (name in metric_names) {
  values <- sapply(all_metrics, function(x) x[[name]])
  avg_metrics[[name]] <- mean(values)
}

# ==================== 学习曲线分析 ====================
best_params_for_lc <- all_best_params[[best_fold_idx]]
learning_curve_results <- analyze_learning_curve(X, y, best_params_for_lc)
plot_learning_curve(learning_curve_results, OUTPUT_DIR)

# ==================== 绘制交叉验证性能箱线图 ====================
plot_cv_performance(all_metrics, OUTPUT_DIR)

# ==================== 绘制DCA曲线 ====================
plot_dca(all_predictions, y_factor, OUTPUT_DIR)

# ==================== 特征重要性分析 ====================
importance_vec <- best_model_final$variable.importance
importance_df <- data.frame(
  Feature = names(importance_vec),
  Importance = as.numeric(importance_vec),
  stringsAsFactors = FALSE
)
importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]
rownames(importance_df) <- NULL

# 绘制特征重要性图
#tiff(file.path(OUTPUT_DIR, "feature_importance.tiff"), 
#     width = 800, height = 600, res = 600, compression = "lzw")
#par(mar = c(5, 8, 4, 2))
#barplot(importance_df$Importance[1:min(20, nrow(importance_df))], 
#        names.arg = importance_df$Feature[1:min(20, nrow(importance_df))],
#        horiz = TRUE, las = 1,
#        main = gsub("_", " ", basename(OUTPUT_DIR)),
#        xlab = "Variable Importance",
#        col = "steelblue")
#dev.off()

# 保存数据到Excel
#write_xlsx(list(Importance = importance_df), 
#           path = file.path(OUTPUT_DIR, "feature_importance.xlsx"))

# ==================== 保存optimal_parameters.txt ====================
optimal_params_file <- file.path(OUTPUT_DIR, "optimal_parameters.txt")
sink(optimal_params_file)
cat("# 最优随机森林参数 #\n\n")
cat("特征选择方法: 原始数据\n")
cat("模型选择指标:", SELECTION_METRIC, "\n\n")
cat("参数值:\n")
best_params_final <- all_best_params[[best_fold_idx]]
for (param_name in names(best_params_final)) {
  cat(sprintf("%s = %s\n", param_name, best_params_final[[param_name]]))
}

metrics_to_check <- c(
  "Precision" = "精确率 (Precision)",
  "Recall" = "召回率 (Recall)",
  "F1" = "F1分数",
  "Kappa" = "Kappa系数",
  "Mean_Sensitivity" = "平均敏感度",
  "Mean_Specificity" = "平均特异度",
  "Mean_Pos_Pred_Value" = "平均正预测值",
  "Mean_Neg_Pred_Value" = "平均负预测值",
  "Mean_Detection_Rate" = "平均检测率",
  "Mean_Balanced_Accuracy" = "平均平衡准确率"
)

cat("\n平均性能指标:\n")
cat(sprintf("准确率 (Accuracy): %.4f\n", avg_metrics$Accuracy))

for (metric in names(metrics_to_check)) {
  if (metric %in% names(avg_metrics)) {
    cat(sprintf("%s: %.4f\n", metrics_to_check[metric], avg_metrics[[metric]]))
  }
}

cat(sprintf("AUC: %.4f\n", avg_metrics$AUC))

cat("\n选中的特征 (共", ncol(X), "个):\n")
cat(paste(colnames(X), collapse = ", "), "\n")

cat("\n最佳模型来自：Fold", best_fold_idx, "\n")
cat(sprintf("最佳折叠%s：%.4f\n", SELECTION_METRIC, fold_scores[best_fold_idx]))

cat("\n## 过拟合诊断 (学习曲线分析) ##\n")
for (ratio in names(learning_curve_results)) {
  result <- learning_curve_results[[ratio]]
  cat(sprintf("样本比例 %.1f%% (n=%d): 训练AUC=%.4f, 验证AUC=%.4f, 差距=%.4f\n",
              as.numeric(ratio) * 100, result$sample_size, 
              result$train_auc, result$val_auc, result$gap))
}

sink()

# ==================== 保存结果到RDS ====================
results <- list(
  selected_features = colnames(X),
  feature_scores = setNames(rep(1, ncol(X)), colnames(X)),
  best_params = best_params_final,
  metrics = avg_metrics,
  predictions = all_predictions,
  models = all_models
)

saveRDS(results, file = file.path(OUTPUT_DIR, "feature_selection_results.rds"))

cat("\n==================== 训练完成 ====================\n")
cat("所有结果已保存到:", OUTPUT_DIR, "\n")