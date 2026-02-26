# 加载必要的库
library(readxl)
library(xgboost)
library(SHAPforxgboost)
library(ggplot2)
library(dplyr)
library(reshape2)
library(writexl)
library(gridExtra)
library(grid)
library(viridis)
library(tidyr)
library(broom)
library(caret)

# 设置工作路径
setwd("")
shap_dir <- ""
dir.create(shap_dir, showWarnings = FALSE)

# 读取原始数据
data <- read_excel("")
y <- data[[1]]
X <- as.data.frame(data[, -1])

# 检查是否为二分类问题
is_binary <- length(unique(y)) == 2
if(!is_binary) {
  stop("当前数据不是二分类问题")
}

# 获取所有XGBoost模型文件夹
model_folders <- list.dirs(path = ".", full.names = TRUE, recursive = FALSE)
model_folders <- model_folders[grepl("XGBoost_", model_folders)]

# 函数:在交叉验证中计算SHAP值
analyze_shap_cv <- function(model_folder, X, y, method_name, n_folds = 5) {
  cat(paste0("\n分析 ", method_name, " 模型的SHAP值 (交叉验证方式)...\n"))
  
  # 创建模型特定的结果目录
  result_dir <- file.path(shap_dir, basename(model_folder))
  dir.create(result_dir, showWarnings = FALSE)
  
  # 加载模型参数
  params_file <- file.path(model_folder, "optimal_parameters.txt")
  if (!file.exists(params_file)) {
    cat(paste0("警告: 找不到参数文件 ", params_file, "\n"))
    return(NULL)
  }
  
  params_text <- readLines(params_file)
  
  # 从参数文件提取特征
  feature_line_idx <- grep("选中的特征", params_text)
  if (length(feature_line_idx) > 0) {
    end_idx <- grep("---END_FEATURES---", params_text)
    feature_lines <- params_text[(feature_line_idx+1):(end_idx-1)]
    feature_lines <- paste(feature_lines, collapse = " ")
    selected_features <- unlist(strsplit(feature_lines, ","))
    selected_features <- trimws(selected_features)
    selected_features <- selected_features[nchar(selected_features) > 0]
    
    missing_features <- selected_features[!selected_features %in% colnames(X)]
    if (length(missing_features) > 0) {
      cat("警告: 以下选中的特征在数据中不存在:", paste(missing_features, collapse=", "), "\n")
      selected_features <- selected_features[selected_features %in% colnames(X)]
    }
  } else {
    cat("参数文件中未找到特征信息,使用所有特征\n")
    selected_features <- colnames(X)
  }
  
  if (length(selected_features) == 0) {
    cat("错误: 没有有效的特征可供分析\n")
    return(NULL)
  }
  
  # 准备用于SHAP分析的数据
  X_selected <- X[, selected_features, drop = FALSE]
  
  
  # 提取XGBoost超参数
  xgb_params_result <- extract_xgb_params(params_text)
  xgb_params <- xgb_params_result$params
  nrounds <- xgb_params_result$nrounds
  
  cat(sprintf("XGBoost参数: eta=%.3f, max_depth=%d, nrounds=%d\n", 
              xgb_params$eta, xgb_params$max_depth, nrounds))
  
  # 创建交叉验证折叠
  set.seed(123)
  folds <- createFolds(y, k = n_folds, list = TRUE)
  
  # 存储每个折叠的SHAP结果
  fold_shap_values <- list()
  fold_feature_importance <- list()
  fold_models <- list()
  
  # 对每个折叠进行分析
  for (fold_idx in 1:n_folds) {
    cat(sprintf("\n处理折叠 %d/%d...\n", fold_idx, n_folds))
    
    # 划分训练集和验证集
    val_indices <- folds[[fold_idx]]
    train_indices <- setdiff(1:nrow(X_selected), val_indices)
    
    X_train <- X_selected[train_indices, , drop = FALSE]
    X_val <- X_selected[val_indices, , drop = FALSE]
    y_train <- y[train_indices]
    y_val <- y[val_indices]
    
    # 确保特征顺序一致
    X_train <- X_train[, selected_features, drop = FALSE]
    X_val <- X_val[, selected_features, drop = FALSE]
    model_feature_names <- selected_features
    
    # 检查数据有效性
    if (any(is.na(X_train)) || any(is.na(X_val))) {
      cat("警告: 数据中存在NA值,进行填充...\n")
      X_train[is.na(X_train)] <- 0
      X_val[is.na(X_val)] <- 0
    }
    
    # 创建DMatrix
    tryCatch({
      dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
      dval <- xgb.DMatrix(data = as.matrix(X_val), label = y_val)
      
      # 设置特征名称
      setinfo(dtrain, "feature_name", model_feature_names)
      setinfo(dval, "feature_name", model_feature_names)
      
      # 训练XGBoost模型
      cat("训练折叠模型...\n")
      xgb_model <- xgb.train(
        params = xgb_params,
        data = dtrain,
        nrounds = nrounds,
        watchlist = list(train = dtrain, val = dval),
        early_stopping_rounds = 10,
        verbose = 0,
        print_every_n = 0
      )
      
      fold_models[[fold_idx]] <- xgb_model
      
      # 计算验证集的SHAP值
      cat("计算SHAP值...\n")
      shap_contrib <- predict(xgb_model, dval, predcontrib = TRUE)
      
      # 提取SHAP值(排除BIAS列)
      shap_scores <- shap_contrib[, 1:ncol(X_val), drop = FALSE]
      colnames(shap_scores) <- colnames(X_val)
      
      # 计算基准值
      base_value <- mean(predict(xgb_model, dval))
      
      # 存储折叠结果
      fold_shap_values[[fold_idx]] <- list(
        shap_score = shap_scores,
        mean_value = base_value,
        val_indices = val_indices
      )
      
      # 计算特征重要性
      feature_imp <- colMeans(abs(shap_scores))
      fold_feature_importance[[fold_idx]] <- data.frame(
        Feature = names(feature_imp),
        Importance = as.numeric(feature_imp),
        Fold = fold_idx
      )
      
      cat(sprintf("折叠 %d 完成,基准值: %.4f\n", fold_idx, base_value))
      
    }, error = function(e) {
      cat(sprintf("折叠 %d 训练失败: %s\n", fold_idx, e$message))
      cat("数据维度: X_train =", paste(dim(X_train), collapse="x"), 
          ", X_val =", paste(dim(X_val), collapse="x"), "\n")
      cat("标签长度: y_train =", length(y_train), ", y_val =", length(y_val), "\n")
      cat("特征名称:", paste(model_feature_names, collapse=", "), "\n")
      return(NULL)
    })
  }
  
  # 保存各折叠的原始结果
  saveRDS(fold_shap_values, file.path(result_dir, "fold_shap_values.rds"))
  saveRDS(fold_models, file.path(result_dir, "fold_models.rds"))
  
  # 检查是否有成功的折叠
  successful_folds <- sapply(fold_shap_values, function(x) !is.null(x))
  if (sum(successful_folds) == 0) {
    cat("错误: 所有折叠都训练失败,无法进行聚合分析\n")
    return(NULL)
  }
  
  if (sum(successful_folds) < n_folds) {
    cat(sprintf("警告: 只有 %d/%d 个折叠训练成功\n", sum(successful_folds), n_folds))
  }
  
  # 聚合SHAP值结果(只使用成功的折叠)
  cat("\n聚合交叉验证结果...\n")
  aggregated_results <- aggregate_cv_shap(fold_shap_values[successful_folds], 
                                          fold_feature_importance[successful_folds], 
                                          X_selected, result_dir, method_name)
  
  # 生成可视化
  generate_cv_visualizations(aggregated_results, fold_shap_values, X_selected, 
                            result_dir, method_name)
  
  # 生成特征交互分析
  generate_feature_interactions_cv(aggregated_results, fold_shap_values, X_selected,
                                   result_dir, method_name)
  
  cat(paste0(method_name, " 的交叉验证SHAP分析完成,结果保存在 ", result_dir, "\n"))
  
  return(aggregated_results)
}

# 函数:从参数文本中提取XGBoost参数
extract_xgb_params <- function(params_text) {
  params <- list()
  
  # 提取常见参数的辅助函数
  extract_param <- function(param_name, default_value, as_numeric = TRUE) {
    line <- grep(param_name, params_text, value = TRUE, ignore.case = TRUE)
    if (length(line) > 0) {
      # 提取冒号后的值
      value <- sub(".*[=:]\\s*", "", line[1])
      value <- trimws(value)
      if (as_numeric) {
        value <- as.numeric(value)
        if (is.na(value)) return(default_value)
      }
      return(value)
    }
    return(default_value)
  }
  
  # 提取参数
  params$eta <- extract_param("eta|学习率|learning.rate", 0.1)
  params$max_depth <- extract_param("max_depth|最大深度|max.depth", 6)
  params$min_child_weight <- extract_param("min_child_weight|min.child.weight", 1)
  params$subsample <- extract_param("subsample|子样本", 0.8)
  params$colsample_bytree <- extract_param("colsample_bytree|colsample.bytree", 0.8)
  params$gamma <- extract_param("gamma", 0)
  params$lambda <- extract_param("lambda|正则化", 1)
  params$alpha <- extract_param("alpha", 0)
  
  # 迭代次数
  nrounds <- extract_param("nrounds|迭代次数|n.rounds|num.boost.round", 100)
  
  # 构建参数列表(不包括nrounds)
  params_list <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = params$eta,
    max_depth = as.integer(params$max_depth),
    min_child_weight = params$min_child_weight,
    subsample = params$subsample,
    colsample_bytree = params$colsample_bytree,
    gamma = params$gamma,
    lambda = params$lambda,
    alpha = params$alpha
  )
  
  return(list(params = params_list, nrounds = as.integer(nrounds)))
}

# 函数:聚合交叉验证的SHAP结果
aggregate_cv_shap <- function(fold_shap_values, fold_feature_importance, 
                               X_selected, result_dir, method_name) {
  
  # 聚合特征重要性
  all_importance <- do.call(rbind, fold_feature_importance)
  
  importance_summary <- all_importance %>%
    group_by(Feature) %>%
    summarize(
      Mean_Importance = mean(Importance),
      SD_Importance = sd(Importance),
      Min_Importance = min(Importance),
      Max_Importance = max(Importance),
      CV_Importance = sd(Importance) / mean(Importance),  # 变异系数
      N_Folds = n()  # 出现在多少个折叠中
    ) %>%
    arrange(desc(Mean_Importance))
  
  # 保存聚合的特征重要性
  write.csv(importance_summary, 
            file.path(result_dir, "feature_importance_aggregated.csv"), 
            row.names = FALSE)
  write.csv(all_importance, 
            file.path(result_dir, "feature_importance_all_folds.csv"), 
            row.names = FALSE)
  
  # 聚合SHAP值(通过样本索引重新组合)
  # 获取第一个成功折叠的特征列表
  first_fold <- fold_shap_values[[1]]
  feature_names <- colnames(first_fold$shap_score)
  n_features <- length(feature_names)
  n_samples <- nrow(X_selected)
  
  aggregated_shap <- matrix(NA, nrow = n_samples, ncol = n_features)
  colnames(aggregated_shap) <- feature_names
  
  for (fold_idx in 1:length(fold_shap_values)) {
    fold_result <- fold_shap_values[[fold_idx]]
    if (!is.null(fold_result)) {
      val_indices <- fold_result$val_indices
      aggregated_shap[val_indices, ] <- fold_result$shap_score
    }
  }
  
  # 保存聚合的SHAP值
  saveRDS(list(shap_score = aggregated_shap, 
               mean_value = mean(sapply(fold_shap_values, function(x) {
                 if (!is.null(x)) x$mean_value else NA
               }), na.rm = TRUE)),
          file.path(result_dir, "aggregated_shap_values.rds"))
  
  return(list(
    importance_summary = importance_summary,
    all_importance = all_importance,
    aggregated_shap = aggregated_shap,
    feature_names = feature_names
  ))
}

# 函数:生成交叉验证的可视化
generate_cv_visualizations <- function(aggregated_results, fold_shap_values, 
                                       X_selected, result_dir, method_name) {
  
  importance_summary <- aggregated_results$importance_summary
  all_importance <- aggregated_results$all_importance
  aggregated_shap <- aggregated_results$aggregated_shap
  feature_names <- aggregated_results$feature_names
  
  # 特征重要性条形图
  top_n <- min(20, nrow(importance_summary))
  top_features <- importance_summary$Feature[1:top_n]
  
  p1 <- ggplot(importance_summary[1:top_n, ], 
               aes(x = reorder(Feature, Mean_Importance), y = Mean_Importance)) +
    geom_bar(stat = "identity", fill = "gray40", alpha = 0.8) +
    geom_errorbar(aes(ymin = pmax(Mean_Importance - SD_Importance, 0), 
                     ymax = Mean_Importance + SD_Importance),
                 width = 0.3, color = "black", linewidth = 0.5) +
    coord_flip() +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
    labs(title = paste0(method_name, " - Feature Importance (Cross-Validation)"),
         x = "", y = "Mean |SHAP value| ± SD") +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.text = element_text(size = 10, color = "black"),
      axis.title = element_text(size = 12, color = "black"),
      panel.grid = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black", linewidth = 0.5)
    )
  
  ggsave(file.path(result_dir, "feature_importance_cv.tiff"), p1, 
         width = 10, height = 8, dpi = 600, device = "tiff", compression = "lzw")
  
  # 各折叠特征重要性对比图
  top_features_plot <- all_importance %>% 
    filter(Feature %in% top_features)
  
  p2 <- ggplot(top_features_plot, 
               aes(x = reorder(Feature, Importance, FUN = mean), 
                   y = Importance, fill = as.factor(Fold))) +
    geom_boxplot(alpha = 0.7) +
    coord_flip() +
    labs(title = paste0(method_name, " - Feature Importance Distribution Across Folds"),
         x = "", y = "SHAP Importance", fill = "Fold") +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.text = element_text(size = 10, color = "black"),
      axis.title = element_text(size = 12, color = "black"),
      legend.title = element_text(size = 11),
      legend.text = element_text(size = 10),
      panel.grid = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black", linewidth = 0.5)
    )
  
  ggsave(file.path(result_dir, "feature_importance_folds_boxplot.tiff"), 
         p2, width = 12, height = 8, dpi = 600, device = "tiff", compression = "lzw")
  
  #SHAP摘要图
  valid_rows <- complete.cases(aggregated_shap)
  
  if (sum(valid_rows) == 0) {
    cat("Warning: No complete SHAP values, skipping summary plot\n")
    return(invisible(NULL))
  }
  
  shap_long <- as.data.frame(aggregated_shap[valid_rows, , drop = FALSE])
  shap_long$row_id <- which(valid_rows)
  shap_long <- reshape2::melt(shap_long, id.vars = "row_id", 
                              variable.name = "Feature", value.name = "SHAP_value")
  
  features_in_x <- feature_names[feature_names %in% colnames(X_selected)]
  
  if (length(features_in_x) > 0) {
    feature_values <- X_selected[valid_rows, features_in_x, drop = FALSE]
    feature_values$row_id <- which(valid_rows)
    feature_long <- reshape2::melt(feature_values, id.vars = "row_id", 
                                  variable.name = "Feature", value.name = "Feature_value")
    
    plot_data <- merge(shap_long, feature_long, by = c("row_id", "Feature"), all.x = TRUE)
    
    plot_data <- plot_data %>%
      group_by(Feature) %>%
      mutate(Feature_value_norm = (Feature_value - min(Feature_value, na.rm = TRUE)) / 
               (max(Feature_value, na.rm = TRUE) - min(Feature_value, na.rm = TRUE) + 1e-8)) %>%
      ungroup()
    
    top_features_in_data <- intersect(top_features, features_in_x)
    plot_data_subset <- plot_data %>% filter(Feature %in% top_features_in_data)
    
    if (nrow(plot_data_subset) > 0) {
      p3 <- ggplot(plot_data_subset, 
                   aes(x = reorder(Feature, abs(SHAP_value), FUN = function(x) mean(abs(x), na.rm = TRUE)), 
                       y = SHAP_value, color = Feature_value_norm)) +
        geom_jitter(width = 0.1, alpha = 0.6, size = 1.5) +
        scale_color_gradient2(low = "blue", mid = "#B2FFFC", high = "red", midpoint = 0.5) +
        coord_flip() +
        labs(title = "SHAP Summary (Cross-Validation Aggregated)",
             x = "", y = "SHAP value", color = "Standardized\nFeature Value") +
        theme_minimal() +
        theme(
          plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          axis.text = element_text(size = 10, color = "black"),
          axis.title = element_text(size = 12, color = "black"),
          legend.title = element_text(size = 11),
          legend.text = element_text(size = 10),
          panel.grid = element_blank(),
          axis.line = element_line(color = "black", linewidth = 0.5),
          axis.ticks = element_line(color = "black", linewidth = 0.5)
        )
      
      ggsave(file.path(result_dir, "shap_summary_cv.tiff"), p3, 
             width = 10, height = 8, dpi = 600, device = "tiff", compression = "lzw")
      write.csv(plot_data_subset, file.path(result_dir, "shap_summary_data.csv"), row.names = FALSE)
    }
  }
  
  #稳定性分析图
  p4 <- ggplot(importance_summary[1:top_n, ], 
               aes(x = reorder(Feature, CV_Importance), y = CV_Importance)) +
    geom_bar(stat = "identity", fill = "coral", alpha = 0.7) +
    geom_hline(yintercept = 0.3, linetype = "dashed", color = "red", linewidth = 0.8) +
    coord_flip() +
    labs(title = paste0(method_name, " - Feature Importance Stability"),
         subtitle = "Coefficient of Variation (CV) = SD/Mean, lower is more stable",
         x = "", y = "Coefficient of Variation") +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 10),
      axis.text = element_text(size = 10, color = "black"),
      axis.title = element_text(size = 12, color = "black"),
      panel.grid = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black", linewidth = 0.5)
    )
  
  ggsave(file.path(result_dir, "feature_stability_cv.tiff"), p4, 
         width = 10, height = 8, dpi = 600, device = "tiff", compression = "lzw")
  
  #依赖图
  if (exists("plot_data") && nrow(plot_data) > 0) {
    dep_features <- intersect(head(top_features, 10), features_in_x)
    
    for (i in 1:length(dep_features)) {
      feat <- dep_features[i]
      feat_data <- plot_data[plot_data$Feature == feat & !is.na(plot_data$Feature_value), ]
      
      if (nrow(feat_data) > 10) {
        # 计算X轴刻度
        x_min <- floor(min(feat_data$Feature_value, na.rm = TRUE))
        x_max <- ceiling(max(feat_data$Feature_value, na.rm = TRUE))
        x_range <- x_max - x_min
        
        # 调整刻度间隔
        if (x_range > 20) {
          x_breaks <- seq(x_min, x_max, by = ceiling(x_range / 10))
        } else {
          x_breaks <- seq(x_min, x_max, by = 1)
        }
        
        p_dep <- ggplot(feat_data, aes(x = Feature_value, y = SHAP_value)) +
          geom_point(alpha = 0.5, size = 2, color = "gray30") +
          geom_smooth(method = "loess", formula = y ~ x, se = TRUE, 
                     color = "red", fill = "pink", alpha = 0.3, linewidth = 1) +
          scale_x_continuous(breaks = x_breaks, labels = as.integer(x_breaks)) +
          labs(x = feat, y = "SHAP value") +
          theme_minimal() +
          theme(
            axis.text = element_text(size = 11, color = "black"),
            axis.title = element_text(size = 13, color = "black", face = "bold"),
            axis.title.x = element_text(margin = margin(t = 10)),
            axis.title.y = element_text(margin = margin(r = 10)),
            panel.grid = element_blank(),
            axis.line = element_line(color = "black", linewidth = 0.5),
            axis.ticks = element_line(color = "black", linewidth = 0.5),
            panel.background = element_rect(fill = "white", color = NA),
            plot.background = element_rect(fill = "white", color = NA)
          )
        
        ggsave(file.path(result_dir, paste0("dependency_cv_", gsub("[^a-zA-Z0-9]", "_", feat), ".tiff")), 
               p_dep, width = 8, height = 6, dpi = 600, device = "tiff", compression = "lzw")
        
        # 纯散点图
        p_dep_scatter <- ggplot(feat_data, aes(x = Feature_value, y = SHAP_value)) +
          geom_point(alpha = 0.5, size = 2, color = "gray30") +
          scale_x_continuous(breaks = x_breaks, labels = as.integer(x_breaks)) +
          labs(x = feat, y = "SHAP value") +
          theme_minimal() +
          theme(
            axis.text = element_text(size = 11, color = "black"),
            axis.title = element_text(size = 13, color = "black", face = "bold"),
            axis.title.x = element_text(margin = margin(t = 10)),
            axis.title.y = element_text(margin = margin(r = 10)),
            panel.grid = element_blank(),
            axis.line = element_line(color = "black", linewidth = 0.5),
            axis.ticks = element_line(color = "black", linewidth = 0.5),
            panel.background = element_rect(fill = "white", color = NA),
            plot.background = element_rect(fill = "white", color = NA)
          )
        
        ggsave(file.path(result_dir, paste0("dependency_cv_", gsub("[^a-zA-Z0-9]", "_", feat), "_scatter.tiff")), 
               p_dep_scatter, width = 8, height = 6, dpi = 600, device = "tiff", compression = "lzw")
        
        write.csv(feat_data, file.path(result_dir, paste0("dependency_cv_", gsub("[^a-zA-Z0-9]", "_", feat), "_data.csv")), 
                  row.names = FALSE)
      }
    }
  }
  
  #总结报告
  sink(file.path(result_dir, "shap_cv_analysis_summary.txt"))
  cat(paste0("# ", method_name, " Cross-Validation SHAP Analysis Summary #\n\n"))
  
  cat("## Aggregated Feature Importance Ranking (Mean ± SD) ##\n")
  for (i in 1:min(20, nrow(importance_summary))) {
    feat <- importance_summary$Feature[i]
    mean_imp <- importance_summary$Mean_Importance[i]
    sd_imp <- importance_summary$SD_Importance[i]
    cv_imp <- importance_summary$CV_Importance[i]
    n_folds <- importance_summary$N_Folds[i]
    cat(sprintf("%2d. %-30s: %.4f ± %.4f (CV=%.2f, appears in %d folds)\n", 
                i, feat, mean_imp, sd_imp, cv_imp, n_folds))
  }
  
  cat("\n## Analysis Conclusions ##\n")
  cat("1. Most influential feature for model prediction:", importance_summary$Feature[1], "\n")
  cat("2. Top 5 most important features:", 
      paste(importance_summary$Feature[1:min(5, nrow(importance_summary))], collapse = ", "), "\n")
  
  stable_features <- importance_summary %>% filter(CV_Importance < 0.3)
  if (nrow(stable_features) > 0) {
    cat("3. High stability features (CV<0.3):", 
        paste(head(stable_features$Feature, 5), collapse = ", "), "\n")
  } else {
    cat("3. No high stability features with CV<0.3\n")
  }
  
  cat("\n## Recommendations ##\n")
  cat("- Cross-validation ensures SHAP analysis stability and avoids overfitting risks\n")
  cat("- Focus on features with low coefficient of variation for more reliable interpretations\n")
  cat("- Features with high CV values should be interpreted more cautiously\n")
  sink()
}

# 函数:特征交互分析
generate_feature_interactions_cv <- function(aggregated_results, fold_shap_values,
                                            X_selected, result_dir, method_name) {
  
  cat("\n生成特征交互分析...\n")
  
  importance_summary <- aggregated_results$importance_summary
  aggregated_shap <- aggregated_results$aggregated_shap
  feature_names <- aggregated_results$feature_names
  
  # 获取在原始数据中存在的特征
  features_in_x <- feature_names[feature_names %in% colnames(X_selected)]
  top_features_all <- importance_summary$Feature
  top_features_dep <- intersect(top_features_all, features_in_x)
  
  # 显示可用特征列表
  cat("\n可用于交互分析的特征列表:\n")
  for (i in 1:min(20, length(top_features_dep))) {
    cat(sprintf("%2d. %s\n", i, top_features_dep[i]))
  }
  
  # ====================================================================
  # 手动设置要分析的特征对组合
  # 格式: c(特征1索引, 特征2索引)  ===================================================================
  feature_pairs <- list(
    c(3, 1),   # 特征3 vs 特征1
    c(3, 2),   # 特征3 vs 特征2
    c(3, 4),   # 特征3 vs 特征4
    c(3, 5),   # 特征3 vs 特征5
    c(3, 6),   # 特征3 vs 特征6
    c(3, 8),   # 特征3 vs 特征8
    c(3, 7),   # 特征3 vs 特征7
    c(3, 9),   # 特征3 vs 特征9
    c(3, 10)   # 特征3 vs 特征10
   )
  # ====================================================================
  
  # 创建交互分析子目录
  interaction_dir <- file.path(result_dir, "feature_interactions_cv")
  dir.create(interaction_dir, showWarnings = FALSE)
  
  # 对每个特征对生成交互分析
  for (pair_idx in 1:length(feature_pairs)) {
    selected <- feature_pairs[[pair_idx]]
    
    # 验证选择
    if (length(selected) != 2 || any(is.na(selected)) || 
        any(selected < 1) || any(selected > length(top_features_dep))) {
      cat(sprintf("特征对 #%d 的序号无效,跳过此组合\n", pair_idx))
      next
    }
    
    # 获取选择的特征
    feat1 <- top_features_dep[selected[1]]
    feat2 <- top_features_dep[selected[2]]
    
    cat(sprintf("\n分析特征交互 #%d: %s (%d) vs %s (%d)\n", 
                pair_idx, feat1, selected[1], feat2, selected[2]))
    
    # 检查特征是否在原始数据中
    if (!feat1 %in% colnames(X_selected)) {
      cat(sprintf("警告: 特征'%s'不在原始数据中,跳过\n", feat1))
      next
    }
    if (!feat2 %in% colnames(X_selected)) {
      cat(sprintf("警告: 特征'%s'不在原始数据中,跳过\n", feat2))
      next
    }
    
    # 提取这两个特征的值
    x1 <- X_selected[[feat1]]
    x2 <- X_selected[[feat2]]
    
    # 提取聚合的SHAP值
    feat1_idx <- which(colnames(aggregated_shap) == feat1)
    feat2_idx <- which(colnames(aggregated_shap) == feat2)
    
    if (length(feat1_idx) == 0 || length(feat2_idx) == 0) {
      cat(sprintf("警告: 无法找到特征的SHAP值索引,跳过\n"))
      next
    }
    
    shap1 <- aggregated_shap[, feat1_idx]
    shap2 <- aggregated_shap[, feat2_idx]
    
    # 创建交互数据框
    valid_rows <- !is.na(shap1) & !is.na(shap2)
    
    if (sum(valid_rows) < 10) {
      cat(sprintf("警告: 有效数据点太少(%d个),跳过此特征对\n", sum(valid_rows)))
      next
    }
    
    interaction_data <- data.frame(
      Feature1_value = x1[valid_rows],
      Feature2_value = x2[valid_rows],
      SHAP1 = shap1[valid_rows],
      SHAP2 = shap2[valid_rows],
      Combined_SHAP = shap1[valid_rows] + shap2[valid_rows]
    )
    
    # 计算特征1和特征2的标准化值
    interaction_data$Feature1_norm <- (interaction_data$Feature1_value - min(interaction_data$Feature1_value)) / 
      (max(interaction_data$Feature1_value) - min(interaction_data$Feature1_value) + 1e-8)
    
    interaction_data$Feature2_norm <- (interaction_data$Feature2_value - min(interaction_data$Feature2_value)) / 
      (max(interaction_data$Feature2_value) - min(interaction_data$Feature2_value) + 1e-8)
    
    # ===== 聚合各折叠的交互数据 =====
    # 收集每个折叠的交互强度
    fold_interaction_strength <- data.frame()
    
    for (fold_idx in 1:length(fold_shap_values)) {
      fold_result <- fold_shap_values[[fold_idx]]
      if (!is.null(fold_result)) {
        val_indices <- fold_result$val_indices
        
        # 检查特征是否在该折叠中
        if (feat1 %in% colnames(fold_result$shap_score) && 
            feat2 %in% colnames(fold_result$shap_score)) {
          
          fold_shap1 <- fold_result$shap_score[, feat1]
          fold_shap2 <- fold_result$shap_score[, feat2]
          
          # 计算交互强度(Combined SHAP的方差)
          combined_shap <- fold_shap1 + fold_shap2
          interaction_strength <- sd(combined_shap, na.rm = TRUE)
          
          fold_interaction_strength <- rbind(fold_interaction_strength, 
                                            data.frame(
                                              Fold = fold_idx,
                                              Interaction_Strength = interaction_strength,
                                              Mean_Combined_SHAP = mean(combined_shap, na.rm = TRUE),
                                              SD_Combined_SHAP = sd(combined_shap, na.rm = TRUE)
                                            ))
        }
      }
    }
    
    # 保存交互数据到CSV文件
    interaction_data_file <- file.path(interaction_dir, 
                                      paste0("interaction_", pair_idx, "_",
                                             gsub("[^[:alnum:]]", "_", feat1), 
                                             "_vs_", 
                                             gsub("[^[:alnum:]]", "_", feat2), 
                                             "_data_cv.csv"))
    write.csv(interaction_data, file = interaction_data_file, row.names = FALSE)
    cat(paste0("交互数据已保存到: ", interaction_data_file, "\n"))
    
    # 保存折叠级别的交互强度统计
    if (nrow(fold_interaction_strength) > 0) {
      fold_stats_file <- file.path(interaction_dir,
                                   paste0("interaction_", pair_idx, "_",
                                          gsub("[^[:alnum:]]", "_", feat1),
                                          "_vs_",
                                          gsub("[^[:alnum:]]", "_", feat2),
                                          "_fold_stats.csv"))
      write.csv(fold_interaction_strength, file = fold_stats_file, row.names = FALSE)
      
      # 计算跨折叠的平均交互强度
      mean_strength <- mean(fold_interaction_strength$Interaction_Strength, na.rm = TRUE)
      sd_strength <- sd(fold_interaction_strength$Interaction_Strength, na.rm = TRUE)
      cat(sprintf("交互强度: %.4f ± %.4f (跨%d个折叠)\n", 
                  mean_strength, sd_strength, nrow(fold_interaction_strength)))
    }
    
    # 创建交互散点图
    p_interact <- ggplot(interaction_data, 
                         aes(x = Feature1_value, y = Feature2_value, color = Combined_SHAP)) +
      geom_point(alpha = 0.7, size = 3) +
      scale_color_gradient2(low = "blue", mid = "#B2FFFC", high = "red", midpoint = 0) +
      labs(title = paste0(feat1, " vs ", feat2, " (CV-aggregated)"),
           x = feat1, y = feat2, color = "Combined\nSHAP") +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, size = 20, vjust = -0.5, face = "bold"),
        axis.text = element_text(size = 18, color = "black"),
        axis.title = element_text(size = 18, color = "black", face = "bold"),
        axis.title.y = element_text(margin = margin(r = 10)),
        axis.title.x = element_text(margin = margin(t = 10)),
        axis.line.x = element_blank(),  
        axis.line.y = element_blank(),  
        axis.ticks = element_blank(),  
        panel.grid.major.x = element_blank(),  
        panel.grid.minor = element_blank(),    
        panel.grid.major.y = element_line(color = "gray70", linetype = "dashed", linewidth = 0.5),  
        legend.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.position = "right",
        panel.background = element_rect(fill = "white", color = NA),
        plot.background = element_rect(fill = "white", color = NA)
      ) +
      scale_y_continuous(
        limits = c(-0.25, 1.25), 
        breaks = c(0, 1), 
        labels = c("Negative", "Positive")
      ) +
      guides(color = guide_colourbar(barwidth = 4, barheight = 10))
    
    # 保存交互图
    interaction_plot_file <- file.path(interaction_dir, 
                                       paste0("interaction_", pair_idx, "_",
                                              gsub("[^[:alnum:]]", "_", feat1), 
                                              "_vs_", 
                                              gsub("[^[:alnum:]]", "_", feat2), 
                                              "_cv.tiff"))
    ggsave(interaction_plot_file, p_interact, width = 8, height = 7, dpi = 600, 
           device = "tiff", compression = "lzw")
    cat(paste0("Interaction plot saved to: ", interaction_plot_file, "\n"))
    
    # ===== 生成交互强度热图 =====
    if (nrow(fold_interaction_strength) > 1) {
      p_strength <- ggplot(fold_interaction_strength, 
                           aes(x = as.factor(Fold), y = Interaction_Strength)) +
        geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
        geom_hline(yintercept = mean_strength, linetype = "dashed", color = "red", linewidth = 0.8) +
        labs(title = paste0("Interaction Strength Across Folds: ", feat1, " vs ", feat2),
             subtitle = sprintf("Mean Strength: %.4f ± %.4f", mean_strength, sd_strength),
             x = "Fold", y = "Interaction Strength (SD of Combined SHAP)") +
        theme_minimal() +
        theme(
          plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5, size = 11),
          axis.text = element_text(size = 10, color = "black"),
          axis.title = element_text(size = 12, color = "black"),
          panel.grid = element_blank(),
          axis.line = element_line(color = "black", linewidth = 0.5),
          axis.ticks = element_line(color = "black", linewidth = 0.5),
          panel.background = element_rect(fill = "white", color = NA),
          plot.background = element_rect(fill = "white", color = NA)
        )
      
      strength_plot_file <- file.path(interaction_dir,
                                      paste0("interaction_", pair_idx, "_strength_cv.tiff"))
      ggsave(strength_plot_file, p_strength, width = 8, height = 6, dpi = 600,
             device = "tiff", compression = "lzw")
    }
  }
  
  # 交互分析总结报告
  sink(file.path(interaction_dir, "interaction_analysis_summary.txt"))
  cat("# Feature Interaction Analysis Summary (Cross-Validation Aggregated) #\n\n")
  
  cat("## Analyzed Feature Pairs ##\n")
  for (pair_idx in 1:length(feature_pairs)) {
    selected <- feature_pairs[[pair_idx]]
    if (length(selected) == 2 && all(selected <= length(top_features_dep))) {
      feat1 <- top_features_dep[selected[1]]
      feat2 <- top_features_dep[selected[2]]
      cat(sprintf("%d. %s (index:%d) vs %s (index:%d)\n", 
                  pair_idx, feat1, selected[1], feat2, selected[2]))
    }
  }
  
 
  sink()
  
  cat("\nFeature interaction analysis completed, results saved in:", interaction_dir, "\n")
}

# 主分析流程
cat("开始交叉验证SHAP分析...\n")

# 设置交叉验证折叠数
n_folds <- 5

for (i in 1:length(model_folders)) {
  folder <- model_folders[i]
  method_name <- gsub("XGBoost_", "", basename(folder))
  
  cat(paste0("\n分析 ", method_name, " 模型 (", i, "/", length(model_folders), ")\n"))
  result <- analyze_shap_cv(folder, X, y, method_name, n_folds = n_folds)
  
  cat(paste0("已完成 ", i, "/", length(model_folders), " 个模型的分析\n"))
}

cat("\n交叉验证SHAP分析完成!所有结果已保存到", shap_dir, "文件夹\n")