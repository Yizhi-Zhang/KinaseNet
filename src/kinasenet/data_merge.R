merge_by_rownames <- function(x, y) {
  merged <- merge(x, y, by = "row.names", all = TRUE)
  rownames(merged) <- merged$Row.names
  merged$Row.names <- NULL
  return(merged)
}

process_data <- function(cancer_type, data_path, output_path, file_list, site_filter_threshold = 0.3) {
  library(dplyr)
  library(tibble)
  library(arrow)
  library(stringr)
  library(DreamAI)
  library(sva)
  library(ggplot2)
  
  if (!dir.exists(output_path)) {
    dir.create(output_path)
  }
  
  # 1. Load and merge files
  batch_labels <- c()
  
  if (length(file_list) > 1){
    data_list <- lapply(seq_along(file_list), function(i) {
      current_data <- as.data.frame(read_feather(file.path(data_path, file_list[i])))
      rownames(current_data) <- current_data$index
      current_data$index <- NULL
      
      batch_label <- str_extract(basename(file_list[i]), "PDC\\d+")
      batch_labels <<- c(batch_labels, rep(batch_label, ncol(current_data)))
      
      return(current_data)
    })
    
    merged_data <- Reduce(merge_by_rownames, data_list)
  }
  
  else{
    current_data <- as.data.frame(read_feather(file.path(data_path, file_list[1])))
    rownames(current_data) <- current_data$index
    current_data$index <- NULL
    
    merged_data <- current_data
  }
  
  # 2. Filter sites
  non_na_count <- rowSums(!is.na(merged_data))
  merged_data <- merged_data[non_na_count >= ncol(merged_data) * site_filter_threshold, ]
  
  # 3. Missing value imputation
  impute <- DreamAI(merged_data, k = 10, maxiter_MF = 10, ntree = 100,
                    maxnodes = NULL, maxiter_ADMIN = 30, tol = 10^(-2),
                    gamma_ADMIN = NA, gamma = 50, CV = FALSE,
                    fillmethod = "row_mean", maxiter_RegImpute = 10,
                    conv_nrmse = 1e-06, iter_SpectroFM = 40, 
                    method = c("KNN"),
                    out = c("Ensemble"))
  imputed_data <- impute$Ensemble
  
  if (!is.null(batch_labels)){
    # 4-1. Visualization batch effect
    pca_result <- prcomp(t(imputed_data), scale. = TRUE, na.action = na.omit)
    pca_data <- data.frame(pca_result$x)
    pca_data$batch <- batch_labels
    
    p <- ggplot(pca_data, aes(x = PC1, y = PC2, color = batch)) +
      geom_point(size = 3) +
      labs(title = "PCA of Merged Data", x = "Principal Component 1", y = "Principal Component 2") +
      theme_gray()
    ggsave(file.path(output_path, "batch_effect.png"), plot = p, width = 8, height = 6, dpi = 300)
    
    # 4-2. Remove batch effect and visualization
    combat_data <- ComBat(imputed_data, batch_labels, mod = NULL, par.prior = TRUE, 
                          prior.plots = FALSE, mean.only = FALSE, ref.batch = NULL)
    
    pca_result <- prcomp(t(combat_data), scale. = TRUE, na.action = na.omit)
    pca_data <- data.frame(pca_result$x)
    pca_data$batch <- batch_labels
    
    p <- ggplot(pca_data, aes(x = PC1, y = PC2, color = batch)) +
      geom_point(size = 3) +
      labs(title = "PCA of ComBat Data", x = "Principal Component 1", y = "Principal Component 2") +
      theme_gray()
    ggsave(file.path(output_path, "combat.png"), plot = p, width = 8, height = 6, dpi = 300)
  }
  
  else{
    combat_data <- imputed_data
  }

  combat_data <- rownames_to_column(as.data.frame(combat_data), var = "index")
  write_feather(combat_data, file.path(output_path, paste0(cancer_type, '.feather')))
}
