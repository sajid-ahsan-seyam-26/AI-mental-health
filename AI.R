install.packages(c("tm", "tidytext", "SnowballC", "cluster", 
                   "factoextra", "text2vec", "umap", "dplyr"))
install.packages("dbscan")
install.packages("clusterSim")
library(clusterSim)

library(tm)
library(tidytext)
library(SnowballC)
library(cluster)
library(factoextra)
library(text2vec)
library(umap)
library(dplyr)
library(dbscan)

data <- read.csv("human_ai_dataset.csv")
nrow(data)
texts <- data$abstract
texts[1]

corpus <- Corpus(VectorSource(texts))

corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)
corpus <- tm_map(corpus, stripWhitespace)

inspect(corpus[1])

dtm <- DocumentTermMatrix(corpus)

# Remove sparse terms
dtm <- removeSparseTerms(dtm, 0.99)

# Convert to matrix
matrix <- as.matrix(dtm)

# TF-IDF
tfidf <- weightTfIdf(dtm)
tfidf_matrix <- as.matrix(tfidf)

tfidf_matrix

set.seed(42)

cat("--- STEP 5: Dimensionality Reduction (UMAP) ---\n")
n_nbrs <- min(15, nrow(tfidf_matrix) - 1)

umap_out <- umap(
  tfidf_matrix,
  n_components  = 2,
  n_neighbors   = n_nbrs,
  min_dist      = 0.1,
  metric        = "cosine",
  n_epochs      = 200,
  learning_rate = 1.0,
  verbose       = FALSE,
  n_threads     = 1
)

umap_df <- data.frame(
  UMAP1 = umap_out$layout[, 1],
  UMAP2 = umap_out$layout[, 2]
)


cat("UMAP Dimension 1 range:", round(min(umap_df$UMAP1), 3), "to", round(max(umap_df$UMAP1), 3), "\n")
cat("UMAP Dimension 2 range:", round(min(umap_df$UMAP2), 3), "to", round(max(umap_df$UMAP2), 3), "\n\n")


fviz_nbclust(umap_df, kmeans, method = "wss")

k <- 5
kmeans_result <- kmeans(umap_df, centers = k)

kmeans_clusters <- kmeans_result$cluster

kmeans_clusters

dist_matrix <- dist(umap_df)
hc <- hclust(dist_matrix, method = "ward.D2")
hc_clusters <- cutree(hc, k = k)

hc_clusters


hdb <- hdbscan(umap_df, minPts = 10)
hdb_clusters <- hdb$cluster   # 0 = noise

hdb_clusters


# ================================
# 8. Evaluation (Silhouette + Davies-Bouldin)
# ================================

# Install if needed


# ----------- KMeans -----------
sil_kmeans <- silhouette(kmeans_clusters, dist(umap_df))
db_kmeans <- index.DB(umap_df, kmeans_clusters)$DB

cat("\nKMeans Results:\n")
cat("Silhouette Score:", mean(sil_kmeans[,3]), "\n")
cat("Davies-Bouldin Index:", db_kmeans, "\n")

# ----------- Hierarchical -----------
sil_hc <- silhouette(hc_clusters, dist(umap_df))
db_hc <- index.DB(umap_df, hc_clusters)$DB

cat("\nHierarchical Results:\n")
cat("Silhouette Score:", mean(sil_hc[,3]), "\n")
cat("Davies-Bouldin Index:", db_hc, "\n")

# ----------- HDBSCAN -----------
valid_idx <- hdb_clusters != 0  # remove noise

if (length(unique(hdb_clusters[valid_idx])) > 1) {
  sil_hdb <- silhouette(hdb_clusters[valid_idx], 
                        dist(umap_df[valid_idx, ]))
  
  db_hdb <- index.DB(umap_df[valid_idx, ], 
                     hdb_clusters[valid_idx])$DB
  
  cat("\nHDBSCAN Results:\n")
  cat("Silhouette Score:", mean(sil_hdb[,3]), "\n")
  cat("Davies-Bouldin Index:", db_hdb, "\n")
} else {
  cat("\nHDBSCAN produced insufficient clusters\n")
}

install.packages("ggplot2")
library(ggplot2)

umap_df$KMeans <- as.factor(kmeans_clusters)
umap_df$Hierarchical <- as.factor(hc_clusters)
umap_df$HDBSCAN <- as.factor(hdb_clusters)


k_vis <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = KMeans)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "UMAP Visualization (K-Means Clustering)",
       x = "UMAP Dimension 1",
       y = "UMAP Dimension 2",
       color = "Cluster") +
  theme_minimal()

k_vis

hc_vis <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Hierarchical)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "UMAP Visualization (Hierarchical Clustering)",
       x = "UMAP Dimension 1",
       y = "UMAP Dimension 2",
       color = "Cluster") +
  theme_minimal()
hc_vis

hdb_vis <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = HDBSCAN)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "UMAP Visualization (HDBSCAN Clustering)",
       x = "UMAP Dimension 1",
       y = "UMAP Dimension 2",
       color = "Cluster (0 = Noise)") +
  theme_minimal()

hdb_vis

ggplot(umap_df, aes(x = UMAP1, y = UMAP2)) +
  geom_point(aes(color = HDBSCAN == 0), size = 2, alpha = 0.8) +
  scale_color_manual(values = c("FALSE" = "blue", "TRUE" = "red")) +
  labs(title = "HDBSCAN (Noise vs Clusters)",
       x = "UMAP1",
       y = "UMAP2",
       color = "Noise") +
  theme_minimal()

# Generate colors for clusters
cluster_ids <- unique(hdb_clusters)
cluster_ids <- sort(cluster_ids)

# Assign colors (black for noise)
cluster_colors <- rainbow(length(cluster_ids))
cluster_colors[cluster_ids == 0] <- "black"

# Map colors
point_colors <- cluster_colors[match(hdb_clusters, cluster_ids)]

# Plot
plot(umap_df$UMAP1, umap_df$UMAP2,
     col = point_colors,
     pch = 19,
     main = "HDBSCAN Clusters",
     xlab = "UMAP1",
     ylab = "UMAP2")

# Legend
legend("topright",
       legend = paste("Cluster", cluster_ids),
       col = cluster_colors,
       pch = 19)

num_clusters <- length(unique(hdb_clusters[hdb_clusters != 0]))
num_clusters
length(unique(hdb_clusters))

ggplot(umap_df, aes(x = UMAP1, y = UMAP2)) +
  geom_point(aes(color = HDBSCAN), size = 2, alpha = 0.8) +
  
  # Make noise black and clusters colorful
  scale_color_manual(
    values = c(
      "0" = "black",                         # noise
      setNames(rainbow(length(unique(hdb_clusters[hdb_clusters != 0]))),
               unique(hdb_clusters[hdb_clusters != 0]))
    )
  ) +
  
  labs(title = "HDBSCAN Clusters with Noise",
       x = "UMAP1",
       y = "UMAP2",
       color = "Cluster (0 = Noise)") +
  
  theme_minimal()

library(tidyr)

# Use TF-IDF (better than raw DTM)
dtm_matrix <- tfidf_matrix

# Use HDBSCAN clusters
cluster_labels <- hdb_clusters

# Remove noise (cluster 0)
valid_idx <- cluster_labels != 0
dtm_matrix <- dtm_matrix[valid_idx, ]
cluster_labels <- cluster_labels[valid_idx]

# Convert to dataframe
df <- as.data.frame(dtm_matrix)
df$cluster <- cluster_labels

# Get top keywords per cluster
top_terms <- df %>%
  group_by(cluster) %>%
  summarise(across(everything(), mean)) %>%
  pivot_longer(-cluster, names_to = "term", values_to = "value") %>%
  arrange(cluster, desc(value)) %>%
  group_by(cluster) %>%
  slice_head(n = 10)

# Show results
top_terms

top_terms %>%
  group_by(cluster) %>%
  summarise(keywords = paste(term, collapse = ", "))
