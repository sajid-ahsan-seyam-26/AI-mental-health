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
library(tidyr)
library(dbscan)
library(ggplot2)

set.seed(42)

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

dtm <- removeSparseTerms(dtm, 0.99)

matrix <- as.matrix(dtm)

tfidf <- weightTfIdf(dtm)
tfidf_matrix <- as.matrix(tfidf)

tfidf_matrix

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

umap_df


umap_numeric <- umap_df[, c("UMAP1", "UMAP2")]


set.seed(42)

fviz_nbclust(
  umap_df,
  kmeans,
  method = "wss",
  k.max = 10,
  nstart = 25
)

k <- 3
kmeans_result <- kmeans(umap_df, centers = k)

kmeans_clusters <- kmeans_result$cluster

kmeans_clusters

distk_matrix <- dist(umap_numeric)

sil_kmeans <- silhouette(kmeans_clusters, distk_matrix)
sil_score_kmeans <- mean(sil_kmeans[, 3])
sil_score_kmeans


db_score_kmeans <- index.DB(umap_df, kmeans_clusters)$DB
db_score_kmeans

umap_df$KMeans <- factor(kmeans_clusters)


ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = KMeans)) +
  geom_point(size = 2) +
  labs(title = "Visualization (KMeans)", color = "Cluster") +
  theme_minimal()

tfidf_df <- as.data.frame(tfidf_matrix)

tfidf_df$cluster <- kmeans_clusters

top_keywords <- tfidf_df %>%
  group_by(cluster) %>%
  summarise(across(where(is.numeric), mean)) %>%  # safe version
  pivot_longer(
    cols = -cluster,
    names_to = "term",
    values_to = "tfidf"
  ) %>%
  group_by(cluster) %>%
  slice_max(order_by = tfidf, n = 10) %>%
  ungroup()

cluster_summary <- top_keywords %>%
  group_by(cluster) %>%
  summarise(keywords = paste(term, collapse = ", "))

cluster_summary

dist_matrix <- dist(umap_df)
hc <- hclust(dist_matrix, method = "ward.D2")
hc_clusters <- cutree(hc, k = k)

hc_clusters

plot(hc,
     labels = FALSE,
     hang = -1,
     main = "Dendrogram (Hierarchical Clustering)",
     xlab = "",
     sub = "",
     cex = 0.6)

sil_hc <- silhouette(hc_clusters, dist_matrix)
silhc_score <- mean(sil_hc[, 3])

silhc_score

umap_numeric <- umap_df[, c("UMAP1", "UMAP2")]

dbhc_score <- index.DB(umap_numeric, hc_clusters)$DB
dbhc_score

umap_df$HC <- factor(hc_clusters)

ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = HC)) +
  geom_point(size = 2) +
  labs(title = "UMAP Visualization (Hierarchical Clustering)",
       color = "Cluster") +
  theme_minimal()

tfidf_df$cluster <- hc_clusters

top_keywords_hc <- tfidf_df %>%
  group_by(cluster) %>%
  summarise(across(where(is.numeric), mean)) %>%
  pivot_longer(
    cols = -cluster,
    names_to = "term",
    values_to = "tfidf"
  ) %>%
  group_by(cluster) %>%
  slice_max(order_by = tfidf, n = 10) %>%
  ungroup()

cluster_summary_hc <- top_keywords_hc %>%
  group_by(cluster) %>%
  summarise(keywords = paste(term, collapse = ", "))

cluster_summary_hc

hdb <- hdbscan(umap_numeric, minPts = 10)
hdb_clusters <- hdb$cluster   # 0 = noise

hdb_clusters

valid_idx <- hdb_clusters != 0

data_valid <- umap_numeric[valid_idx, ]
clusters_valid <- hdb_clusters[valid_idx]


distdb_matrix <- dist(data_valid)
sil_scan <- silhouette(clusters_valid, distdb_matrix)
silscan_score <- mean(sil_scan[, 3])
silscan_score

scandb_score <- index.DB(data_valid, clusters_valid)$DB
scandb_score
library(scales)
umap_plot <- as.data.frame(umap_numeric)
umap_plot$cluster <- factor(hdb$cluster)

levs <- levels(umap_plot$cluster)
pal  <- setNames(hue_pal()(length(levs)), levs)

if ("0" %in% names(pal)) pal["0"] <- "black"

ggplot(umap_plot, aes(x = UMAP1, y = UMAP2, color = cluster)) +
  geom_point(size = 2) +
  scale_color_manual(values = pal) +
  labs(title = "UMAP Visualization (HDBSCAN Clustering)",
       color = "Cluster") +
  theme_minimal()


tfidf_df <- as.data.frame(tfidf_matrix)

tfidf_valid <- tfidf_df[valid_idx, ]

tfidf_valid$cluster <- clusters_valid

top_keywords_hdb <- tfidf_valid %>%
  group_by(cluster) %>%
  summarise(across(where(is.numeric), mean)) %>%
  pivot_longer(
    cols = -cluster,
    names_to = "term",
    values_to = "tfidf"
  ) %>%
  group_by(cluster) %>%
  slice_max(order_by = tfidf, n = 10) %>%
  ungroup()

cluster_summary_hdb <- top_keywords_hdb %>%
  group_by(cluster) %>%
  summarise(keywords = paste(term, collapse = ", "))

cluster_summary_hdb


comparison_df <- data.frame(
  Algorithm = c("KMeans", "Hierarchical", "HDBSCAN"),
  Silhouette = c(sil_score_kmeans, silhc_score, silscan_score),
  Davies_Bouldin = c(db_score_kmeans, dbhc_score, scandb_score)
)

comparison_df



