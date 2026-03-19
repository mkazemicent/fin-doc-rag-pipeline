terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23.0"
    }
  }
}

provider "kubernetes" {
  config_path    = "~/.kube/config"
  config_context = "minikube"
}

resource "kubernetes_namespace" "deal_analyzer" {
  metadata { name = "deal-analyzer-prod" }
}

# --- CONFIGURATION ---
resource "kubernetes_config_map" "app_config" {
  metadata {
    name      = "analyzer-config"
    namespace = kubernetes_namespace.deal_analyzer.metadata[0].name
  }
  data = {
    TMPDIR          = "/app/data/tmp"
    DATA_DIR        = "/app/data"
    OLLAMA_BASE_URL = "http://ollama-service:11434"
    CHROMA_HOST     = "chroma-service"
    CHROMA_PORT     = "8000"
    LLM_MODEL       = "qwen2.5:14b"
    EMBEDDING_MODEL = "nomic-embed-text"
  }
}

# --- ENTERPRISE STORAGE ---
resource "kubernetes_persistent_volume_claim" "chroma_data" {
  metadata {
    name      = "chroma-data-pvc"
    namespace = kubernetes_namespace.deal_analyzer.metadata[0].name
  }
  spec {
    access_modes = ["ReadWriteOnce"]
    resources { requests = { storage = "10Gi" } }
  }
}

resource "kubernetes_persistent_volume_claim" "ollama_data" {
  metadata {
    name      = "ollama-data-pvc"
    namespace = kubernetes_namespace.deal_analyzer.metadata[0].name
  }
  spec {
    access_modes = ["ReadWriteOnce"]
    resources { requests = { storage = "20Gi" } }
  }
}

# --- OLLAMA (GPU WORKLOAD) ---
resource "kubernetes_deployment" "ollama" {
  metadata {
    name      = "ollama"
    namespace = kubernetes_namespace.deal_analyzer.metadata[0].name
  }
  spec {
    replicas = 1
    strategy {
      type = "Recreate"
    }
    selector { match_labels = { app = "ollama" } }
    template {
      metadata { labels = { app = "ollama" } }
      spec {
        container {
          image = "ollama/ollama:0.6.2"
          name  = "ollama"
          port { container_port = 11434 }
          resources {
            limits = {
              "nvidia.com/gpu" = "1"
              memory           = "16Gi"
            }
          }
          volume_mount {
            name       = "ollama-storage"
            mount_path = "/root/.ollama"
          }
        }
        volume {
          name = "ollama-storage"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim.ollama_data.metadata[0].name
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "ollama" {
  metadata {
    name      = "ollama-service"
    namespace = kubernetes_namespace.deal_analyzer.metadata[0].name
  }
  spec {
    selector = { app = "ollama" }
    port {
      port        = 11434
      target_port = 11434
    }
  }
}

# --- CHROMADB ---
resource "kubernetes_deployment" "chroma" {
  metadata {
    name      = "chroma"
    namespace = kubernetes_namespace.deal_analyzer.metadata[0].name
  }
  spec {
    replicas = 1
    selector { match_labels = { app = "chroma" } }
    template {
      metadata { labels = { app = "chroma" } }
      spec {
        container {
          image = "chromadb/chroma:1.5.4"
          name  = "chroma"
          port { container_port = 8000 }
          env {
            name  = "ANONYMIZED_TELEMETRY"
            value = "False"
          }
          resources {
            limits = { memory = "4Gi" }
          }

          volume_mount {
            name       = "chroma-storage"
            mount_path = "/chroma/chroma"
          }
        }

        volume {
          name = "chroma-storage"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim.chroma_data.metadata[0].name
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "chroma" {
  metadata {
    name      = "chroma-service"
    namespace = kubernetes_namespace.deal_analyzer.metadata[0].name
  }
  spec {
    selector = { app = "chroma" }
    port {
      port        = 8000
      target_port = 8000
    }
  }
}

# --- STREAMLIT APP ---
resource "kubernetes_persistent_volume_claim" "app_data" {
  metadata {
    name      = "app-data-pvc"
    namespace = kubernetes_namespace.deal_analyzer.metadata[0].name
  }
  spec {
    access_modes = ["ReadWriteOnce"]
    resources {
      requests = {
        storage = "5Gi"
      }
    }
  }
}

resource "kubernetes_deployment" "app" {
  metadata {
    name      = "deal-analyzer-app"
    namespace = kubernetes_namespace.deal_analyzer.metadata[0].name
  }
  spec {
    replicas = 1
    selector { match_labels = { app = "deal-analyzer-app" } }
    template {
      metadata { labels = { app = "deal-analyzer-app" } }
      spec {
        
        security_context {
          fs_group = 1000
        }

        init_container {
          name  = "take-volume-ownership"
          image = "busybox"
          # Create the sub-directory for /tmp and take ownership of the whole disk
          command = ["sh", "-c", "mkdir -p /app/data/tmp && chown -R 1000:1000 /app/data"]
          
          security_context {
            run_as_user = 0
          }

          volume_mount {
            name       = "app-storage"
            mount_path = "/app/data"
          }
        }

        container {
          image             = "deal-analyzer-app:v1"
          name              = "app"
          image_pull_policy = "Never" 
          port { container_port = 8501 }

          resources {
            limits = {
              cpu    = "1"
              memory = "2Gi"
            }
            requests = {
              cpu    = "100m"
              memory = "512Mi"
            }
          }
          
          env_from {
            config_map_ref {
              name = kubernetes_config_map.app_config.metadata[0].name
            }
          }

          # 1. Mount the main volume for your app data
          volume_mount {
            name       = "app-storage"
            mount_path = "/app/data"
          }

          # 2. THE FIX: Mount a sub-folder of the SAME volume directly over /tmp
          volume_mount {
            name       = "app-storage"
            mount_path = "/tmp"
            sub_path   = "tmp"
          }
        }

        volume {
          name = "app-storage"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim.app_data.metadata[0].name
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "app" {
  metadata {
    name      = "app-service"
    namespace = kubernetes_namespace.deal_analyzer.metadata[0].name
  }
  spec {
    type     = "NodePort"
    selector = { app = "deal-analyzer-app" }
    port {
      port        = 8501
      target_port = 8501
      node_port   = 30501
    }
  }
}

# --- AUTO-SCALING (HPA) ---
resource "kubernetes_horizontal_pod_autoscaler_v2" "app_hpa" {
  metadata {
    name      = "deal-analyzer-hpa"
    namespace = kubernetes_namespace.deal_analyzer.metadata[0].name
  }
  spec {
    max_replicas = 12
    min_replicas = 1
    
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = kubernetes_deployment.app.metadata[0].name
    }
    
    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = 75
        }
      }
    }
  }
}