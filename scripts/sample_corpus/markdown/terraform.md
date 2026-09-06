# Terraform Reference Guide

## Overview
Terraform is an Infrastructure-as-Code (IaC) tool that lets you define cloud and on-prem infrastructure in declarative configuration files. It tracks real-world resource state and computes the minimal set of changes needed to move from the current state to the desired state.

## Core Concepts

### Providers
Plugins that let Terraform interact with a specific platform's API (AWS, GCP, Azure, Kubernetes, etc.).

```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "ap-south-1"
}
```

### Resources
The core building block — declares a piece of infrastructure to create/manage.

```hcl
resource "aws_s3_bucket" "rag_corpus" {
  bucket = "my-rag-corpus-bucket"
}

resource "aws_instance" "training_node" {
  ami           = "ami-0123456789abcdef0"
  instance_type = "g4dn.xlarge"

  tags = {
    Name = "gpu-training-node"
  }
}
```

### Variables & Outputs

```hcl
variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"
}

output "bucket_arn" {
  value = aws_s3_bucket.rag_corpus.arn
}
```

### State
Terraform stores the current state of managed infrastructure in a state file (`terraform.tfstate`). For team environments, state should be stored remotely (e.g. S3 + DynamoDB for locking) rather than locally.

```hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state-bucket"
    key            = "rag-infra/terraform.tfstate"
    region         = "ap-south-1"
    dynamodb_table = "terraform-locks"
  }
}
```

## Core Workflow

```bash
terraform init      # Initialize working directory, download providers
terraform plan       # Preview changes before applying
terraform apply      # Apply changes to reach desired state
terraform destroy    # Tear down managed infrastructure
terraform fmt        # Auto-format configuration files
terraform validate   # Check configuration syntax/consistency
```

## Modules
Reusable, parameterized groups of resources — analogous to functions in a programming language.

```hcl
module "vpc" {
  source     = "./modules/vpc"
  cidr_block = "10.0.0.0/16"
}
```

Modules can be local (relative path), from the Terraform Registry, or from a Git repository.

## Example: EKS Cluster for ML Workloads

```hcl
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  cluster_name    = "ml-platform-cluster"
  cluster_version = "1.29"
  subnet_ids      = module.vpc.private_subnets
  vpc_id          = module.vpc.vpc_id

  eks_managed_node_groups = {
    gpu_nodes = {
      instance_types = ["g4dn.xlarge"]
      min_size       = 1
      max_size       = 4
      desired_size   = 1
    }
  }
}
```

## State Manipulation Commands

```bash
terraform state list                  # List resources tracked in state
terraform state show <resource>       # Show details of a tracked resource
terraform import <resource> <id>      # Bring an existing resource under Terraform management
terraform state rm <resource>         # Stop tracking a resource without destroying it
```

## Workspaces
Allow managing multiple environments (dev/staging/prod) from the same configuration with separate state files.

```bash
terraform workspace new staging
terraform workspace select staging
terraform workspace list
```

## Best Practices
- Always run `terraform plan` and review the diff before `apply`, especially in shared/production environments.
- Store state remotely with locking enabled to prevent concurrent modification conflicts.
- Never commit `terraform.tfstate` or `.tfvars` files containing secrets to version control.
- Pin provider and module versions to avoid unexpected breaking changes.
- Use `variables.tf`, `outputs.tf`, and `main.tf` file conventions for readability in larger projects.
- Tag all resources consistently (environment, project, owner) for cost tracking and cleanup.

## Common Troubleshooting
- **"Error acquiring the state lock"**: another `apply`/`plan` is in progress, or a previous run crashed without releasing the lock — check the DynamoDB lock table (if using S3 backend) and manually release if safe.
- **Plan shows unexpected destroy/recreate**: often caused by a change to an immutable resource attribute (e.g. AMI ID forcing instance replacement) — review resource documentation for which attributes force replacement.
- **Provider version conflicts**: pin exact versions in `required_providers` and run `terraform init -upgrade` deliberately rather than picking up unplanned updates.
- **Drift between actual infrastructure and state**: run `terraform plan` regularly to detect drift caused by manual console changes; reconcile with `terraform import` or by reapplying configuration.

## Relevance to MLOps Infrastructure
Terraform is commonly used to provision the infrastructure that hosts ML systems — for example, an EKS/GKE cluster for model serving, S3/GCS buckets for datasets and MLflow artifacts, RDS/Cloud SQL for an MLflow Postgres backend, and IAM roles/policies scoping access for CI/CD pipelines — keeping infrastructure changes versioned and reviewable alongside application code.
