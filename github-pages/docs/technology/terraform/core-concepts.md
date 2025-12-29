---
layout: docs
title: "Terraform: Core Concepts"
permalink: /docs/technology/terraform/core-concepts.html
toc: true
toc_sticky: true
---

## Getting Started

Before writing any infrastructure code, you need a few things in place. The good news: most of these are free and take only minutes to set up.

### Prerequisites

| Requirement | Why You Need It | How to Get It |
|-------------|-----------------|---------------|
| **Command line basics** | Terraform runs from the terminal | Practice with `cd`, `ls`, `mkdir` |
| **Cloud account** | A place to deploy infrastructure | AWS/Azure/GCP free tier |
| **Text editor** | Writing configuration files | VS Code with HCL extension |
| **Terraform CLI** | The tool itself | [terraform.io/downloads](https://www.terraform.io/downloads) |

### Verifying Your Installation

After installing Terraform, confirm it works:

```bash
terraform version
# Terraform v1.7.0 on linux_amd64
```

## Terraform Crash Course: Zero to Hero in 30 Minutes

This hands-on crash course takes you from zero knowledge to deploying real cloud infrastructure. By the end, you will understand the complete Terraform workflow.

---

### Step 1: Your First Terraform File (5 minutes)

Every Terraform project starts with a configuration file. Create a new directory and file:

```bash
mkdir terraform-tutorial && cd terraform-tutorial
touch main.tf
```

Add this configuration to `main.tf`. Notice how it reads almost like a description of what you want:

```hcl
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = "us-east-1"  # Uses AWS_ACCESS_KEY_ID env var
}

resource "aws_s3_bucket" "my_first_bucket" {
  bucket = "my-unique-bucket-12345"  # Must be globally unique
  tags   = { Name = "My First Terraform Bucket" }
}
```

**What is happening here:**
- `terraform {}` block configures Terraform itself and declares required providers
- `provider "aws"` tells Terraform how to connect to AWS
- `resource` declares infrastructure you want to create

### Step 2: The Terraform Workflow

Terraform follows a predictable three-step workflow. Understanding this workflow is essential because you will repeat it hundreds of times.

| Command | What It Does | When to Use |
|---------|--------------|-------------|
| `terraform init` | Downloads providers, sets up backend | Once per project, or after adding providers |
| `terraform plan` | Shows what will change (dry run) | Before every apply, to review changes |
| `terraform apply` | Creates/updates real infrastructure | When you are ready to make changes |
| `terraform destroy` | Removes all managed resources | Cleanup, or tearing down environments |

```bash
# Step 1: Initialize (downloads the AWS provider)
terraform init

# Step 2: Preview changes (nothing is created yet)
terraform plan

# Step 3: Apply changes (type 'yes' to confirm)
terraform apply
```

---

### Step 3: Modifying Infrastructure

The real power of Terraform shows when you change things. Add versioning to your bucket:

```hcl
resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.my_first_bucket.id
  versioning_configuration { status = "Enabled" }
}
```

Run `terraform plan` again. Terraform shows exactly what will change - in this case, one resource added. This predictability is why teams trust Terraform with production infrastructure.

---

### Step 4: Understanding State

After applying, check your directory. You will see a `terraform.tfstate` file. This is Terraform's memory of what it created.

```bash
terraform state list          # See managed resources
terraform state show aws_s3_bucket.my_first_bucket  # Details
```

**Why state matters:** Terraform compares your configuration against state to determine what changes are needed. Without state, Terraform would try to create everything fresh each time.

---

### Step 5: Clean Up

When you are done experimenting, remove everything:

```bash
terraform destroy   # Type 'yes' to confirm
```

**What you accomplished:**
- Wrote Infrastructure as Code
- Created real cloud resources
- Modified existing infrastructure safely
- Understood state management
- Cleaned up automatically

---

## Core Concepts

Now that you have hands-on experience, let's understand the principles that make Terraform work.

### Declarative vs Imperative: Why It Matters

Consider the following approaches to creating a web server:

| Imperative (Scripts) | Declarative (Terraform) |
|---------------------|-------------------------|
| "Create a VM. If it fails, retry. Then install nginx. Configure the firewall..." | "I want a VM with nginx and these firewall rules" |
| You specify every step | You specify the end state |
| Order matters | Terraform figures out the order |
| Partial failures leave mess | Terraform tracks what succeeded |

Terraform's declarative approach means you describe *what* you want, not *how* to get there. This eliminates entire categories of bugs.

### Why Infrastructure as Code Changes Everything

When infrastructure becomes code, you gain powerful capabilities:

- **Version Control**: Track every change through git history
- **Code Review**: Infrastructure changes go through pull requests
- **Rollback**: Return to any previous state instantly
- **Reusability**: Package patterns into modules
- **Testing**: Validate before deploying

---

## How Terraform Works Under the Hood

Understanding Terraform's internals helps you write better configurations and debug issues faster.

### Dependency Graphs: Automatic Ordering

Consider the following scenario: You need a database, a web server that connects to it, and a load balancer in front of the web server. These must be created in order - you cannot connect to a database that does not exist yet.

Terraform automatically figures out this ordering by building a dependency graph. When you reference one resource from another (like `database.endpoint`), Terraform knows to create the database first.

**What this means for you:**
- You do not need to specify creation order
- Terraform creates independent resources in parallel (faster deployments)
- Circular dependencies are caught before anything is created

### The Plan-Apply Cycle

When you run `terraform plan`, Terraform performs a three-way comparison:

1. **Your Configuration**: What you wrote in `.tf` files
2. **Current State**: What Terraform remembers from the last apply
3. **Real Infrastructure**: What actually exists in the cloud (via API calls)

This comparison produces one of five actions for each resource:

| Action | Symbol | Meaning |
|--------|--------|---------|
| Create | `+` | Resource does not exist, will be created |
| Update | `~` | Resource exists but configuration changed |
| Replace | `-/+` | Must destroy and recreate (e.g., changing instance type) |
| Delete | `-` | Resource in state but not in configuration |
| No-op | (none) | Everything matches, no action needed |

**Practical tip:** Always review the plan output before applying. The symbols make it easy to spot unexpected changes.

---

## Providers: Connecting to Cloud Services

Providers are plugins that let Terraform communicate with cloud platforms and services. Think of them as translators between Terraform's configuration language and each platform's API.

### How Providers Work

When you declare a provider, Terraform:
1. Downloads the provider plugin during `terraform init`
2. Uses it to translate your configuration into API calls
3. Handles authentication, retries, and rate limiting automatically

```hcl
# Tell Terraform which providers you need
terraform {
  required_providers {
    aws   = { source = "hashicorp/aws", version = "~> 5.0" }
    azure = { source = "hashicorp/azurerm", version = "~> 3.0" }
  }
}
```

### Commonly Used Providers

| Provider | Use Case | Example Resources |
|----------|----------|-------------------|
| `aws` | Amazon Web Services | EC2 instances, S3 buckets, RDS databases |
| `azurerm` | Microsoft Azure | VMs, Storage accounts, SQL databases |
| `google` | Google Cloud | Compute instances, GCS buckets, Cloud SQL |
| `kubernetes` | K8s clusters | Deployments, Services, ConfigMaps |
| `random` | Generate random values | Random IDs, passwords, pet names |

### Provider Authentication

Terraform supports multiple authentication methods. Choose based on your security requirements:

| Method | Best For | Security Level |
|--------|----------|----------------|
| Environment variables | CI/CD pipelines | Good |
| Shared credentials file | Local development | Moderate |
| IAM roles (assume role) | Cross-account access | Excellent |
| Instance profiles | EC2-based automation | Excellent |

```hcl
provider "aws" {
  region = "us-east-1"

  # Option 1: Use environment variables (recommended for CI/CD)
  # Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY

  # Option 2: Assume a role (recommended for cross-account)
  assume_role {
    role_arn = "arn:aws:iam::123456789:role/TerraformRole"
  }

  # Apply default tags to all resources
  default_tags {
    tags = { ManagedBy = "Terraform" }
  }
}
```

**Security tip:** Never hardcode credentials in Terraform files. Use environment variables or IAM roles instead.

---

## Resource Lifecycles

Every resource in Terraform goes through a lifecycle: creation, updates, and eventual deletion. Understanding this lifecycle helps you handle special cases.

### The Lifecycle Meta-Argument

Sometimes you need to customize how Terraform handles resource changes. The `lifecycle` block lets you do this:

```hcl
resource "aws_instance" "web" {
  ami           = "ami-12345678"
  instance_type = "t3.micro"

  lifecycle {
    create_before_destroy = true  # Create new before destroying old
    prevent_destroy       = true  # Block accidental deletion
    ignore_changes        = [tags]  # Ignore external tag changes
  }
}
```

### When to Use Each Lifecycle Option

| Option | Use Case | Example |
|--------|----------|---------|
| `create_before_destroy` | Zero-downtime updates | Load balancers, DNS records |
| `prevent_destroy` | Critical resources | Production databases, S3 buckets with data |
| `ignore_changes` | Externally managed attributes | Tags set by other tools, autoscaling counts |
| `replace_triggered_by` | Force replacement | When AMI changes, replace instance |



---

## Variables and Types

Variables make your Terraform code reusable. Instead of hardcoding values, you define variables that can change between environments.

### Variable Types at a Glance

| Type | Example | Use Case |
|------|---------|----------|
| `string` | `"us-east-1"` | Regions, names, IDs |
| `number` | `3` | Counts, sizes, ports |
| `bool` | `true` | Feature flags |
| `list(string)` | `["a", "b"]` | Multiple values of same type |
| `map(string)` | `{env = "prod"}` | Key-value pairs |
| `object({...})` | Complex structures | Grouped configuration |

### Defining Variables with Validation

Catch errors before they reach production by validating inputs:

```hcl
variable "environment" {
  type        = string
  description = "Deployment environment"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "instance_type" {
  type    = string
  default = "t3.micro"

  validation {
    condition     = can(regex("^t3\\.", var.instance_type))
    error_message = "Only t3 instance types are allowed."
  }
}
```

### Variable Precedence

Variables can be set in multiple places. Terraform uses them in this order (later overrides earlier):

1. Default values in variable definitions
2. Environment variables (`TF_VAR_name`)
3. `terraform.tfvars` file
4. `*.auto.tfvars` files (alphabetical order)
5. `-var-file` command line flag
6. `-var` command line flag

**Practical tip:** Use `terraform.tfvars` for environment-specific values and keep it out of version control for sensitive data.

### Using Variables in Practice

Here is a complete example showing variables in action:

```hcl
# variables.tf
variable "region" {
  description = "AWS region"
  default     = "us-west-2"
}

variable "bucket_name" {
  description = "Name for the S3 bucket"
  type        = string  # Required - no default
}

# main.tf
provider "aws" {
  region = var.region
}

resource "aws_s3_bucket" "data" {
  bucket = var.bucket_name
}

# terraform.tfvars
bucket_name = "my-company-data-bucket"
```

When you run `terraform apply`, Terraform uses the values from `terraform.tfvars` automatically.

