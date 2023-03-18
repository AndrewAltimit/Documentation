Terraform
========================================

Overview
--------

Terraform is an open-source infrastructure-as-code (IaC) tool developed by HashiCorp. It enables you to define and manage your infrastructure using a declarative language (HCL). Terraform's goal is to manage resources across different cloud providers and on-premises environments in a consistent and efficient manner.

In this documentation, we will go through the fundamentals of Terraform while working with Amazon Web Services (AWS) as our cloud provider. By the end of this guide, you will have a better understanding of the following concepts:

* Terraform syntax and structure
* Resource creation and management
* Variable usage
* Outputs and remote state management
* Modules


1\. Getting Started
------------------------------------------

### 1.1. Configuring AWS Credentials

To use Terraform with AWS, you will need to configure your AWS credentials. There are several methods to provide AWS credentials to Terraform, but the recommended approach is to use the AWS CLI configuration file (`~/.aws/credentials`). Install the [AWS CLI](https://aws.amazon.com/cli/) and run `aws configure` to set up your credentials.

### 1.2. Initializing a Terraform Project

Create a new directory for your Terraform project and navigate to it:

```bash
mkdir terraform-aws-example
cd terraform-aws-example
``` 

Create a new file named `main.tf`:

```bash
touch main.tf
```

Open `main.tf` and add the following code to define the AWS provider:

```terraform
provider "aws" {
  region = "us-west-2"
}
```

This code tells Terraform to use the AWS provider with the `us-west-2` region as the default.

Initialize your Terraform project by running:

```bash
terraform init
```

This command downloads the required provider plugins and sets up the backend for storing the Terraform state.

2\. Creating and Managing Resources
-----------------------------------

### 2.1. Creating an Amazon S3 Bucket

Let's create an Amazon S3 bucket as an example resource. Add the following code to your `main.tf` file:

```terraform
resource "aws_s3_bucket" "example_bucket" {
  bucket = "my-example-bucket-terraform"
  acl    = "private"
}
```

This code defines a new AWS S3 bucket with the specified name and access control list (ACL) set to private.

To create the S3 bucket, run:

```bash
terraform apply
``` 

Terraform will show you a plan of the changes to be made and prompt you to confirm. Type `yes` and press Enter to proceed. Once the S3 bucket is created, you should see it in the [AWS S3 Management Console](https://s3.console.aws.amazon.com/).

### 2.2. Updating and Destroying Resources

To update a resource, modify its configuration in your `main.tf` file and run `terraform apply` again. For example, change the `acl` of the S3 bucket to "public-read":

```terraform
resource "aws_s3_bucket" "example_bucket" {
  bucket = "my-example-bucket-terraform"
  acl    = "public-read"
}
``` 

Run `terraform apply` and confirm the changes. The S3 bucket's ACL will be updated to "public-read".

To destroy a resource, run:

```bash
terraform destroy
``` 

Terraform will show you a plan of the resources to be destroyed and prompt you to confirm. Type `yes` and press Enter to proceed. The S3 bucket will be deleted.

3\. Using Variables
--------------------------------

Variables in Terraform allow you to define reusable and customizable values for your configurations. This section demonstrates how to use variables in your Terraform project.

### 3.1. Defining Variables

Create a new file named `variables.tf`:

```bash
touch variables.tf
``` 

Open `variables.tf` and add the following variable definitions:

```terraform
variable "region" {
  description = "AWS region"
  default     = "us-west-2"
}

variable "bucket_name" {
  description = "The name of the S3 bucket"
  type        = string
}

variable "bucket_acl" {
  description = "The access control list for the S3 bucket"
  type        = string
  default     = "private"
}
```

These variable definitions include a description, type, and optional default value.

### 3.2. Using Variables in Configurations

Update your `main.tf` file to use the defined variables:

```terraform
provider "aws" {
  region = var.region
}

resource "aws_s3_bucket" "example_bucket" {
  bucket = var.bucket_name
  acl    = var.bucket_acl
}
```

### 3.3. Providing Variable Values

Create a new file named `terraform.tfvars`:

```bash
touch terraform.tfvars
``` 

Open `terraform.tfvars` and add the following variable values:

```terraform
bucket_name = "my-example-bucket-terraform"
```

Now, when you run `terraform apply`, Terraform will use the provided variable values from `terraform.tfvars`.

4\. Outputs and Remote State Management
---------------------------------------

Outputs in Terraform are used to display specific values from your configuration after it is applied.

### 4.1. Defining Outputs

Create a new file named `outputs.tf`:

```bash
touch outputs.tf
``` 

Open `outputs.tf` and add the following output definition:

```terraform
output "bucket_arn" {
  description = "The Amazon Resource Name (ARN) of the created S3 bucket"
  value       = aws_s3_bucket.example_bucket.arn
}
```

After applying your configuration with `terraform apply`, the output value will be displayed.

### 4.2. Remote State Management

By default, Terraform stores the state of your infrastructure in a local file named `terraform.tfstate`. To store the state remotely and enable collaboration, you can use remote state backends like Amazon S3.

To configure a remote backend, update your `main.tf` file with the following code:

```terraform
terraform {
  backend "s3" {
    bucket = "my-terraform-state-bucket"
    key    = "terraform-aws-example/terraform.tfstate"
    region = "us-west-2"
  }
}
```

Replace `"my-terraform-state-bucket"` with the name of an existing S3 bucket in your AWS account. The `key` specifies the path in the bucket where the state file will be stored.

After updating the `main.tf`, run `terraform init` again:

```bash
terraform init
```

Terraform will prompt you to confirm the migration of the local state to the remote backend. Type `yes` and press Enter to proceed. From now on, Terraform will store the state in the specified S3 bucket.

5\. Modules
------------------------

Modules in Terraform are self-contained, reusable packages of Terraform configurations. They allow you to organize your infrastructure into smaller, maintainable units and promote the reuse of common configurations.

### 5.1. Creating a Module

Create a new directory named `modules` in your Terraform project and create a subdirectory named `s3_bucket`:

```bash
mkdir -p modules/s3_bucket
```

Inside the `s3_bucket` directory, create two files: `main.tf` and `variables.tf`.

Open `modules/s3_bucket/main.tf` and add the following code:

```terraform
resource "aws_s3_bucket" "bucket" {
  bucket = var.bucket_name
  acl    = var.bucket_acl
}
```

Open `modules/s3_bucket/variables.tf` and add the following variable definitions:

```terraform
variable "bucket_name" {
  description = "The name of the S3 bucket"
  type        = string
}

variable "bucket_acl" {
  description = "The access control list for the S3 bucket"
  type        = string
  default     = "private"
}
```

### 5.2. Using a Module in Your Configuration

Update your `main.tf` file in the root directory of your Terraform project to use the `s3_bucket` module:

```terraform
provider "aws" {
  region = var.region
}

module "example_bucket" {
  source     = "./modules/s3_bucket"
  bucket_name = var.bucket_name
  bucket_acl  = var.bucket_acl
}
```

Now, when you run `terraform apply`, Terraform will use the `s3_bucket` module to create the S3 bucket.

### 5.3. Module Outputs

To use the output values from a module, you need to define them in the module configuration. Add the following output definition to `modules/s3_bucket/outputs.tf`:

```terraform
output "bucket_arn" {
  description = "The Amazon Resource Name (ARN) of the created S3 bucket"
  value       = aws_s3_bucket.bucket.arn
}
```

To access this output value in your root configuration, update `outputs.tf` in the root directory:

```terraform
output "bucket_arn" {
  description = "The Amazon Resource Name (ARN) of the created S3 bucket"
  value       = module.example_bucket.bucket_arn
}
```

After applying your configuration with `terraform apply`, the output value will be displayed.
