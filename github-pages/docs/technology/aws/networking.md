---
layout: docs
title: AWS Networking & Content Delivery
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "network-wired"
---

# AWS Networking & Content Delivery

This guide covers VPC networking, CloudFront CDN, API Gateway, and load balancing. Learn how to build secure, high-performance network architectures on AWS.

## VPC - Your Private Cloud Network

VPC (Virtual Private Cloud) lets you create isolated networks in AWS. Think of it as your own private data center in the cloud, complete with subnets, routing rules, and security controls.

### Key VPC Concepts

1. **Subnets**: Subdivide your VPC into public and private segments
2. **Internet Gateway**: Allows public internet access
3. **NAT Gateway**: Enables private instances to reach the internet
4. **Security Groups**: Virtual firewalls for instances
5. **Network ACLs**: Stateless subnet-level firewalls
6. **Route Tables**: Control traffic routing

### Multi-AZ Architecture

Spread resources across Availability Zones for resilience:

```
VPC (10.0.0.0/16)
├── Public Subnet AZ-A (10.0.1.0/24)
│   ├── NAT Gateway
│   └── Load Balancer
├── Public Subnet AZ-B (10.0.2.0/24)
│   ├── NAT Gateway  
│   └── Load Balancer
├── Private Subnet AZ-A (10.0.10.0/24)
│   └── Application Servers
├── Private Subnet AZ-B (10.0.20.0/24)
│   └── Application Servers
├── Data Subnet AZ-A (10.0.100.0/24)
│   └── RDS Primary
└── Data Subnet AZ-B (10.0.200.0/24)
    └── RDS Standby
```

### Networking: Connecting Your Application

#### Amazon VPC - Your Private Cloud Network
VPC (Virtual Private Cloud) lets you create isolated networks in AWS. Think of it as your own private data center in the cloud, complete with subnets, routing rules, and security controls.

This is where things get more complex, but understanding VPC is crucial for production applications. Let's build up gradually:

1. **Basic VPC**: A simple network with public and private subnets
2. **Internet Access**: Add an Internet Gateway for public-facing resources
3. **Security Groups**: Virtual firewalls controlling traffic to your resources
4. **Multi-AZ Design**: Spread resources across Availability Zones for resilience

**Real-world example**: An enterprise application runs web servers in public subnets (accessible from internet) and databases in private subnets (only accessible from web servers). This layered security approach protects sensitive data while serving public traffic.

### API Gateway Advanced Patterns

#### REST API with Request Validation

```hcl
# api-gateway.tf - Advanced API Gateway with request validation

# REST API
resource "aws_api_gateway_rest_api" "main" {
  name        = "${var.environment}-advanced-api"
  description = "Advanced REST API with validation and security"
  
  endpoint_configuration {
    types = ["REGIONAL"]
  }
  
  # API Policy for resource limits
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = "*"
        Action = "execute-api:Invoke"
        Resource = "*"
      },
      {
        Effect = "Deny"
        Principal = "*"
        Action = "execute-api:Invoke"
        Resource = "*"
        Condition = {
          IpAddressNotEquals = {
            "aws:SourceIp" = var.allowed_ip_ranges
          }
        }
      }
    ]
  })
}

# Request validator
resource "aws_api_gateway_request_validator" "main" {
  name                        = "${var.environment}-validator"
  rest_api_id                 = aws_api_gateway_rest_api.main.id
  validate_request_body       = true
  validate_request_parameters = true
}

# API Models for request/response validation
resource "aws_api_gateway_model" "user" {
  rest_api_id  = aws_api_gateway_rest_api.main.id
  name         = "User"
  content_type = "application/json"
  
  schema = jsonencode({
    "$schema" = "http://json-schema.org/draft-04/schema#"
    type = "object"
    required = ["email", "name"]
    properties = {
      email = {
        type = "string"
        format = "email"
        pattern = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
      }
      name = {
        type = "string"
        minLength = 1
        maxLength = 100
      }
      age = {
        type = "integer"
        minimum = 0
        maximum = 150
      }
      preferences = {
        type = "object"
        properties = {
          notifications = {
            type = "boolean"
          }
          theme = {
            type = "string"
            enum = ["light", "dark", "auto"]
          }
        }
      }
    }
  })
}

resource "aws_api_gateway_model" "error" {
  rest_api_id  = aws_api_gateway_rest_api.main.id
  name         = "Error"
  content_type = "application/json"
  
  schema = jsonencode({
    "$schema" = "http://json-schema.org/draft-04/schema#"
    type = "object"
    required = ["message"]
    properties = {
      message = {
        type = "string"
      }
      code = {
        type = "string"
      }
      requestId = {
        type = "string"
      }
    }
  })
}

# API Resources
resource "aws_api_gateway_resource" "v1" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  parent_id   = aws_api_gateway_rest_api.main.root_resource_id
  path_part   = "v1"
}

resource "aws_api_gateway_resource" "users" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  parent_id   = aws_api_gateway_resource.v1.id
  path_part   = "users"
}

resource "aws_api_gateway_resource" "user" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  parent_id   = aws_api_gateway_resource.users.id
  path_part   = "{userId}"
}

# OPTIONS method for CORS
resource "aws_api_gateway_method" "users_options" {
  rest_api_id   = aws_api_gateway_rest_api.main.id
  resource_id   = aws_api_gateway_resource.users.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "users_options" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.users.id
  http_method = aws_api_gateway_method.users_options.http_method
  type        = "MOCK"
  
  request_templates = {
    "application/json" = "{\"statusCode\": 200}"
  }
}

resource "aws_api_gateway_method_response" "users_options" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.users.id
  http_method = aws_api_gateway_method.users_options.http_method
  status_code = "200"
  
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true
    "method.response.header.Access-Control-Allow-Methods" = true
    "method.response.header.Access-Control-Allow-Origin"  = true
  }
  
  response_models = {
    "application/json" = "Empty"
  }
}

resource "aws_api_gateway_integration_response" "users_options" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.users.id
  http_method = aws_api_gateway_method.users_options.http_method
  status_code = aws_api_gateway_method_response.users_options.status_code
  
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"
    "method.response.header.Access-Control-Allow-Methods" = "'GET,OPTIONS,POST,PUT,DELETE'"
    "method.response.header.Access-Control-Allow-Origin"  = "'*'"
  }
}

# POST method with validation
resource "aws_api_gateway_method" "create_user" {
  rest_api_id          = aws_api_gateway_rest_api.main.id
  resource_id          = aws_api_gateway_resource.users.id
  http_method          = "POST"
  authorization        = "CUSTOM"
  authorizer_id        = aws_api_gateway_authorizer.jwt.id
  request_validator_id = aws_api_gateway_request_validator.main.id
  
  request_models = {
    "application/json" = aws_api_gateway_model.user.name
  }
  
  request_parameters = {
    "method.request.header.X-Correlation-ID" = true
  }
}

# Lambda integration
resource "aws_api_gateway_integration" "create_user" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.users.id
  http_method = aws_api_gateway_method.create_user.http_method
  
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.create_user.invoke_arn
  
  timeout_milliseconds = 29000
  
  request_templates = {
    "application/json" = jsonencode({
      statusCode = 200
    })
  }
}

# Method responses
resource "aws_api_gateway_method_response" "create_user_200" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.users.id
  http_method = aws_api_gateway_method.create_user.http_method
  status_code = "200"
  
  response_models = {
    "application/json" = aws_api_gateway_model.user.name
  }
  
  response_parameters = {
    "method.response.header.X-Correlation-ID" = true
  }
}

resource "aws_api_gateway_method_response" "create_user_400" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.users.id
  http_method = aws_api_gateway_method.create_user.http_method
  status_code = "400"
  
  response_models = {
    "application/json" = aws_api_gateway_model.error.name
  }
}

# Custom authorizer
resource "aws_api_gateway_authorizer" "jwt" {
  name                   = "${var.environment}-jwt-authorizer"
  rest_api_id            = aws_api_gateway_rest_api.main.id
  type                   = "TOKEN"
  authorizer_uri         = aws_lambda_function.authorizer.invoke_arn
  authorizer_credentials = aws_iam_role.api_gateway_authorizer.arn
  identity_source        = "method.request.header.Authorization"
  
  # Cache auth results
  authorizer_result_ttl_in_seconds = 300
}

# Lambda authorizer function
resource "aws_lambda_function" "authorizer" {
  filename         = "authorizer.zip"
  function_name    = "${var.environment}-api-authorizer"
  role            = aws_iam_role.lambda_authorizer.arn
  handler         = "index.handler"
  runtime         = "python3.9"
  timeout         = 5
  memory_size     = 256
  
  environment {
    variables = {
      JWT_SECRET        = aws_secretsmanager_secret_version.jwt_secret.secret_string
      ENVIRONMENT       = var.environment
      USER_POOL_ID      = aws_cognito_user_pool.main.id
      TOKEN_USE         = "access"
    }
  }
}

# API deployment
resource "aws_api_gateway_deployment" "main" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  
  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.users.id,
      aws_api_gateway_method.create_user.id,
      aws_api_gateway_integration.create_user.id,
    ]))
  }
  
  lifecycle {
    create_before_destroy = true
  }
  
  depends_on = [
    aws_api_gateway_method.create_user,
    aws_api_gateway_integration.create_user
  ]
}

# API stages
resource "aws_api_gateway_stage" "prod" {
  deployment_id = aws_api_gateway_deployment.main.id
  rest_api_id   = aws_api_gateway_rest_api.main.id
  stage_name    = "prod"
  
  # Enable logging
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn
    format = jsonencode({
      requestId               = "$context.requestId"
      extendedRequestId       = "$context.extendedRequestId"
      ip                      = "$context.identity.sourceIp"
      caller                  = "$context.identity.caller"
      user                    = "$context.identity.user"
      requestTime             = "$context.requestTime"
      httpMethod              = "$context.httpMethod"
      resourcePath            = "$context.resourcePath"
      status                  = "$context.status"
      protocol                = "$context.protocol"
      responseLength          = "$context.responseLength"
      error                   = "$context.error.message"
      integrationLatency      = "$context.integration.latency"
      integrationStatus       = "$context.integration.status"
      integrationErrorMessage = "$context.integrationErrorMessage"
      authorizerError         = "$context.authorizer.error"
    })
  }
  
  # Caching
  cache_cluster_enabled = true
  cache_cluster_size    = "0.5"
  
  # Throttling
  throttle_burst_limit = 5000
  throttle_rate_limit  = 10000
  
  # X-Ray tracing
  xray_tracing_enabled = true
  
  variables = {
    deployed_at = timestamp()
  }
}

# Method settings for all methods
resource "aws_api_gateway_method_settings" "all" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  stage_name  = aws_api_gateway_stage.prod.stage_name
  method_path = "*/*"
  
  settings = {
    metrics_enabled        = true
    logging_level          = "INFO"
    data_trace_enabled     = true
    throttling_burst_limit = 2000
    throttling_rate_limit  = 1000
    caching_enabled        = false
  }
}

# API key and usage plan
resource "aws_api_gateway_api_key" "main" {
  name = "${var.environment}-api-key"
  
  tags = {
    Environment = var.environment
  }
}

resource "aws_api_gateway_usage_plan" "main" {
  name         = "${var.environment}-usage-plan"
  description  = "Usage plan for API clients"
  
  api_stages {
    api_id = aws_api_gateway_rest_api.main.id
    stage  = aws_api_gateway_stage.prod.stage_name
  }
  
  quota_settings {
    limit  = 10000
    period = "DAY"
  }
  
  throttle_settings {
    rate_limit  = 500
    burst_limit = 1000
  }
}

resource "aws_api_gateway_usage_plan_key" "main" {
  key_id        = aws_api_gateway_api_key.main.id
  key_type      = "API_KEY"
  usage_plan_id = aws_api_gateway_usage_plan.main.id
}

# Custom domain
resource "aws_api_gateway_domain_name" "main" {
  domain_name              = "api.${var.domain_name}"
  regional_certificate_arn = aws_acm_certificate_validation.api.certificate_arn
  
  endpoint_configuration {
    types = ["REGIONAL"]
  }
  
  security_policy = "TLS_1_2"
  
  mutual_tls_authentication {
    truststore_uri = "s3://${aws_s3_bucket.truststore.bucket}/truststore.pem"
  }
}

resource "aws_api_gateway_base_path_mapping" "main" {
  api_id      = aws_api_gateway_rest_api.main.id
  stage_name  = aws_api_gateway_stage.prod.stage_name
  domain_name = aws_api_gateway_domain_name.main.domain_name
  base_path   = "v1"
}

# WAF for API Gateway
resource "aws_wafv2_web_acl_association" "api_gateway" {
  resource_arn = aws_api_gateway_stage.prod.arn
  web_acl_arn  = aws_wafv2_web_acl.api.arn
}
```
## Performance at Scale: Making Applications Fast

Performance isn't just about speed - it's about delivering consistent experiences whether you have 10 or 10 million users. AWS provides tools to optimize every layer of your application.

### The Performance Journey

Most applications follow a predictable performance evolution:

1. **Single Region, Basic Setup**: Works fine for hundreds of users
2. **Caching Added**: Handles thousands without breaking a sweat
3. **Multi-Region Deployment**: Serves millions with low latency globally
4. **Edge Optimization**: Delivers content in milliseconds worldwide

Let's explore each optimization technique and when to apply it.

### CloudFront: Your Global Accelerator

CloudFront is AWS's content delivery network (CDN). Instead of users fetching data from your servers in Virginia, CloudFront caches content at 400+ edge locations worldwide. Users get data from the nearest location, reducing latency from seconds to milliseconds.

**When to Use CloudFront:**
- Static assets (images, CSS, JavaScript) - immediate 10x performance boost
- API responses that don't change frequently
- Video streaming - adaptive bitrate based on user connection
- Global applications - consistent performance worldwide

**Real-world impact**: A news website serving images from S3 in US-East to users in Australia saw 2-second load times. After adding CloudFront, Australian users get 200ms load times from the Sydney edge location.

```hcl
# cloudfront.tf - CloudFront distribution with Lambda@Edge

# S3 bucket for static content
resource "aws_s3_bucket" "static_content" {
  bucket = "${var.environment}-static-content-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "static_content" {
  bucket = aws_s3_bucket.static_content.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "static_content" {
  bucket = aws_s3_bucket.static_content.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.s3.id
    }
  }
}

# CloudFront Origin Access Control
resource "aws_cloudfront_origin_access_control" "main" {
  name                              = "${var.environment}-oac"
  description                       = "Origin Access Control for S3"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# Lambda@Edge functions
data "archive_file" "lambda_edge_viewer_request" {
  type        = "zip"
  source_file = "lambda-edge/viewer-request.js"
  output_path = "lambda-edge-viewer-request.zip"
}

resource "aws_lambda_function" "edge_viewer_request" {
  provider         = aws.us_east_1  # Lambda@Edge must be in us-east-1
  filename         = data.archive_file.lambda_edge_viewer_request.output_path
  function_name    = "${var.environment}-edge-viewer-request"
  role            = aws_iam_role.lambda_edge.arn
  handler         = "viewer-request.handler"
  runtime         = "nodejs18.x"
  timeout         = 5
  memory_size     = 128
  publish         = true
  
  source_code_hash = data.archive_file.lambda_edge_viewer_request.output_base64sha256
}

data "archive_file" "lambda_edge_origin_response" {
  type        = "zip"
  source_file = "lambda-edge/origin-response.js"
  output_path = "lambda-edge-origin-response.zip"
}

resource "aws_lambda_function" "edge_origin_response" {
  provider         = aws.us_east_1
  filename         = data.archive_file.lambda_edge_origin_response.output_path
  function_name    = "${var.environment}-edge-origin-response"
  role            = aws_iam_role.lambda_edge.arn
  handler         = "origin-response.handler"
  runtime         = "nodejs18.x"
  timeout         = 5
  memory_size     = 128
  publish         = true
  
  source_code_hash = data.archive_file.lambda_edge_origin_response.output_base64sha256
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "main" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "${var.environment} CloudFront distribution"
  default_root_object = "index.html"
  aliases             = [var.domain_name, "www.${var.domain_name}"]
  price_class         = "PriceClass_200"  # US, Canada, Europe, Asia
  
  # S3 origin
  origin {
    domain_name              = aws_s3_bucket.static_content.bucket_regional_domain_name
    origin_access_control_id = aws_cloudfront_origin_access_control.main.id
    origin_id                = "S3-${aws_s3_bucket.static_content.id}"
    
    origin_shield {
      enabled              = true
      origin_shield_region = var.aws_region
    }
  }
  
  # ALB origin for dynamic content
  origin {
    domain_name = aws_lb.main.dns_name
    origin_id   = "ALB-${aws_lb.main.id}"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
      
      origin_keepalive_timeout = 60
      origin_read_timeout      = 60
    }
    
    custom_header {
      name  = "X-Origin-Verify"
      value = random_password.origin_verify.result
    }
  }
  
  # Origin group for failover
  origin_group {
    origin_id = "origin-group-1"
    
    failover_criteria {
      status_codes = [500, 502, 503, 504]
    }
    
    member {
      origin_id = "ALB-${aws_lb.main.id}"
    }
    
    member {
      origin_id = "ALB-${aws_lb.secondary.id}"
    }
  }
  
  # Default cache behavior
  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-${aws_s3_bucket.static_content.id}"
    
    forwarded_values {
      query_string = false
      headers      = ["Origin", "Access-Control-Request-Method", "Access-Control-Request-Headers"]
      
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 86400
    max_ttl                = 31536000
    compress               = true
    
    # Lambda@Edge associations
    lambda_function_association {
      event_type   = "viewer-request"
      lambda_arn   = aws_lambda_function.edge_viewer_request.qualified_arn
      include_body = false
    }
    
    lambda_function_association {
      event_type   = "origin-response"
      lambda_arn   = aws_lambda_function.edge_origin_response.qualified_arn
      include_body = false
    }
  }
  
  # Cache behavior for API
  ordered_cache_behavior {
    path_pattern     = "/api/*"
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "origin-group-1"
    
    cache_policy_id            = aws_cloudfront_cache_policy.api.id
    origin_request_policy_id   = aws_cloudfront_origin_request_policy.api.id
    response_headers_policy_id = aws_cloudfront_response_headers_policy.security.id
    
    viewer_protocol_policy = "https-only"
    compress               = true
  }
  
  # Cache behavior for static assets
  ordered_cache_behavior {
    path_pattern     = "/static/*"
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-${aws_s3_bucket.static_content.id}"
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 604800   # 7 days
    max_ttl                = 31536000  # 1 year
    compress               = true
    
    forwarded_values {
      query_string = false
      
      cookies {
        forward = "none"
      }
    }
  }
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate_validation.main.certificate_arn
    minimum_protocol_version = "TLSv1.2_2021"
    ssl_support_method       = "sni-only"
  }
  
  web_acl_id = aws_wafv2_web_acl.cloudfront.arn
  
  logging_config {
    include_cookies = false
    bucket          = aws_s3_bucket.cloudfront_logs.bucket_domain_name
    prefix          = "cloudfront/"
  }
  
  tags = {
    Environment = var.environment
  }
  
  depends_on = [
    aws_s3_bucket_policy.static_content
  ]
}

# Cache policy for API
resource "aws_cloudfront_cache_policy" "api" {
  name        = "${var.environment}-api-cache-policy"
  comment     = "Cache policy for API endpoints"
  default_ttl = 0
  max_ttl     = 3600
  min_ttl     = 0
  
  parameters_in_cache_key_and_forwarded_to_origin {
    enable_accept_encoding_gzip   = true
    enable_accept_encoding_brotli = true
    
    cookies_config {
      cookie_behavior = "all"
    }
    
    headers_config {
      header_behavior = "whitelist"
      headers {
        items = [
          "Authorization",
          "CloudFront-Viewer-Country",
          "CloudFront-Is-Mobile-Viewer",
          "CloudFront-Is-Tablet-Viewer",
          "CloudFront-Is-Desktop-Viewer"
        ]
      }
    }
    
    query_strings_config {
      query_string_behavior = "all"
    }
  }
}

# Origin request policy
resource "aws_cloudfront_origin_request_policy" "api" {
  name    = "${var.environment}-api-origin-request-policy"
  comment = "Origin request policy for API"
  
  cookies_config {
    cookie_behavior = "all"
  }
  
  headers_config {
    header_behavior = "allViewer"
  }
  
  query_strings_config {
    query_string_behavior = "all"
  }
}

# Response headers policy
resource "aws_cloudfront_response_headers_policy" "security" {
  name    = "${var.environment}-security-headers-policy"
  comment = "Security headers policy"
  
  security_headers_config {
    content_type_options {
      override = true
    }
    
    frame_options {
      frame_option = "DENY"
      override     = true
    }
    
    referrer_policy {
      referrer_policy = "strict-origin-when-cross-origin"
      override        = true
    }
    
    xss_protection {
      mode_block = true
      protection = true
      override   = true
    }
    
    strict_transport_security {
      access_control_max_age_sec = 63072000
      include_subdomains         = true
      preload                    = true
      override                   = true
    }
    
    content_security_policy {
      content_security_policy = "default-src 'self'; img-src 'self' data: https:; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';"
      override                = true
    }
  }
  
  cors_config {
    access_control_allow_credentials = true
    
    access_control_allow_headers {
      items = ["*"]
    }
    
    access_control_allow_methods {
      items = ["GET", "HEAD", "PUT", "POST", "DELETE", "OPTIONS"]
    }
    
    access_control_allow_origins {
      items = ["https://${var.domain_name}"]
    }
    
    access_control_max_age_sec = 86400
    origin_override            = true
  }
  
  custom_headers_config {
    items {
      header   = "X-Environment"
      value    = var.environment
      override = false
    }
  }
}

# CloudFront monitoring
resource "aws_cloudwatch_metric_alarm" "4xx_errors" {
  alarm_name          = "${var.environment}-cloudfront-4xx-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "4xxErrorRate"
  namespace           = "AWS/CloudFront"
  period              = "300"
  statistic           = "Average"
  threshold           = "5"
  alarm_description   = "This metric monitors CloudFront 4xx error rate"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    DistributionId = aws_cloudfront_distribution.main.id
  }
}

# Real-time logs
resource "aws_cloudfront_realtime_log_config" "main" {
  name = "${var.environment}-realtime-logs"
  
  endpoint {
    stream_type = "Kinesis"
    
    kinesis_stream_config {
      role_arn   = aws_iam_role.cloudfront_logs.arn
      stream_arn = aws_kinesis_stream.cloudfront_logs.arn
    }
  }
  
  fields = [
    "timestamp",
    "c-ip",
    "sc-status",
    "cs-uri-stem",
    "cs-user-agent",
    "cs-referer",
    "x-edge-location",
    "x-edge-request-id",
    "x-edge-response-result-type",
    "time-taken"
  ]
  
  sampling_rate = 1  # 1% sampling
}
```


## Load Balancing Options

| Type | Use Case | Layer |
|------|----------|-------|
| Application Load Balancer (ALB) | HTTP/HTTPS traffic, path-based routing | Layer 7 |
| Network Load Balancer (NLB) | Ultra-low latency, TCP/UDP | Layer 4 |
| Gateway Load Balancer (GWLB) | Third-party appliances | Layer 3 |
| Classic Load Balancer | Legacy applications | Layer 4/7 |

## See Also

- [AWS Hub](./) - Overview of all AWS documentation
- [Compute Services](compute.html) - EC2 and Lambda in VPC
- [Security](security.html) - Network security and WAF
- [Infrastructure & Operations](infrastructure.html) - VPC IaC templates
- [Networking Fundamentals](../networking.html) - General networking concepts
- [Kubernetes on AWS](../kubernetes/) - EKS networking
