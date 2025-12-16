---
layout: docs
title: AWS Cloud Services
permalink: /docs/technology/aws/
toc: false
---

# AWS Cloud Services Hub

Think of AWS as a massive technology toolkit in the cloud. Instead of buying and maintaining your own servers, you rent computing power, storage, and dozens of other services from Amazon's data centers around the world. This hub provides comprehensive documentation organized by service category.

<div class="hub-intro">
  <p class="lead">Whether you're deploying your first EC2 instance or architecting multi-region systems, you'll find detailed guides, practical patterns, and real-world examples to help you build on AWS.</p>
</div>

## Quick Navigation

### [Compute Services](compute.html)
EC2 instances, Lambda functions, Auto Scaling, and serverless patterns.
- Virtual server setup and optimization
- Serverless computing with Lambda
- Spot instances and cost savings
- Container basics with ECS/Fargate

### [Storage Services](storage.html)
S3 buckets, EBS volumes, and file systems for every use case.
- Object storage with S3
- Block storage with EBS
- File systems with EFS
- Storage classes and lifecycle policies

### [Database Services](databases.html)
RDS, DynamoDB, and managed database solutions.
- Relational databases with RDS/Aurora
- NoSQL with DynamoDB
- Caching with ElastiCache
- Database performance optimization

### [Networking & Content Delivery](../networking.html)
VPC, CloudFront, API Gateway, and load balancing.
- Virtual private clouds (VPC)
- Content delivery with CloudFront
- API Gateway patterns
- Load balancer configuration

### [Security & Identity](security.html)
IAM, Security Hub, KMS, and compliance.
- Identity and access management
- Encryption with KMS
- Security Hub and compliance
- WAF and network protection

### [Infrastructure & Operations](infrastructure.html)
CloudFormation, monitoring, cost optimization, and troubleshooting.
- Infrastructure as Code (CloudFormation, CDK)
- Monitoring with CloudWatch
- Cost optimization strategies
- Architecture patterns and case studies
- Troubleshooting guide

---

## Getting Started

**New to AWS?** Start with the [Compute Services](compute.html) page which includes a 30-minute crash course that walks you through deploying your first cloud application.

**Setting up security?** The [Security](security.html) guide covers IAM best practices and account security fundamentals.

**Planning architecture?** Check [Infrastructure & Operations](infrastructure.html) for architecture patterns, including case studies from Netflix, Airbnb, and Slack.

## Core Concepts

### Regions and Availability Zones
AWS operates in multiple geographic regions worldwide. Each region contains multiple Availability Zones (AZs) - essentially separate data centers with independent power, cooling, and networking. This geographic distribution is your foundation for building resilient applications.

### The Pay-as-You-Go Model
Unlike traditional IT where you pay upfront for capacity you might not use, AWS charges based on actual consumption. Launch 100 servers for an hour? You pay for 100 server-hours.

### Shared Responsibility Model
AWS secures the infrastructure ("security of the cloud"), while you secure your data and applications ("security in the cloud").

## Learning Path

| Level | Focus | Key Services |
|-------|-------|--------------|
| **Beginner** | Core services, basic deployments | EC2, S3, RDS, IAM |
| **Intermediate** | Scaling, automation, security | Auto Scaling, Lambda, VPC, CloudWatch |
| **Advanced** | Architecture, optimization, IaC | CloudFormation, Step Functions, Cost Explorer |
| **Expert** | Multi-region, enterprise patterns | Organizations, Transit Gateway, Control Tower |

## Quick Reference

- **AWS CLI Cheat Sheet**: See [Infrastructure & Operations](infrastructure.html#cli-reference)
- **Common IAM Policies**: See [Security](security.html#common-policies)
- **Cost Optimization Tips**: See [Infrastructure & Operations](infrastructure.html#cost-optimization)
- **Troubleshooting**: See [Infrastructure & Operations](infrastructure.html#troubleshooting)

## See Also

- [Terraform for AWS](../terraform/) - Infrastructure as Code alternative
- [Kubernetes on AWS](../kubernetes/) - Container orchestration with EKS
- [Docker on AWS](../docker/) - Containerization fundamentals
- [Distributed Systems](../../distributed-systems/) - Architecture patterns
