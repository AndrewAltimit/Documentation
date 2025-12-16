---
layout: docs
title: AWS Storage Services
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "hdd"
---

# AWS Storage Services

AWS provides storage solutions for every use case - from object storage (S3) to block storage (EBS) and file systems (EFS). This guide covers storage selection, configuration, and optimization.

## Amazon S3 - Object Storage

S3 (Simple Storage Service) stores files (called objects) in buckets. Unlike traditional file systems, S3 is designed for the internet age - accessible from anywhere, virtually unlimited capacity, and extremely durable (99.999999999% - "eleven 9s").

**Real-world example**: Netflix stores its entire video library in S3. When you stream a movie, it's delivered from S3 through CloudFront (CDN) to your device.

### S3 Storage Classes

| Class | Use Case | Cost | Availability |
|-------|----------|------|--------------|
| Standard | Frequently accessed data | $$$ | 99.99% |
| Intelligent-Tiering | Unknown access patterns | $$$ | 99.9% |
| Standard-IA | Infrequent access | $$ | 99.9% |
| One Zone-IA | Reproducible data | $ | 99.5% |
| Glacier Instant | Archive with instant access | $ | 99.9% |
| Glacier Flexible | Archive with hours retrieval | ¢ | 99.99% |
| Glacier Deep Archive | Long-term archive | ¢ | 99.99% |

### S3 Best Practices

```python
import boto3

s3 = boto3.client('s3')

# 1. Use lifecycle policies for cost optimization
lifecycle_config = {
    'Rules': [
        {
            'ID': 'MoveToIAAfter30Days',
            'Status': 'Enabled',
            'Filter': {'Prefix': 'logs/'},
            'Transitions': [
                {'Days': 30, 'StorageClass': 'STANDARD_IA'},
                {'Days': 90, 'StorageClass': 'GLACIER'},
            ],
            'Expiration': {'Days': 365}
        }
    ]
}

s3.put_bucket_lifecycle_configuration(
    Bucket='my-bucket',
    LifecycleConfiguration=lifecycle_config
)

# 2. Enable versioning for important data
s3.put_bucket_versioning(
    Bucket='my-bucket',
    VersioningConfiguration={'Status': 'Enabled'}
)

# 3. Enable server-side encryption by default
s3.put_bucket_encryption(
    Bucket='my-bucket',
    ServerSideEncryptionConfiguration={
        'Rules': [{
            'ApplyServerSideEncryptionByDefault': {
                'SSEAlgorithm': 'aws:kms',
                'KMSMasterKeyID': 'alias/my-key'
            }
        }]
    }
)
```

### Static Website Hosting

S3 can host static websites without servers:

```bash
# Enable static website hosting
aws s3 website s3://my-bucket \
  --index-document index.html \
  --error-document error.html

# Upload website files
aws s3 sync ./dist s3://my-bucket --delete

# Access at: http://my-bucket.s3-website-region.amazonaws.com
```

## Amazon EBS - Block Storage

EBS provides persistent block storage for EC2 instances. Unlike S3, EBS acts like a traditional hard drive attached to your server.

### EBS Volume Types

| Type | Use Case | IOPS | Throughput |
|------|----------|------|------------|
| gp3 (General Purpose) | Most workloads | 16,000 | 1,000 MB/s |
| io2 (Provisioned IOPS) | Databases | 256,000 | 4,000 MB/s |
| st1 (Throughput) | Big data | 500 | 500 MB/s |
| sc1 (Cold HDD) | Archives | 250 | 250 MB/s |

### EBS Best Practices

```bash
# 1. Create snapshot for backup
aws ec2 create-snapshot \
  --volume-id vol-1234567890abcdef0 \
  --description "Daily backup"

# 2. Enable encryption
aws ec2 create-volume \
  --availability-zone us-east-1a \
  --size 100 \
  --volume-type gp3 \
  --encrypted \
  --kms-key-id alias/my-key

# 3. Optimize performance with gp3
aws ec2 modify-volume \
  --volume-id vol-1234567890abcdef0 \
  --iops 10000 \
  --throughput 500
```

## Amazon EFS - Elastic File System

EFS provides scalable, elastic file storage that can be mounted by multiple EC2 instances simultaneously. Perfect for shared content or application data.

### EFS Use Cases

- **Web serving**: Shared content across multiple web servers
- **CMS**: Shared media files for content management
- **Big data**: Shared datasets for analytics clusters
- **Home directories**: Centralized user storage

```bash
# Mount EFS on EC2
sudo mount -t efs -o tls fs-12345678:/ /mnt/efs

# Add to /etc/fstab for persistence
fs-12345678:/ /mnt/efs efs _netdev,tls 0 0
```

## Storage Selection Guide

```
Start
  │
  ├─ Need block storage for EC2?
  │   └─ Yes → EBS
  │       ├─ High IOPS (databases) → io2
  │       ├─ General workloads → gp3
  │       └─ Cold data → sc1
  │
  ├─ Need shared file system?
  │   └─ Yes → EFS
  │       ├─ Linux workloads → EFS
  │       └─ Windows workloads → FSx for Windows
  │
  └─ Need object storage?
      └─ Yes → S3
          ├─ Frequently accessed → Standard
          ├─ Unknown patterns → Intelligent-Tiering
          ├─ Infrequent access → Standard-IA
          └─ Archive → Glacier
```

## See Also

- [AWS Hub](index.html) - Overview of all AWS documentation
- [Compute Services](compute.html) - EBS with EC2, S3 with Lambda
- [Databases](databases.html) - Database storage considerations
- [Infrastructure & Operations](infrastructure.html) - Storage IaC and cost optimization
- [Security](security.html) - S3 bucket policies and encryption
