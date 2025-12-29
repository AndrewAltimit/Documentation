---
layout: docs
title: AWS Storage Services
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "hdd"
---

# AWS Storage Services

## Why Storage Services Matter

Every application needs to store data somewhere. The question is: what kind of data, how much, how often do you access it, and how fast does access need to be? AWS offers different storage services optimized for different answers to these questions.

Understanding the three main storage types will help you make the right choice:

| Storage Type | What It Is | AWS Service | Best For |
|--------------|------------|-------------|----------|
| **Object Storage** | Files stored as objects with metadata | S3 | Images, videos, backups, static websites |
| **Block Storage** | Raw storage volumes (like hard drives) | EBS | Databases, boot volumes, high-IOPS workloads |
| **File Storage** | Shared file systems (like NAS) | EFS | Shared content across multiple servers |

Consider the following before choosing: How often will you access this data? Does it need to be shared across servers? What latency can your application tolerate?

---

## Amazon S3 - Object Storage

S3 (Simple Storage Service) stores files (called objects) in buckets. Unlike traditional file systems, S3 is designed for the internet age: accessible from anywhere, virtually unlimited capacity, and extremely durable (99.999999999%, often called "eleven 9s").

**Real-world example**: Netflix stores its entire video library in S3. When you stream a movie, it is delivered from S3 through CloudFront (CDN) to your device.

### When to Use S3

- **Static website hosting**: HTML, CSS, JavaScript, images
- **Application assets**: User uploads, media files, documents
- **Data lakes**: Raw data for analytics pipelines
- **Backups and archives**: Database dumps, log files, compliance data
- **Software distribution**: Application packages, updates

### Choosing the Right S3 Storage Class

S3 offers multiple storage classes at different price points. The key question: how often will you access this data?

| Storage Class | Access Pattern | Retrieval Time | Cost per GB/month* | Best For |
|---------------|----------------|----------------|-------------------|----------|
| **Standard** | Frequent | Instant | ~$0.023 | Active application data |
| **Intelligent-Tiering** | Unknown | Instant | ~$0.023 + monitoring fee | Unpredictable access |
| **Standard-IA** | Monthly | Instant | ~$0.0125 | Backups accessed occasionally |
| **One Zone-IA** | Monthly | Instant | ~$0.01 | Reproducible data, secondary backups |
| **Glacier Instant Retrieval** | Quarterly | Instant | ~$0.004 | Archives needing immediate access |
| **Glacier Flexible Retrieval** | Yearly | 1-12 hours | ~$0.0036 | Compliance archives |
| **Glacier Deep Archive** | Rarely | 12-48 hours | ~$0.00099 | Long-term retention (7+ years) |

*Prices are approximate for US East. Actual prices vary by region.

### S3 Best Practices

**1. Use Lifecycle Policies for Cost Optimization**

Automatically move data to cheaper storage classes as it ages:

```bash
# Example: Move logs to IA after 30 days, Glacier after 90, delete after 1 year
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-bucket --lifecycle-configuration file://lifecycle.json
```

**2. Enable Versioning for Important Data**

Versioning protects against accidental deletions and overwrites:

```bash
aws s3api put-bucket-versioning --bucket my-bucket \
  --versioning-configuration Status=Enabled
```

**3. Enable Encryption by Default**

All new objects are automatically encrypted:

```bash
aws s3api put-bucket-encryption --bucket my-bucket \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
```

**4. Block Public Access (Unless You Need It)**

Most buckets should not be publicly accessible:

```bash
aws s3api put-public-access-block --bucket my-bucket \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
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

---

## Amazon EBS - Block Storage

EBS (Elastic Block Store) provides persistent block storage for EC2 instances. Think of EBS volumes as virtual hard drives that you attach to your servers. Unlike the instance's local storage, EBS volumes persist independently of the EC2 instance lifecycle.

**When to use EBS over S3**: Use EBS when you need a filesystem that your application can mount and use like a regular disk (databases, application data directories, boot volumes). Use S3 for files accessed via HTTP/API.

### Choosing the Right EBS Volume Type

Consider the following: Does your workload need high IOPS (database transactions), high throughput (large file processing), or just general storage?

| Volume Type | Best For | Max IOPS | Max Throughput | Cost |
|-------------|----------|----------|----------------|------|
| **gp3** (General Purpose SSD) | Most workloads, boot volumes | 16,000 | 1,000 MB/s | Lowest SSD |
| **gp2** (General Purpose SSD) | Legacy, burst-based | 16,000 | 250 MB/s | Moderate |
| **io2** (Provisioned IOPS SSD) | Databases requiring consistent IOPS | 256,000 | 4,000 MB/s | Highest |
| **st1** (Throughput HDD) | Big data, log processing | 500 | 500 MB/s | Low |
| **sc1** (Cold HDD) | Infrequently accessed archives | 250 | 250 MB/s | Lowest |

**Recommendation**: Start with gp3 for most workloads. It offers the best price-to-performance ratio and lets you configure IOPS and throughput independently.

### EBS Best Practices

**1. Always Create Snapshots**

Snapshots are incremental backups stored in S3. They protect against data loss:

```bash
aws ec2 create-snapshot --volume-id vol-xxx --description "Daily backup"
```

**2. Encrypt Everything**

Enable encryption for data at rest:

```bash
aws ec2 create-volume --availability-zone us-east-1a \
  --size 100 --volume-type gp3 --encrypted
```

**3. Right-Size Your Volumes**

You can resize volumes without downtime. Start small and grow as needed:

```bash
aws ec2 modify-volume --volume-id vol-xxx --size 200
```

---

## Amazon EFS - Elastic File System

EFS provides scalable, elastic file storage that can be mounted by multiple EC2 instances simultaneously. Unlike EBS, which attaches to a single instance, EFS is a shared filesystem accessible from anywhere in your VPC.

**When to use EFS**: When multiple servers need to read/write the same files. Think shared application code, user uploads, or configuration files that need to be consistent across a fleet of servers.

### EFS vs EBS: Quick Comparison

| Feature | EBS | EFS |
|---------|-----|-----|
| Attachment | Single EC2 instance | Multiple EC2 instances |
| Capacity | Fixed (you choose size) | Automatic (grows/shrinks with usage) |
| Performance | Higher IOPS possible | Lower latency for shared access |
| Cost | Per GB provisioned | Per GB used |
| Use case | Databases, boot volumes | Shared content, home directories |

### When to Use EFS

- **Web serving**: Shared content across multiple web servers
- **Content management**: Shared media files accessed by multiple app servers
- **Big data and analytics**: Shared datasets for processing clusters
- **Development environments**: Shared code repositories and tools
- **Container storage**: Persistent storage for ECS/EKS containers

### Mounting EFS

```bash
# Mount EFS on EC2 (with encryption in transit)
sudo mount -t efs -o tls fs-12345678:/ /mnt/efs

# Add to /etc/fstab for automatic mounting on reboot
# fs-12345678:/ /mnt/efs efs _netdev,tls 0 0
```

---

## Storage Selection Guide

Use this decision tree to choose the right storage service:

**Question 1: Do you need to mount this as a filesystem on EC2?**
- No, accessing via API is fine: **Use S3**
- Yes, I need a mounted drive: Continue to Question 2

**Question 2: Do multiple servers need to access the same files?**
- No, single server only: **Use EBS**
- Yes, shared access needed: **Use EFS**

### Quick Reference by Use Case

| Use Case | Recommended Service | Why |
|----------|---------------------|-----|
| Static website assets | S3 + CloudFront | Global delivery, no servers to manage |
| Database storage | EBS (io2 or gp3) | Low latency, consistent IOPS |
| Application logs | S3 with lifecycle policies | Cost-effective, easy archival |
| Shared web content | EFS | Multiple servers, automatic scaling |
| Backup and disaster recovery | S3 Glacier | Low cost, high durability |
| Container persistent storage | EFS or EBS (via CSI) | Depends on sharing requirements |
| Big data processing | S3 (source) + EBS (compute) | S3 for data lake, EBS for processing nodes |

## See Also

- [AWS Hub](./) - Overview of all AWS documentation
- [Compute Services](compute.html) - EBS with EC2, S3 with Lambda
- [Databases](databases.html) - Database storage considerations
- [Infrastructure & Operations](infrastructure.html) - Storage IaC and cost optimization
- [Security](security.html) - S3 bucket policies and encryption
